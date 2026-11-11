# ABSOLUTE PATH: src/ipomdp/models/history_filter.py
# ==============================================================================
# CAUSAL TRANSFORMER OVER THE ACTION-OBSERVATION HISTORY (model.filter=transformer)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why (2026-11-10; Stage B of the 2026-11-09 audit plan):
#    - The recurrent filter z_{t+1} = GRU([a_t, o_{t+1}], z_t) must carry everything a decision needs in one vector
#      updated in place. The measured failure sits at the latents an IMAGINED turn produces: the head misreads "the
#      cell ahead is unvisited" 19-35% of the time and halves P(gold ahead) there (feature_localise.py, Step D),
#      i.e. the recurrent update of a turn moves the latent off the support where the head was fitted.
#    - A causal Transformer reads the whole history at every step: the latent after a turn is computed by attending
#      to the same history tokens as before the turn, plus one new token, so the visited cells stay addressable
#      instead of being re-encoded by a recurrence. It is the history-based information state of POMDP theory
#      (the history is a sufficient statistic; the belief is its compression).
#
# 2. Contract (identical to BeliefFilter.unroll, so heads, probes and every loss are unchanged):
#    - unroll(actions (B, T, A) one-hot, observations (B, T, O) one-hot, initial (B, O) one-hot or None) ->
#      latents (B, T + 1, D): position t reads tokens 0..t and its output is z_t.
#    - Token t = E_a[a_{t-1}] + E_o[o_t] + E_pos[t]; token 0 uses the learned null action, and the null observation
#      when the domain emits no o_0. The input vocabularies are therefore |A| + 1 and |O| + 1 (index |A|, |O| = null).
#    - There is no step(z, a, o): a history filter's state is its history. Search uses token states
#      (planning/search_model.py, TokenSearchModel), the trainer builds them from the stored episodes.
#
# 3. Architecture: pre-norm blocks (RMSNorm, causal multi-head attention, RMSNorm, SwiGLU MLP of width 2D),
#    `layers` blocks, a final RMSNorm. Attention is written out (no fused kernel) so that the keys and values of a
#    parent history can be reused:
#    - encode(tokens (N, L, 2) int64, lengths (N,)) -> (h_last (N, D), keys, values (N, layers, heads, L, D/heads)):
#      the parents' last hidden states and every layer's keys and values; padded positions (>= length) are masked.
#    - append(keys, values, lengths, actions (N, C), observations (N, C)) -> (N, C, D): for each parent, C one-token
#      children at position `length`, each attending to the parent's keys/values (read-only) and to itself. This is
#      the KV cache of one expansion: it is never stored per tree node (16 KB x 5e5 nodes would be 8.6 GB).
#    - The cached path equals a full re-encode of the extended sequence up to float rounding (tested).
# ==============================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import RMSNorm


class _Block(nn.Module):
    """Pre-norm causal attention + SwiGLU MLP, with the attention written out for KV reuse."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}.")
        self.heads, self.head_dim = heads, dim // heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.w12 = nn.Linear(dim, 4 * dim, bias=False)                  # SwiGLU, hidden 2 * dim
        self.w3 = nn.Linear(2 * dim, dim, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        """(N, L, D) -> (N, heads, L, D/heads)."""
        n, length, _ = x.shape
        return x.view(n, length, self.heads, self.head_dim).transpose(1, 2)

    def _mlp(self, x: Tensor) -> Tensor:
        a, b = self.w12(self.norm2(x)).chunk(2, dim=-1)
        return x + self.w3(F.silu(a) * b)

    def forward(self, x: Tensor, key_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x (N, L, D), key_mask (N, L) True = valid -> (x', keys, values), keys/values (N, heads, L, D/heads)."""
        q, k, v = (self._split(t) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        length = x.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & key_mask[:, None, None, :]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = scores.masked_fill(~allowed, -torch.inf).softmax(-1).nan_to_num(0.0)   # fully masked rows (padding) -> 0
        out = (attn @ v).transpose(1, 2).reshape(x.shape)
        x = x + self.proj(out)
        return self._mlp(x), k, v

    def append(self, x: Tensor, keys: Tensor, values: Tensor, key_mask: Tensor) -> Tensor:
        """
        One new token per child. x (N, C, D); keys/values (N, heads, L, D/heads) of the parent; key_mask (N, L).
        Each child attends to the parent's valid keys and to its own key.
        """
        q, k, v = (self._split(t) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))   # (N, heads, C, d)
        parent = (q @ keys.transpose(-1, -2)) / math.sqrt(self.head_dim)                # (N, heads, C, L)
        parent = parent.masked_fill(~key_mask[:, None, None, :], -torch.inf)
        own = (q * k).sum(-1, keepdim=True) / math.sqrt(self.head_dim)                  # (N, heads, C, 1)
        attn = torch.cat([parent, own], dim=-1).softmax(-1)
        out = attn[..., :-1] @ values + attn[..., -1:] * v                              # (N, heads, C, d)
        x = x + self.proj(out.transpose(1, 2).reshape(x.shape))
        return self._mlp(x)


class CausalTransformerFilter(nn.Module):
    """History-token filter with BeliefFilter's unroll contract (module header)."""

    def __init__(self, num_actions: int, num_observations: int, latent_dim: int, layers: int, heads: int, max_len: int):
        super().__init__()
        self.num_actions, self.num_observations, self.latent_dim, self.max_len = num_actions, num_observations, latent_dim, max_len
        self.action_embedding = nn.Embedding(num_actions + 1, latent_dim)              # index |A| = null action
        self.observation_embedding = nn.Embedding(num_observations + 1, latent_dim)    # index |O| = null observation
        self.position_embedding = nn.Embedding(max_len, latent_dim)
        self.blocks = nn.ModuleList(_Block(latent_dim, heads) for _ in range(layers))
        self.norm = RMSNorm(latent_dim)

    @property
    def null_action(self) -> int:
        return self.num_actions

    @property
    def null_observation(self) -> int:
        return self.num_observations

    def _embed(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """tokens (..., 2) int64 [previous action, observation], positions (...) -> (..., D)."""
        if bool((positions >= self.max_len).any()):
            raise ValueError(f"history longer than max_len {self.max_len}.")
        return (self.action_embedding(tokens[..., 0]) + self.observation_embedding(tokens[..., 1])
                + self.position_embedding(positions))

    def history_tokens(self, actions: Tensor, observations: Tensor, initial: Tensor | None) -> Tensor:
        """Index tensors a_t (B, T), o_{t+1} (B, T), o_0 (B,) or None -> tokens (B, T + 1, 2)."""
        batch = actions.shape[0]
        first_obs = (initial if initial is not None
                     else torch.full((batch,), self.null_observation, dtype=torch.long, device=actions.device))
        first = torch.stack([torch.full_like(first_obs, self.null_action), first_obs], dim=-1)[:, None]
        rest = torch.stack([actions, observations], dim=-1)
        return torch.cat([first, rest], dim=1)

    def encode(self, tokens: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            tokens: (N, L, 2) int64; positions >= lengths are padding (any valid index).
            lengths: (N,) int64 in 1..L.

        Returns:
            h_last (N, D) at position length - 1, keys and values (N, layers, heads, L, D/heads).
        """
        n, length, _ = tokens.shape
        positions = torch.arange(length, device=tokens.device).expand(n, -1)
        key_mask = positions < lengths[:, None]
        x = self._embed(tokens, positions)
        keys, values = [], []
        for block in self.blocks:
            x, k, v = block(x, key_mask)
            keys.append(k)
            values.append(v)
        h = self.norm(x)
        return h[torch.arange(n, device=tokens.device), lengths - 1], torch.stack(keys, 1), torch.stack(values, 1)

    def append(self, keys: Tensor, values: Tensor, lengths: Tensor, actions: Tensor, observations: Tensor) -> Tensor:
        """
        Args:
            keys, values: (N, layers, heads, L, D/heads) from encode (read-only).
            lengths: (N,) parent lengths; the children sit at position lengths.
            actions, observations: (N, C) int64, the children's new tokens.

        Returns:
            (N, C, D) hidden states of the children.
        """
        n, c = actions.shape
        key_mask = torch.arange(keys.shape[3], device=keys.device)[None] < lengths[:, None]
        x = self._embed(torch.stack([actions, observations], dim=-1), lengths[:, None].expand(n, c))
        for i, block in enumerate(self.blocks):
            x = block.append(x, keys[:, i], values[:, i], key_mask)
        return self.norm(x)

    def unroll(self, actions: Tensor, observations: Tensor, initial_observation: Tensor | None) -> Tensor:
        """BeliefFilter's contract: one-hot actions (B, T, A), observations (B, T, O), o_0 (B, O) or None -> (B, T + 1, D)."""
        tokens = self.history_tokens(actions.argmax(-1), observations.argmax(-1),
                                     None if initial_observation is None else initial_observation.argmax(-1))
        batch, length, _ = tokens.shape
        positions = torch.arange(length, device=tokens.device).expand(batch, -1)
        x = self._embed(tokens, positions)
        key_mask = torch.ones(batch, length, dtype=torch.bool, device=tokens.device)
        for block in self.blocks:
            x, _, _ = block(x, key_mask)
        return self.norm(x)
