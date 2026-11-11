# ABSOLUTE PATH: tests/test_history_filter.py
"""The causal Transformer history filter (models/history_filter.py): unroll contract, encode, KV-cached append."""

import torch
import torch.nn.functional as F

from ipomdp.models.history_filter import CausalTransformerFilter

A, O, D = 5, 33, 32


def _filter() -> CausalTransformerFilter:
    torch.manual_seed(0)
    return CausalTransformerFilter(A, O, D, layers=2, heads=4, max_len=20).eval()


def _history(batch: int, steps: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, A, (batch, steps), generator=g), torch.randint(0, O, (batch, steps), generator=g),
            torch.randint(0, O, (batch,), generator=g))


def test_unroll_is_causal_and_matches_encode_at_every_prefix():
    f = _filter()
    a, o, o0 = _history(3, 6)
    with torch.no_grad():
        z = f.unroll(F.one_hot(a, A).float(), F.one_hot(o, O).float(), F.one_hot(o0, O).float())
        assert z.shape == (3, 7, D)
        tokens = f.history_tokens(a, o, o0)
        for length in range(1, 8):                                    # padding beyond the prefix is ignored
            h, _, _ = f.encode(tokens, torch.full((3,), length))
            assert torch.allclose(h, z[:, length - 1], atol=1e-5)
        # a later token does not change an earlier latent
        o2 = o.clone()
        o2[:, -1] = (o2[:, -1] + 1) % O
        z2 = f.unroll(F.one_hot(a, A).float(), F.one_hot(o2, O).float(), F.one_hot(o0, O).float())
        assert torch.allclose(z2[:, :-1], z[:, :-1], atol=1e-6) and not torch.allclose(z2[:, -1], z[:, -1])


def test_kv_cache_invariant_under_imagined_turn():
    # The manifest's test: append a TURN token to a parent; the parent's keys/values are untouched, and the child
    # equals a full re-encode of the extended sequence.
    f = _filter()
    a, o, o0 = _history(4, 5)
    tokens = f.history_tokens(a, o, o0)
    lengths = torch.tensor([6, 4, 2, 6])
    with torch.no_grad():
        _, keys, values = f.encode(tokens, lengths)
        k0, v0 = keys.clone(), values.clone()
        new_a = torch.tensor([[1, 2], [1, 0], [2, 4], [0, 1]])
        new_o = torch.tensor([[0, 3], [8, 0], [32, 1], [4, 4]])
        child = f.append(keys, values, lengths, new_a, new_o)
        assert torch.equal(keys, k0) and torch.equal(values, v0)
        assert child.shape == (4, 2, D)
        for i in range(4):
            for c in range(2):
                ext = torch.cat([tokens[i, :lengths[i]], torch.tensor([[new_a[i, c], new_o[i, c]]])])[None]
                h, _, _ = f.encode(ext, torch.tensor([lengths[i] + 1]))
                assert torch.allclose(h[0], child[i, c], atol=1e-5)


def test_no_initial_observation_uses_the_null_token():
    f = _filter()
    a, o, _ = _history(2, 3)
    with torch.no_grad():
        z = f.unroll(F.one_hot(a, A).float(), F.one_hot(o, O).float(), None)
    tokens = f.history_tokens(a, o, None)
    assert int(tokens[0, 0, 0]) == A and int(tokens[0, 0, 1]) == O and z.shape == (2, 4, D)
