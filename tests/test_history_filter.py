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


def test_token_search_model_children_equal_update_and_reencode():
    # planning/search_model.py, section 10: next_states[a, o] == update(state, a, o); next_latents equal the latent of
    # the extended history; values and probabilities come from the heads on those latents; chunking changes nothing.
    from ipomdp.models import ObservationHead, RewardHead, TwoHotSymlog, ValueHead
    from ipomdp.planning import TokenSearchModel
    f = _filter()
    torch.manual_seed(1)
    reward, obs_head, value = RewardHead(D, A, 16, 1, 31), ObservationHead(D, A, O, 16, 1), ValueHead(D, 16, 1, 31)
    for net in (reward, value):
        for p in net.parameters():
            p.data.add_(0.1 * torch.randn_like(p))
    codec = TwoHotSymlog(31, 100.0)
    model = TokenSearchModel(f, reward, obs_head, value, codec, A, O, 0.9, chunk=2)
    a, o, o0 = _history(3, 4)
    states = model.initial_states(3, o0)
    for t in range(4):
        states = model.update(states, a[:, t], o[:, t])
    states = torch.cat([states, model.initial_states(1, torch.tensor([5]))])          # mixed lengths (5 and 1)
    with torch.no_grad():
        e = model.expand(states)
        assert e.next_states.shape == (4, A, O, 1 + 2 * f.max_len)
        for i, (aa, oo) in enumerate([(1, 3), (0, 32), (4, 0), (2, 8)]):
            child = model.update(states[i:i + 1], torch.tensor([aa]), torch.tensor([oo]))
            assert torch.equal(e.next_states[i, aa, oo], child[0])
            assert torch.allclose(e.next_latents[i, aa, oo], model.latents(child)[0], atol=1e-5)
            assert torch.allclose(e.next_values[i, aa, oo], model.values(child)[0], atol=1e-4)
        h = model.latents(states)
        probs = torch.softmax(obs_head(h.repeat_interleave(A, 0), torch.eye(A).repeat(4, 1)), -1).view(4, A, O)
        assert torch.allclose(e.observation_probs, probs, atol=1e-6)
        whole = TokenSearchModel(f, reward, obs_head, value, codec, A, O, 0.9, chunk=64).expand(states)
        assert torch.allclose(whole.next_values, e.next_values, atol=1e-5)
