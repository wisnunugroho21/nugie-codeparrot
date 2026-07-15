"""Verification of the NoPE MLA layer (multi_latent_attention/attention.py):
the streaming step() path (prefill + decode against the preallocated latent
cache) must reproduce the full-sequence __call__ exactly.
"""

import numpy as np
import pytest

import flax.nnx as nnx
import jax.numpy as jnp

from multi_latent_attention.attention import GroupedQueryLatentAttention

EMBED, HQ, HKV, DH = 32, 4, 2, 8


def make_attn(seed=0, **kw):
    return GroupedQueryLatentAttention(
        embed_dim=EMBED, num_q_heads=HQ, num_kv_heads=HKV, head_dim=DH,
        rngs=nnx.Rngs(seed), **kw)


def rand_x(seed, B=2, L=24):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(B, L, EMBED)), jnp.float32)


def test_prefill_matches_call():
    attn = make_attn()
    x = rand_x(0)
    y_call = attn(x)
    y_step, cache = attn.step(x, attn.init_cache(x.shape[0], max_len=32))
    np.testing.assert_allclose(np.asarray(y_step), np.asarray(y_call),
                               rtol=1e-4, atol=1e-5)
    assert int(cache.pos) == x.shape[1]


def test_prefill_then_decode_matches_call():
    """Prefill a prompt, then decode token-by-token; every output must equal
    the corresponding row of the full-sequence forward. Also checks that the
    preallocated-but-unfilled cache slots are correctly masked out."""
    attn = make_attn()
    x = rand_x(1, L=16)
    P = 10
    y_full = attn(x)

    # max_len deliberately larger than L: the tail slots stay zero-filled and
    # must not leak into the attention distribution.
    _, cache = attn.step(x[:, :P], attn.init_cache(x.shape[0], max_len=24))
    outs = []
    for t in range(P, x.shape[1]):
        y, cache = attn.step(x[:, t : t + 1], cache)
        outs.append(y)
    np.testing.assert_allclose(
        np.asarray(jnp.concatenate(outs, axis=1)), np.asarray(y_full[:, P:]),
        rtol=1e-4, atol=1e-5)


def test_gqa_head_divisibility_is_validated():
    with pytest.raises(ValueError, match="divisible"):
        GroupedQueryLatentAttention(
            embed_dim=EMBED, num_q_heads=3, num_kv_heads=2, head_dim=DH,
            rngs=nnx.Rngs(0))
