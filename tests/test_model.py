"""Smoke tests of the full KimiLinear model (kimi_linear_gdn2.py):
forward shapes and aux contract, streaming step vs full forward, generate(),
and the MLA-cache overflow guard.
"""

import numpy as np
import pytest

import flax.nnx as nnx
import jax.numpy as jnp

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig, count_params

CFG = KimiLinearConfig(
    vocab_size=64, d_model=32, n_layers=4, full_attn_period=4,
    gdn_num_heads=2, gdn_head_k_dim=8, gdn_head_v_dim=8, gdn_chunk_size=16,
    mla_num_q_heads=4, mla_num_kv_heads=2, mla_head_dim=8, max_seq_len=64,
    moe_d_ff=32, moe_n_routed=4, moe_n_shared=1, moe_top_k=2,
    moe_n_groups=1, moe_topk_groups=1)


@pytest.fixture(scope="module")
def model():
    return KimiLinear(CFG, rngs=nnx.Rngs(0))


def rand_ids(seed, B=2, L=32):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, CFG.vocab_size, size=(B, L)), jnp.int32)


def test_forward_shapes_and_aux(model):
    ids = rand_ids(0)
    logits, aux = model(ids)
    assert logits.shape == (2, 32, CFG.vocab_size)
    assert logits.dtype == jnp.float32
    assert aux["group_sizes"].shape == (CFG.n_layers, CFG.moe_n_routed)
    assert bool(jnp.all(jnp.isfinite(logits)))
    assert count_params(model) > 0


def test_step_matches_call(model):
    """The streaming path (one prefill step over the whole sequence) must
    reproduce the training forward — GDN-2 caches, MLA cache, and the layer
    dispatch all in agreement."""
    ids = rand_ids(1)
    logits_call, _ = model(ids)
    logits_step, _ = model.step(ids, model.init_cache(ids.shape[0]))
    np.testing.assert_allclose(np.asarray(logits_step), np.asarray(logits_call),
                               rtol=1e-3, atol=1e-4)


def test_generate_shape_and_determinism(model):
    prompt = rand_ids(2, B=1, L=16)
    out1 = model.generate(prompt, max_new_tokens=8)
    out2 = model.generate(prompt, max_new_tokens=8)
    assert out1.shape == (1, 8)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))
    assert bool(jnp.all((out1 >= 0) & (out1 < CFG.vocab_size)))


def test_generate_rejects_undersized_cache(model):
    """An explicit max_len smaller than prompt+continuation must raise instead
    of silently overflowing the preallocated MLA latent buffer."""
    prompt = rand_ids(3, B=1, L=16)
    with pytest.raises(ValueError, match="overflow"):
        model.generate(prompt, max_new_tokens=16, max_len=24)
