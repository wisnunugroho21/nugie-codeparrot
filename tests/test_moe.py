"""Verification of the grouped-GEMM MoE (multi_latent_attention/moe.py).

  * The dispatched (argsort / ragged_dot / scatter-add) forward equals the
    dense reference path on the same weights — any mismatch is a dispatch bug.
  * group_sizes accounting and the aux-loss-free router-bias update rule.
"""

import numpy as np
import pytest

import flax.nnx as nnx
import jax.numpy as jnp

from multi_latent_attention.moe import GroupedGemmMoE, update_router_bias


def make_moe(seed=0, **kw):
    defaults = dict(d_model=16, d_ff=32, n_routed=8, n_shared=1, top_k=2)
    defaults.update(kw)
    return GroupedGemmMoE(**defaults, rngs=nnx.Rngs(seed))


def rand_x(seed, B=2, L=12, d=16):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(B, L, d)), jnp.float32)


@pytest.mark.parametrize("groups", [(1, 1), (4, 2)])
def test_dispatched_matches_dense(groups):
    n_groups, topk_groups = groups
    moe = make_moe(n_groups=n_groups, topk_groups=topk_groups)
    x = rand_x(1)
    y_dispatch, _ = moe(x)
    y_dense = moe.dense_forward(x)
    np.testing.assert_allclose(np.asarray(y_dispatch), np.asarray(y_dense),
                               rtol=1e-4, atol=1e-5)


def test_group_sizes_account_for_every_assignment():
    moe = make_moe()
    x = rand_x(2, B=3, L=10)
    _, aux = moe(x)
    T = 3 * 10
    assert int(aux["group_sizes"].sum()) == T * moe.top_k
    np.testing.assert_allclose(float(aux["load"].sum()), 1.0, rtol=1e-6)
    assert float(aux["aux_loss"]) >= 0.0


def test_group_limited_routing_respects_groups():
    """With n_groups=4 and topk_groups=1, both selected experts of every token
    must come from a single group of 2."""
    moe = make_moe(n_groups=4, topk_groups=1, top_k=2)
    x = rand_x(3, B=2, L=16)
    top_idx, _, _ = moe._route(x.reshape(-1, 16))
    gsize = moe.E // moe.n_groups
    groups = np.asarray(top_idx) // gsize
    assert (groups[:, 0] == groups[:, 1]).all()


def test_update_router_bias_direction():
    """Under-loaded experts get a positive nudge, over-loaded a negative one."""
    bias = jnp.zeros(4)
    group_sizes = jnp.asarray([10, 2, 2, 2])  # expert 0 over-loaded
    new = update_router_bias(bias, group_sizes, lr=1e-3)
    assert float(new[0]) < 0.0
    assert all(float(new[i]) > 0.0 for i in range(1, 4))
