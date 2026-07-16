"""Verification of the from-scratch Muon / MuonClip implementation
(pipeline/muon.py + pipeline/optimizer.py):

  * Newton-Schulz output is (approximately) orthogonal, and batching over a
    leading axis matches per-slice application (the MoE expert-stack case).
  * The Muon update's RMS lands near the consistent-RMS target 0.2 for any
    matrix shape (the property that makes AdamW's LR carry over).
  * Our from-scratch Adam side reproduces optax.adamw step for step.
  * The Muon/AdamW parameter split classifies the real model's params the way
    the Moonlight recipe demands.
  * QK-Clip caps an exploded head's max attention logit at exactly tau and
    leaves under-threshold heads bit-identical.
  * An end-to-end smoke test: a few make_optimizer steps on the tiny model
    reduce the loss and keep every parameter finite.
"""

import numpy as np
import pytest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig
from pipeline import muon as muon_mod
from pipeline.optimizer import _label, apply_qk_clip, make_optimizer

CFG = KimiLinearConfig(
    vocab_size=64, d_model=32, n_layers=4, full_attn_period=4,
    gdn_num_heads=2, gdn_head_k_dim=8, gdn_head_v_dim=8, gdn_chunk_size=16,
    mla_num_q_heads=4, mla_num_kv_heads=2, mla_head_dim=8, max_seq_len=64,
    moe_d_ff=32, moe_n_routed=4, moe_n_shared=1, moe_top_k=2,
    moe_n_groups=1, moe_topk_groups=1)


def rand(seed, *shape):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape),
                       jnp.float32)


# --------------------------------------------------------------------------- #
#  Newton-Schulz orthogonalization.
# --------------------------------------------------------------------------- #
def test_orthogonalize_singular_values_near_one():
    for seed, shape in [(0, (32, 16)), (1, (16, 32)), (2, (24, 24))]:
        o = muon_mod.orthogonalize(rand(seed, *shape))
        sv = np.asarray(jnp.linalg.svd(o, compute_uv=False))
        # The quintic NS coefficients trade tightness for speed: 5 steps land
        # the spectrum only ROUGHLY at 1 (Keller Jordan targets ~(0.5, 1.5)),
        # and near-zero directions lag behind (a square Gaussian's smallest
        # singular value is ~0, and NS(5) cannot fully lift it — by design,
        # that's fine for Muon). Assert that loose contract, not tightness.
        assert sv.max() < 1.5, (shape, sv.max())
        assert 0.6 < sv.mean() < 1.4, (shape, sv.mean())
        assert (sv > 0.4).mean() >= 0.95, (shape, sv)


def test_orthogonalize_batched_matches_per_slice():
    """A stacked [E, m, n] expert tensor must be treated as E independent
    matrices — each slice normalized and iterated on its own."""
    g = rand(3, 5, 12, 8)
    batched = muon_mod.orthogonalize(g)
    per_slice = jnp.stack([muon_mod.orthogonalize(g[e]) for e in range(5)])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(per_slice),
                               rtol=1e-5, atol=1e-6)


def test_orthogonalize_rejects_vectors():
    with pytest.raises(ValueError, match="matrix"):
        muon_mod.orthogonalize(jnp.ones(8))


# --------------------------------------------------------------------------- #
#  Muon update: consistent-RMS scaling.
# --------------------------------------------------------------------------- #
def test_muon_update_rms_matches_adamw_scale():
    """Moonlight's 0.2*sqrt(max(m, n)) scaling: the update RMS should be ~0.2
    regardless of the matrix shape."""
    tx = muon_mod.scale_by_muon()
    for seed, shape in [(0, (64, 16)), (1, (16, 64)), (2, (48, 48))]:
        g = {"w": rand(seed, *shape)}
        updates, _ = tx.update(g, tx.init(g))
        rms = float(jnp.sqrt(jnp.mean(jnp.square(updates["w"]))))
        assert 0.1 < rms < 0.3, (shape, rms)


# --------------------------------------------------------------------------- #
#  The AdamW side must reproduce optax.adamw.
# --------------------------------------------------------------------------- #
def test_adam_side_matches_optax_adamw():
    lr, b1, b2, eps, wd = 3e-3, 0.9, 0.95, 1e-8, 0.1
    ours = optax.chain(
        muon_mod.scale_by_adam(b1, b2, eps),
        muon_mod.add_weight_decay(wd),
        muon_mod.scale_by_lr(lr))
    ref = optax.adamw(lr, b1=b1, b2=b2, eps=eps, weight_decay=wd)

    params = {"a": rand(0, 8, 4), "b": rand(1, 6)}
    s_ours, s_ref = ours.init(params), ref.init(params)
    for step in range(5):
        grads = {"a": rand(10 + step, 8, 4), "b": rand(20 + step, 6)}
        u_ours, s_ours = ours.update(grads, s_ours, params)
        u_ref, s_ref = ref.update(grads, s_ref, params)
        jax.tree.map(
            lambda x, y: np.testing.assert_allclose(
                np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-7),
            u_ours, u_ref)
        params = optax.apply_updates(params, u_ours)


def test_scheduled_lr_is_stepped():
    """scale_by_lr must feed its own step count into an Optax schedule."""
    sched = optax.linear_schedule(1.0, 0.0, transition_steps=4)
    tx = muon_mod.scale_by_lr(sched)
    params = {"w": jnp.ones(3)}
    state = tx.init(params)
    grads = {"w": jnp.ones(3)}
    seen = []
    for _ in range(3):
        updates, state = tx.update(grads, state, params)
        seen.append(float(-updates["w"][0]))
    np.testing.assert_allclose(seen, [1.0, 0.75, 0.5], rtol=1e-6)


# --------------------------------------------------------------------------- #
#  The Muon/AdamW split on the real model.
# --------------------------------------------------------------------------- #
def test_param_split_follows_moonlight_recipe():
    """The project's split rule: every 2D param -> Muon EXCEPT the embedding
    and LM head (Moonlight/Kimi keep both on AdamW), everything else ->
    AdamW. Pin the label of each parameter family so a surprise shape change
    (or a rule regression) fails loudly."""
    model = KimiLinear(CFG, rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    labels, dims = {}, {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        names = [str(getattr(k, a)) for k in path
                 for a in ("key", "name", "idx") if hasattr(k, a)]
        key = "/".join(names)
        labels[key], dims[key] = _label(path, leaf), leaf.ndim

    # The rule itself, over every real param.
    for key, ndim in dims.items():
        exempt = "embed" in key or "lm_head" in key
        expected = "muon" if (ndim == 2 and not exempt) else "adamw"
        assert labels[key] == expected, (key, ndim)

    def label_of(fragment):
        hits = {k: v for k, v in labels.items() if fragment in k}
        assert hits, f"no param path contains {fragment!r}"
        assert len(set(hits.values())) == 1, hits
        return next(iter(hits.values()))

    # Muon: the 2D weight matrices — all Linear kernels (projections, gates,
    # router).
    for frag in ("w_q_uk", "w_dkv", "w_uv_o",
                 "q_proj", "o_norm/gate", "router"):
        assert label_of(frag) == "muon", frag
    # AdamW: the embedding + LM head (2D, but Moonlight/Kimi keep them off
    # Muon — the embedding gradient is row-sparse and orthogonalization would
    # smear a full-RMS update across every vocab row), 1D params (biases,
    # norm gains, the GDN-2 A_log [H] and dt_bias) and 3D params (the stacked
    # MoE experts, the depthwise short-conv kernel). "norm1/weight" not bare
    # "norm": GDN-2's o_norm also CONTAINS the 2D gate Linear above.
    for frag in ("embed", "lm_head", "A_log", "dt_bias", "norm1/weight",
                 "o_norm/norm/weight", "conv", "w_in", "w_out"):
        assert label_of(frag) == "adamw", frag


# --------------------------------------------------------------------------- #
#  QK-Clip.
# --------------------------------------------------------------------------- #
def test_qk_clip_factors_only_touch_exceeding_heads():
    gammas = muon_mod.qk_clip_factors(
        jnp.array([50.0, 100.0, 200.0, -3.0]), tau=100.0)
    np.testing.assert_allclose(np.asarray(gammas), [1.0, 1.0, 0.5, 1.0])


def test_qk_clip_caps_max_logits_at_tau():
    """Inflate the query projection so heads explode, clip, and re-run the
    forward: exceeding heads must land exactly at tau (logits are linear in
    w_q_uk), and heads under tau must be untouched."""
    model = KimiLinear(CFG, rngs=nnx.Rngs(0))
    attn = model.layers[3].token_mixer  # the one MLA layer
    attn.w_q_uk.kernel.set_value(attn.w_q_uk.kernel.get_value() * 300.0)

    ids = jnp.asarray(
        np.random.default_rng(0).integers(0, CFG.vocab_size, size=(2, 32)),
        jnp.int32)
    _, aux = model(ids)
    before = aux["mla_max_logits"][0]  # [Hq]
    tau = float(jnp.sort(before)[before.shape[0] // 2])  # split the heads

    apply_qk_clip(model, aux["mla_max_logits"], tau)
    _, aux2 = model(ids)
    after = np.asarray(aux2["mla_max_logits"][0])

    np.testing.assert_allclose(
        after, np.minimum(np.asarray(before), tau), rtol=1e-4)
    assert (after <= tau * (1 + 1e-4)).all()


def test_qk_clip_row_count_mismatch_raises():
    model = KimiLinear(CFG, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="full-attention"):
        apply_qk_clip(model, jnp.zeros((3, CFG.mla_num_q_heads)), tau=100.0)


def test_muonclip_optimizer_clips_inside_update():
    """MuonClip packaged as ONE optimizer: update(model, grads, max_logits)
    must run QK-Clip after the gradient step. lr=0 zeroes the gradient part,
    so any weight change is the clip — exceeding heads must land exactly at
    tau on the next forward, heads under tau must be untouched."""
    model = KimiLinear(CFG, rngs=nnx.Rngs(0))
    attn = model.layers[3].token_mixer  # the one MLA layer
    attn.w_q_uk.kernel.set_value(attn.w_q_uk.kernel.get_value() * 300.0)

    ids = jnp.asarray(
        np.random.default_rng(0).integers(0, CFG.vocab_size, size=(2, 32)),
        jnp.int32)

    def loss_fn(m):
        logits, aux = m(ids)
        return logits.mean() + aux["aux_loss"], aux

    (_, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    before = aux["mla_max_logits"][0]  # [Hq]
    tau = float(jnp.sort(before)[before.shape[0] // 2])  # split the heads

    optimizer = make_optimizer(model, 0.0, qk_clip_tau=tau, verbose=False)
    optimizer.update(model, grads, max_logits=aux["mla_max_logits"])

    _, aux2 = model(ids)
    after = np.asarray(aux2["mla_max_logits"][0])
    np.testing.assert_allclose(
        after, np.minimum(np.asarray(before), tau), rtol=1e-4)


# --------------------------------------------------------------------------- #
#  End-to-end smoke test: a few real optimizer steps.
# --------------------------------------------------------------------------- #
def test_training_steps_reduce_loss_and_stay_finite():
    model = KimiLinear(CFG, rngs=nnx.Rngs(0))
    optimizer = make_optimizer(model, 1e-2, verbose=False)
    # 33 tokens -> a 32-token input window (a multiple of gdn_chunk_size=16).
    ids = jnp.asarray(
        np.random.default_rng(1).integers(0, CFG.vocab_size, size=(4, 33)),
        jnp.int32)
    batch_in, batch_tgt = ids[:, :-1], ids[:, 1:]

    def loss_fn(m):
        logits, aux = m(batch_in)
        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch_tgt).mean()
        return ce + aux["aux_loss"]

    losses = []
    for _ in range(5):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        losses.append(float(loss))

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], losses  # memorizing one tiny batch
    for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)):
        assert bool(jnp.all(jnp.isfinite(leaf)))
