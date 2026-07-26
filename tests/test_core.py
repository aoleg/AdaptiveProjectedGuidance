"""Offline checks for `lib_apg.core` — no Forge, no model, no GPU.

The reference implementations are small enough to transcribe verbatim (see
`_reference_*` below), so the port's math can be diffed against them on random tensors
instead of being eyeballed.  Run with the webui's python:

    venv/Scripts/python.exe tests/test_core.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_apg import core  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def latents(batch=2, channels=16, size=32, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    cond = torch.randn(batch, channels, size, size, generator=g, dtype=dtype)
    uncond = torch.randn(batch, channels, size, size, generator=g, dtype=dtype)
    return cond, uncond


#   --------------------------------------------------------------- references


class _ReferenceMomentumBuffer:
    """ComfyUI-APG_ImYourCFGNow/nodes.py, verbatim."""

    def __init__(self, momentum):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def _reference_project(v0, v1):
    dtype = v0.dtype
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def _reference_normalized_guidance(pred_cond, pred_uncond, guidance_scale, momentum_buffer=None, eta=1.0, norm_threshold=0.0):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = _reference_project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + (guidance_scale - 1) * normalized_update


def _reference_adaptive_momentum(momentum, adaptive_momentum, t):
    """The node's per-step momentum update, against the discrete SD timestep."""

    signal_scale = momentum
    if adaptive_momentum > 0:
        if momentum < 0:
            signal_scale += -momentum * (adaptive_momentum**4) * (1000 - t)
            if signal_scale > 0:
                signal_scale = 0
        else:
            signal_scale -= momentum * (adaptive_momentum**4) * (1000 - t)
            if signal_scale < 0:
                signal_scale = 0
    return signal_scale


#   --------------------------------------------------------------- tests


def test_apg_matches_reference():
    print("APG vs the ComfyUI node")

    for scale in (4.0, 8.0, 12.0, 32.0):
        for eta in (0.0, 0.5, 1.0):
            for threshold in (0.0, 2.5, 15.0):
                cond, uncond = latents(seed=int(scale * 100 + eta * 10 + threshold))
                got = core.normalized_guidance(cond, uncond, scale, None, eta, threshold)
                want = _reference_normalized_guidance(cond, uncond, scale, None, eta, threshold)
                check(f"cfg {scale} eta {eta} threshold {threshold}", close(got, want, 1e-4), f"max diff {(got - want).abs().max():.2e}")


def test_momentum_matches_reference():
    print("momentum buffer, five consecutive steps")

    cond, uncond = latents(seed=7)
    ours = core.MomentumBuffer(-0.5)
    theirs = _ReferenceMomentumBuffer(-0.5)

    ok = True
    for step in range(5):
        c = cond + 0.1 * step
        got = core.normalized_guidance(c, uncond, 7.5, ours, 0.0, 6.5)
        want = _reference_normalized_guidance(c, uncond, 7.5, theirs, 0.0, 6.5)
        ok = ok and close(got, want, 1e-4)
    check("running average tracks the reference", ok)


def test_projection_recovers_plain_cfg():
    print("degenerate cases")

    cond, uncond = latents(seed=3)

    #   eta 1 keeps the whole update, so the projection reassembles the difference and
    #   APG collapses onto ordinary CFG
    got = core.normalized_guidance(cond, uncond, 7.0, None, eta=1.0, norm_threshold=0.0)
    want = uncond + 7.0 * (cond - uncond)
    check("eta=1, threshold=0 == plain CFG", close(got, want, 1e-4), f"max diff {(got - want).abs().max():.2e}")

    #   and this is the bug the Forge2 original had: threshold 0 must NOT disable APG
    guided = core.normalized_guidance(cond, uncond, 7.0, None, eta=0.0, norm_threshold=0.0)
    check("eta=0, threshold=0 still guides", not close(guided, cond, 1e-3) and not close(guided, want, 1e-3))

    #   eta 0 drops the component along the conditional prediction entirely
    update = (guided - cond) / 6.0
    dot = (update * cond).sum(dim=(-1, -2, -3))
    check("eta=0 leaves an orthogonal update", dot.abs().max() < 1e-2, f"max |<update, cond>| {dot.abs().max():.2e}")


def test_norm_threshold_clips():
    print("norm threshold")

    cond, uncond = latents(seed=11)
    threshold = 1.0
    guided = core.normalized_guidance(cond, uncond, 5.0, None, eta=1.0, norm_threshold=threshold)
    update_norm = ((guided - cond) / 4.0).norm(p=2, dim=(-1, -2, -3))
    check("update norm is capped", bool((update_norm <= threshold + 1e-3).all()), f"norms {update_norm.tolist()}")


def test_adaptive_momentum():
    print("adaptive momentum")

    #   progress maps onto the node's (1000 - t) for an eps model starting at t = 999.
    #   The port measures elapsed time as a fraction of *this pass* rather than against a
    #   hard-coded 1000, so it can also serve the flow-matching predictors, whose
    #   `timestep()` returns 0..1.  On an eps model that leaves the two a single timestep
    #   apart, which is the tolerance asserted here.
    for momentum in (-0.75, -0.5, 0.5):
        for adaptive in (0.0, 0.18, 0.5):
            one_step = abs(momentum) * (adaptive**4) + 1e-6
            ok = True
            for t in (999, 750, 500, 250, 0):
                progress = 1.0 - t / 999.0
                got = core.adaptive_momentum_scale(momentum, adaptive, progress)
                want = _reference_adaptive_momentum(momentum, adaptive, t)
                ok = ok and abs(got - want) <= one_step
            check(f"momentum {momentum} adaptive {adaptive}", ok)

    check("decays towards zero, never past it", core.adaptive_momentum_scale(-0.5, 0.5, 1.0) == 0.0)
    check("clamps progress", core.adaptive_momentum_scale(-0.5, 0.18, 0.0) == -0.5)


def test_fdg():
    print("frequency-decoupled guidance")

    if not core.HAS_KORNIA_PYRAMID:
        check("kornia pyramid available", False, "(build_laplacian_pyramid missing)")
        return

    cond, uncond = latents(channels=4, size=64, seed=5)

    #   the same scale at every level is ordinary CFG, up to pyramid reconstruction error
    got = core.laplacian_guidance(cond, uncond, [6.0, 6.0, 6.0])
    want = uncond + 6.0 * (cond - uncond)
    err = (got - want).abs().max()
    check("uniform scales ~ plain CFG", err < 0.05, f"max diff {err:.2e}")

    #   a lower low-frequency scale must actually change the result
    split = core.laplacian_guidance(cond, uncond, [6.0, 1.5])
    check("split scales differ from plain CFG", (split - want).abs().max() > 0.05)
    check("shape preserved", split.shape == cond.shape)

    parsed = core.parse_fdg_scales("*1.1, 1.5", 7.0)
    check("parse: multipliers", len(parsed) == 2 and abs(parsed[0] - 7.7) < 1e-6 and parsed[1] == 1.5)
    check("parse: three levels", core.parse_fdg_scales("7, 5, 1.5", 7.0) == [7.0, 5.0, 1.5])
    check("parse: rejects a single level", core.parse_fdg_scales("7", 7.0) is None)
    check("parse: rejects junk", core.parse_fdg_scales("high, low", 7.0) is None)


def test_other_methods():
    print("TraSCE / method two / CFG* / tangential damping / ICG")

    cond, uncond = latents(seed=13)
    bias, _ = latents(seed=17)

    check("TraSCE anchors on the empty prediction", close(core.trasce_guidance(bias, cond, uncond, 7.0), bias + 7.0 * (cond - uncond)))
    check("method two is softer than TraSCE", (core.method_two_guidance(bias, cond, uncond, 7.0) - cond).abs().mean() < (core.trasce_guidance(bias, cond, uncond, 7.0) - cond).abs().mean())

    starred = core.cfg_star_uncond(cond, uncond)
    residual = (cond - starred).flatten(1)
    check("CFG* leaves a residual orthogonal to uncond", (residual * uncond.flatten(1)).sum(1).abs().max() < 1e-2)
    check("CFG* is idempotent in direction", close(core.cfg_star_uncond(cond, starred), starred, 1e-3))

    damped = core.tangential_damping(cond, uncond)
    check("tangential damping keeps the shape", damped.shape == uncond.shape)
    check("tangential damping is finite", bool(torch.isfinite(damped).all()))
    check("tangential damping moves the negative", not close(damped, uncond, 1e-3))

    torch.manual_seed(0)
    icg = core.independent_condition_guidance(uncond, 0.0)
    check("ICG at factor 0 is a no-op", close(icg, uncond))
    icg = core.independent_condition_guidance(uncond, 0.05)
    check("ICG at factor 0.05 barely moves", (icg - uncond).abs().mean() < uncond.abs().mean())

    centred = core.centre_to_mean(cond)
    check("centring zeroes the per-channel mean", centred.mean(dim=(-1, -2)).abs().max() < 1e-5)
    check("centring is channel-count agnostic", core.centre_to_mean(torch.randn(1, 4, 8, 8)).shape == (1, 4, 8, 8))


def test_plain_cfg_tweaks():
    print("heuristic / reinhard / rescale / MaHiRo")

    cond, uncond = latents(seed=19)
    noise = cond - uncond

    same = core.heuristic_rescale(noise, uncond, 7.0, 7.0)
    check("heuristic at the live scale is a no-op", close(same, noise, 1e-3))
    sharper = core.heuristic_rescale(noise, uncond, 7.0, 12.0)
    check("a higher heuristic shrinks the update", sharper.abs().mean() < noise.abs().mean())

    normalised, scale = core.reinhard_scale(noise, 7.0, 3.0)
    check("reinhard returns a per-element scale", scale.shape[0] == noise.shape[0] and bool(torch.isfinite(scale).all()))
    check("reinhard output is finite", bool(torch.isfinite(normalised).all()))

    denoised = uncond + 7.0 * noise
    check("rescale at 0 is a no-op", close(core.rescale_cfg(denoised, cond, 0.0), denoised))
    rescaled = core.rescale_cfg(denoised, cond, 1.0)
    check("rescale at 1 matches the conditional std", close(rescaled.std(dim=(1, 2, 3)), cond.std(dim=(1, 2, 3)), 1e-3))

    mahiro = core.mahiro_post_cfg(denoised, cond, uncond, 7.0)
    check("MaHiRo is finite and shaped", mahiro.shape == denoised.shape and bool(torch.isfinite(mahiro).all()))


def test_fade_schedule():
    print("CFG schedule")

    flat = dict(low_cfg1=0.0, boost_step=0.0, high_step=0.0, max_scale=1.0, fade_step=1.0, zero_step=1.0, min_scale=1.0, high_cfg1=1.0)
    check("a flat schedule is weight 1 throughout", all(core.fade_weight(step, 20, **flat) == 1.0 for step in range(21)))

    ramp = dict(low_cfg1=0.1, boost_step=0.2, high_step=0.4, max_scale=2.0, fade_step=0.5, zero_step=0.7, min_scale=0.0, high_cfg1=0.8)
    weights = [core.fade_weight(step, 20, **ramp) for step in range(21)]
    check("CFG 1 for the first steps", weights[0] == 0.0 and weights[1] == 0.0)
    check("boosted in the middle", max(weights) == 2.0)
    check("faded to zero at the end", weights[-1] == 0.0)
    check("monotone down after the peak", all(a >= b - 1e-9 for a, b in zip(weights[8:], weights[9:])))
    check("no division by zero when the ramps are degenerate", core.fade_weight(5, 20, 0.0, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5, 1.0) is not None)
    check("survives last_step 0", core.fade_weight(0, 0, **flat) == 1.0)


def test_fp16_safety():
    print("fp16")

    #   the projection sums 262144 products per sample; in fp16 that accumulation is what
    #   goes wrong, long before anything overflows
    cond, uncond = latents(channels=16, size=128, dtype=torch.float16, seed=23)
    exact = (cond.double() * torch.nn.functional.normalize(uncond.double(), dim=(-1, -2, -3))).sum(dim=(-1, -2, -3))

    def dot(dtype):
        c, u = cond.to(dtype), uncond.to(dtype)
        return (c * torch.nn.functional.normalize(u, dim=(-1, -2, -3))).sum(dim=(-1, -2, -3)).double()

    err16 = ((dot(torch.float16) - exact) / exact).abs().max().item()
    err32 = ((dot(torch.float32) - exact) / exact).abs().max().item()
    check("fp16 loses the projection to accumulation error", err16 > 1e-3, f"relative error {err16:.2e}")
    check("fp32 does not", err32 < 1e-5, f"relative error {err32:.2e}")

    #   an L1 norm over the same tensor does overflow outright — nothing here takes one,
    #   but it is why every path in this file works in fp32 or better
    check("fp16 overflows an L1 norm over a 16-channel latent", not bool(torch.isfinite((cond - uncond).norm(p=1, dim=(-1, -2, -3))).all()))

    guided = core.normalized_guidance(cond.float(), uncond.float(), 8.0, core.MomentumBuffer(-0.5), 0.0, 15.0)
    check("APG in fp32 is finite", bool(torch.isfinite(guided).all()))
    check("casting back to fp16 stays finite", bool(torch.isfinite(guided.to(torch.float16)).all()))


def main():
    torch.manual_seed(0)
    for test in (
        test_apg_matches_reference,
        test_momentum_matches_reference,
        test_projection_recovers_plain_cfg,
        test_norm_threshold_clips,
        test_adaptive_momentum,
        test_fdg,
        test_other_methods,
        test_plain_cfg_tweaks,
        test_fade_schedule,
        test_fp16_safety,
    ):
        test()
        print()

    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
