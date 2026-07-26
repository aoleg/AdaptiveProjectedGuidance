"""Offline checks for `scripts/neo_apg.py` under stubbed `modules` / `backend` / gradio.

What this is for: the failure modes of a Forge script are mostly *not* in the math.  They
are a `ui()` return list that no longer lines up with the arguments the callbacks unpack,
a hook that returns a value in the wrong space, or state that survives a pass it should
not have.  None of that needs a model to test.

    venv/Scripts/python.exe tests/test_script.py
"""

import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


#   --------------------------------------------------------------- stubs


class _Component:
    def __init__(self, value=None, **kwargs):
        self.value = value
        self.kwargs = kwargs

    def change(self, *a, **k):
        pass

    def input(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Ctx(_Component):
    pass


def _make_gradio():
    gradio = types.ModuleType("gradio")
    for name in ("Slider", "Checkbox", "Radio", "Textbox", "Dropdown", "Number"):
        setattr(gradio, name, lambda value=None, **k: _Component(value, **k))
    gradio.HTML = lambda *a, **k: _Component(None)
    gradio.Row = lambda *a, **k: _Ctx(None)
    gradio.Group = lambda *a, **k: _Ctx(None)
    gradio.Accordion = lambda *a, **k: _Ctx(None)
    gradio.update = lambda **k: k
    return gradio


class _Logger:
    def __init__(self):
        self.messages = []

    def _log(self, level, msg):
        self.messages.append((level, str(msg)))

    def info(self, msg):
        self._log("info", msg)

    def warning(self, msg):
        self._log("warning", msg)

    def error(self, msg):
        self._log("error", msg)

    def debug(self, msg):
        self._log("debug", msg)


LOGGER = _Logger()


class _State:
    sampling_step = 0
    sampling_steps = 20


def _install_stubs():
    sys.modules["gradio"] = _make_gradio()

    scripts_mod = types.ModuleType("modules.scripts")

    class Script:
        AlwaysVisible = "AlwaysVisible"

    scripts_mod.Script = Script
    scripts_mod.AlwaysVisible = "AlwaysVisible"
    scripts_mod.scripts_data = []

    callbacks = types.ModuleType("modules.script_callbacks")
    callbacks.on_before_ui = lambda fn: None

    shared = types.ModuleType("modules.shared")
    shared.state = _State()
    shared.sd_model = types.SimpleNamespace(get_learned_conditioning=lambda prompt: torch.zeros(1, 77, 768))

    processing = types.ModuleType("modules.processing")
    processing.logger = LOGGER

    prompt_parser = types.ModuleType("modules.prompt_parser")

    class SdConditioning(list):
        def __init__(self, prompts, is_negative_prompt=False, width=None, height=None):
            super().__init__(prompts)
            self.is_negative_prompt = is_negative_prompt
            self.width = width
            self.height = height

    prompt_parser.SdConditioning = SdConditioning

    ui_components = types.ModuleType("modules.ui_components")
    ui_components.InputAccordion = lambda value=False, **k: _Ctx(value, **k)

    modules = types.ModuleType("modules")
    modules.scripts = scripts_mod
    modules.script_callbacks = callbacks
    modules.shared = shared
    modules.processing = processing
    modules.prompt_parser = prompt_parser
    modules.ui_components = ui_components

    condition = types.ModuleType("backend.sampling.condition")
    condition.compile_conditions = lambda cond: [{"model_conds": {}, "cross_attn": cond}]

    sampling_function = types.ModuleType("backend.sampling.sampling_function")
    #   the empty-prompt prediction TraSCE anchors on; a recognisable constant
    sampling_function.calc_cond_uncond_batch = lambda model, cond, uncond, x, timestep, options: (torch.full_like(x, 0.25), None)

    backend = types.ModuleType("backend")
    backend.sampling = types.ModuleType("backend.sampling")
    backend.sampling.condition = condition
    backend.sampling.sampling_function = sampling_function

    for name, module in {
        "modules": modules,
        "modules.scripts": scripts_mod,
        "modules.script_callbacks": callbacks,
        "modules.shared": shared,
        "modules.processing": processing,
        "modules.prompt_parser": prompt_parser,
        "modules.ui_components": ui_components,
        "backend": backend,
        "backend.sampling": backend.sampling,
        "backend.sampling.condition": condition,
        "backend.sampling.sampling_function": sampling_function,
    }.items():
        sys.modules[name] = module


_install_stubs()

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import neo_apg  # noqa: E402


#   --------------------------------------------------------------- fakes


class _Predictor:
    """Eps-model timesteps: sigma 14.6 -> 999, sigma 0 -> 0."""

    def timestep(self, sigma):
        return torch.clamp(sigma / 14.6, 0.0, 1.0) * 999.0


class _Unet:
    def __init__(self):
        self.model_options = {}
        self.model = types.SimpleNamespace(predictor=_Predictor())
        self.cfg_function = None
        self.post_cfg_functions = []
        self.pre_cfg_functions = []

    def clone(self):
        n = _Unet()
        n.model_options = dict(self.model_options)
        n.model = self.model
        return n

    def set_model_sampler_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.cfg_function = fn
        self.model_options["sampler_cfg_function"] = fn

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.post_cfg_functions.append(fn)

    def add_sampler_pre_cfg_function(self, fn, ensure_uniqueness=False):
        self.pre_cfg_functions.append(fn)


class _P:
    def __init__(self):
        self.width = 1024
        self.height = 1024
        self.extra_generation_params = {}
        self.is_hr_pass = False
        self.sd_model = types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(unet=_Unet()),
            get_learned_conditioning=lambda prompt: torch.zeros(1, 77, 768),
        )


def defaults():
    """Drive the script with the values its own UI declares, so the two cannot drift."""

    script = neo_apg.APGForNeo()
    components = script.ui(False)
    return script, [c.value for c in components]


def patch(script, args, cond_scale=7.0, hr=False):
    p = _P()
    p.is_hr_pass = hr
    script.process(p, *args)
    script.process_before_every_sampling(p, *args)
    return p, p.sd_model.forge_objects.unet


def cfg_args(unet, cond, uncond, x, sigma, cond_scale):
    return {
        "cond": x - cond,
        "uncond": x - uncond,
        "cond_scale": cond_scale,
        "timestep": sigma,
        "input": x,
        "sigma": sigma,
        "cond_denoised": cond,
        "uncond_denoised": uncond,
        "model": unet.model,
        "model_options": unet.model_options,
    }


def latents(seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 4, 32, 32, generator=g, dtype=dtype)
    cond = torch.randn(1, 4, 32, 32, generator=g, dtype=dtype)
    uncond = torch.randn(1, 4, 32, 32, generator=g, dtype=dtype)
    return x, cond, uncond


def run(unet, x, cond, uncond, sigma, cond_scale=7.0):
    """What `sampling_function_inner` does with the hook's return value."""

    out = unet.cfg_function(cfg_args(unet, cond, uncond, x, torch.tensor([sigma]), cond_scale))
    return x - out


def set_arg(script, args, name, value):
    args = list(args)
    args[script.ARG_NAMES.index(name)] = value
    return args


#   --------------------------------------------------------------- tests


def test_ui_arity():
    print("ui() vs the argument names")

    script, args = defaults()
    check("ui() returns one component per argument name", len(args) == len(script.ARG_NAMES), f"{len(args)} components, {len(script.ARG_NAMES)} names")
    check("the accordion is the first argument", script.ARG_NAMES[0] == "enabled" and args[0] is False)
    check("defaults resolve", script._settings(_P(), args)["method"] == neo_apg.APG)


def test_disabled_does_nothing():
    print("disabled")

    script, args = defaults()
    p, unet = patch(script, args)
    check("no hook registered", unet.cfg_function is None)
    check("no infotext", p.extra_generation_params == {})


def test_normal_reproduces_core_cfg():
    print("Normal == the CFG combination it replaces")

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "method", neo_apg.NORMAL)
    _, unet = patch(script, args)

    x, cond, uncond = latents()
    got = run(unet, x, cond, uncond, 5.0, cond_scale=7.0)
    want = uncond + 7.0 * (cond - uncond)
    check("matches uncond + s * (cond - uncond)", torch.allclose(got, want, atol=1e-4), f"max diff {(got - want).abs().max():.2e}")

    #   AND-composition prompts: core multiplies the scale by the summed strengths, and
    #   taking over `sampler_cfg_function` skips that line unless the probe recovers it
    unet.pre_cfg_functions[0](None, [{"strength": 1.5}], None, x, torch.tensor([5.0]), {})
    got = run(unet, x, cond, uncond, 5.0, cond_scale=7.0)
    want = uncond + 7.0 * 1.5 * (cond - uncond)
    check("edit_strength is recovered from the pre-CFG probe", torch.allclose(got, want, atol=1e-4))


def test_apg_path():
    print("APG")

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "method", neo_apg.APG)
    args = set_arg(script, args, "eta", 1.0)
    args = set_arg(script, args, "norm_threshold", 0.0)
    args = set_arg(script, args, "momentum", 0.0)
    _, unet = patch(script, args)

    x, cond, uncond = latents(seed=2)
    got = run(unet, x, cond, uncond, 5.0, cond_scale=7.0)
    want = uncond + 7.0 * (cond - uncond)
    check("eta 1 / no clip / no momentum == plain CFG", torch.allclose(got, want, atol=1e-3), f"max diff {(got - want).abs().max():.2e}")

    #   and with the paper's settings it must differ, in the direction of less saturation
    args = set_arg(script, args, "eta", 0.0)
    args = set_arg(script, args, "norm_threshold", 8.0)
    args = set_arg(script, args, "momentum", -0.5)
    _, unet = patch(script, args)
    apg = run(unet, x, cond, uncond, 5.0, cond_scale=7.0)
    check("APG differs from plain CFG", not torch.allclose(apg, want, atol=1e-2))
    check("APG output is finite", bool(torch.isfinite(apg).all()))
    check("APG output keeps dtype", apg.dtype == cond.dtype)


def test_momentum_lifecycle():
    print("momentum buffer lifecycle")

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "method", neo_apg.APG)
    args = set_arg(script, args, "momentum", -0.5)
    _, unet = patch(script, args)

    x, cond, uncond = latents(seed=4)
    descending = [14.0, 10.0, 6.0, 3.0, 1.0]
    for sigma in descending:
        run(unet, x, cond, uncond, sigma)

    #   sigma going back up is a new trajectory (Hires. fix, a second image, a restart
    #   sampler); the running average from the previous one must not carry over
    first_of_second_pass = run(unet, x, cond, uncond, 14.0)
    fresh_script, fresh_args = defaults()
    fresh_args = set_arg(fresh_script, fresh_args, "enabled", True)
    fresh_args = set_arg(fresh_script, fresh_args, "method", neo_apg.APG)
    fresh_args = set_arg(fresh_script, fresh_args, "momentum", -0.5)
    _, fresh_unet = patch(fresh_script, fresh_args)
    fresh = run(fresh_unet, x, cond, uncond, 14.0)
    check("sigma going up resets the buffer", torch.allclose(first_of_second_pass, fresh, atol=1e-5), f"max diff {(first_of_second_pass - fresh).abs().max():.2e}")

    #   a resolution change mid-pass (should not happen, but must not crash or mix shapes)
    big_x, big_cond, big_uncond = latents(seed=5)
    big_x, big_cond, big_uncond = (t.repeat(1, 1, 2, 2) for t in (big_x, big_cond, big_uncond))
    out = run(unet, big_x, big_cond, big_uncond, 8.0)
    check("a shape change resets rather than crashes", out.shape == big_cond.shape and bool(torch.isfinite(out).all()))

    #   a fresh patch never inherits the previous pass's buffer
    _, unet2 = patch(script, args)
    check("each sampling pass gets its own buffer", torch.allclose(run(unet2, x, cond, uncond, 14.0), fresh, atol=1e-5))


def test_cfg1_and_limiter():
    print("guards")

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "method", neo_apg.APG)
    _, unet = patch(script, args)

    x, cond, _ = latents(seed=6)

    #   Forge leaves an all-zero tensor behind when it skips the negative pass
    out = run(unet, x, cond, torch.zeros_like(cond), 5.0, cond_scale=1.0)
    #   `x - (x - cond)` is not bit-exact in fp32, hence the tolerance
    check("zero uncond returns the conditional prediction", torch.allclose(out, cond, atol=1e-5))

    before = neo_apg.APGForNeo.invocations
    args = set_arg(script, args, "limiter_enabled", True)
    args = set_arg(script, args, "sigma_start", 5.42)
    args = set_arg(script, args, "sigma_end", 0.28)
    _, unet = patch(script, args)
    _, cond2, uncond2 = latents(seed=7)

    check("above the window: unguided", torch.allclose(run(unet, x, cond2, uncond2, 9.0), cond2, atol=1e-5))
    check("below the window: unguided", torch.allclose(run(unet, x, cond2, uncond2, 0.1), cond2, atol=1e-5))
    inside = run(unet, x, cond2, uncond2, 3.0)
    check("inside the window: guided", not torch.allclose(inside, cond2, atol=1e-3))
    check("limited steps are not counted as invocations", neo_apg.APGForNeo.invocations == before + 1)


def test_fade_schedule_applies():
    print("CFG schedule")

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "method", neo_apg.NORMAL)
    args = set_arg(script, args, "fade_enabled", True)
    args = set_arg(script, args, "low_cfg1", 0.1)
    args = set_arg(script, args, "max_scale", 1.0)
    args = set_arg(script, args, "min_scale", 1.0)
    args = set_arg(script, args, "high_cfg1", 0.8)
    _, unet = patch(script, args)

    x, cond, uncond = latents(seed=8)

    _State.sampling_step = 0
    check("step 0 is unguided ('CFG 1 until step')", torch.allclose(run(unet, x, cond, uncond, 12.0), cond, atol=1e-4))

    _State.sampling_step = 10
    mid = run(unet, x, cond, uncond, 5.0)
    check("mid-run is fully guided", torch.allclose(mid, uncond + 7.0 * (cond - uncond), atol=1e-4))

    _State.sampling_step = 19
    check("past 'CFG 1 after step' is unguided", torch.allclose(run(unet, x, cond, uncond, 0.5), cond, atol=1e-4))

    #   boost — step 8 is past "full boost at step" (0.4) but before the fade starts (0.5)
    _State.sampling_step = 8
    args = set_arg(script, args, "max_scale", 2.0)
    _, unet = patch(script, args)
    boosted = run(unet, x, cond, uncond, 5.0)
    check("the boost factor multiplies the CFG scale", torch.allclose(boosted, uncond + 14.0 * (cond - uncond), atol=1e-4))

    _State.sampling_step = 0
    _State.sampling_steps = 20


def test_other_methods_run():
    print("the remaining methods")

    x, cond, uncond = latents(seed=9)

    for method in neo_apg.METHODS:
        script, args = defaults()
        args = set_arg(script, args, "enabled", True)
        args = set_arg(script, args, "method", method)
        p, unet = patch(script, args)
        out = run(unet, x, cond, uncond, 5.0)
        check(f"{method}: finite, shaped, typed", out.shape == cond.shape and out.dtype == cond.dtype and bool(torch.isfinite(out).all()))
        check(f"{method}: logged to infotext", p.extra_generation_params.get("APG_method") == method)

    #   TraSCE must actually use the empty-prompt anchor the stub returns (0.25)
    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "method", neo_apg.TRASCE)
    _, unet = patch(script, args)
    got = run(unet, x, cond, uncond, 5.0)
    want = torch.full_like(x, 0.25) + 7.0 * (cond - uncond)
    check("TraSCE == empty + s * (cond - uncond)", torch.allclose(got, want, atol=1e-4), f"max diff {(got - want).abs().max():.2e}")


def test_modifiers_do_not_mutate():
    print("the prediction tensors are shared — nothing may be written in place")

    x, cond, uncond = latents(seed=10)
    for name in ("cfg_star", "tangential_damp"):
        script, args = defaults()
        args = set_arg(script, args, "enabled", True)
        args = set_arg(script, args, "method", neo_apg.APG)
        args = set_arg(script, args, name, True)
        _, unet = patch(script, args)

        c_before, u_before = cond.clone(), uncond.clone()
        out = run(unet, x, cond, uncond, 5.0)
        check(f"{name}: leaves cond/uncond untouched", torch.equal(cond, c_before) and torch.equal(uncond, u_before))
        check(f"{name}: changes the result", not torch.allclose(out, uncond + 7.0 * (cond - uncond), atol=1e-3))

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "method", neo_apg.APG)
    args = set_arg(script, args, "icg", 0.05)
    args = set_arg(script, args, "icg_start", 0.0)
    _, unet = patch(script, args)
    c_before, u_before = cond.clone(), uncond.clone()
    run(unet, x, cond, uncond, 5.0)
    check("icg: leaves cond/uncond untouched", torch.equal(cond, c_before) and torch.equal(uncond, u_before))


def test_post_cfg_and_hires():
    print("post CFG, Hires. fix, foreign CFG hook")

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "post_cfg", neo_apg.MAHIRO)
    _, unet = patch(script, args)
    check("MaHiRo registers a post-CFG hook", len(unet.post_cfg_functions) == 1)

    x, cond, uncond = latents(seed=11)
    denoised = run(unet, x, cond, uncond, 5.0)
    post = unet.post_cfg_functions[0]({"denoised": denoised, "cond_denoised": cond, "uncond_denoised": uncond, "cond_scale": 7.0, "input": x, "sigma": torch.tensor([5.0]), "model_options": {}})
    check("MaHiRo changes the result", not torch.allclose(post, denoised, atol=1e-4))
    check("MaHiRo passes a zero uncond through", torch.allclose(unet.post_cfg_functions[0]({"denoised": denoised, "cond_denoised": cond, "uncond_denoised": torch.zeros_like(cond), "cond_scale": 7.0, "input": x, "sigma": torch.tensor([5.0]), "model_options": {}}), denoised))

    args = set_arg(script, args, "apply_to_hr", False)
    _, unet = patch(script, args, hr=True)
    check("Hires. fix opt-out is honoured", unet.cfg_function is None)
    args = set_arg(script, args, "apply_to_hr", True)
    _, unet = patch(script, args, hr=True)
    check("Hires. fix opt-in is honoured", unet.cfg_function is not None)

    #   Rescale CFG already owning the slot must produce a warning, not silence
    LOGGER.messages.clear()
    neo_apg.APGForNeo.warned_foreign_cfg = False
    p = _P()
    p.sd_model.forge_objects.unet.model_options["sampler_cfg_function"] = lambda a: a
    script.process_before_every_sampling(p, *args)
    check("warns when it displaces another CFG hook", any("Rescale CFG" in m for _, m in LOGGER.messages))


def test_xyz_overrides():
    print("XYZ plot overrides")

    script, args = defaults()
    p = _P()
    p._apg_xyz = {"enabled": "True", "method": neo_apg.NORMAL, "eta": 0.3, "cfg_star": "False"}
    s = script._settings(p, args)
    check("string booleans are parsed, not truthy-tested", s["enabled"] is True and s["cfg_star"] is False)
    check("floats pass through", s["eta"] == 0.3)
    check("method passes through", s["method"] == neo_apg.NORMAL)

    p._apg_xyz = {"enabled": "False"}
    check("'False' disables", script._settings(p, args)["enabled"] is False)

    p._apg_xyz = {"enabled": "True", "method": "nonsense"}
    check("an unknown method degrades instead of raising", script._settings(p, args)["method"] == neo_apg.NORMAL)


def test_fp16_round_trip():
    print("fp16 round trip")

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    args = set_arg(script, args, "method", neo_apg.APG)
    _, unet = patch(script, args)

    x, cond, uncond = latents(seed=12, dtype=torch.float16)
    out = run(unet, x, cond, uncond, 5.0)
    check("returns fp16 for fp16 input", out.dtype == torch.float16)
    check("and stays finite", bool(torch.isfinite(out).all()))


def test_never_ran_warning():
    print("diagnostics")

    script, args = defaults()
    args = set_arg(script, args, "enabled", True)
    patch(script, args)
    LOGGER.messages.clear()
    script.postprocess(_P(), None, *args)
    check("warns when it was enabled but never fired", any("never modified a step" in m for _, m in LOGGER.messages))
    check("counters are reset afterwards", neo_apg.APGForNeo.invocations == 0 and neo_apg.APGForNeo.active is False)


def main():
    for test in (
        test_ui_arity,
        test_disabled_does_nothing,
        test_normal_reproduces_core_cfg,
        test_apg_path,
        test_momentum_lifecycle,
        test_cfg1_and_limiter,
        test_fade_schedule_applies,
        test_other_methods_run,
        test_modifiers_do_not_mutate,
        test_post_cfg_and_hires,
        test_xyz_overrides,
        test_fp16_round_trip,
        test_never_ran_warning,
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
