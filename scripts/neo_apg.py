"""Adaptive Projected Guidance (and friends) for WebUI Forge Neo.

Merged from three upstream implementations:
  * `AdaptiveProjectedGuidance` (Forge2)             — the method zoo and the UI
  * `ComfyUI-APG_ImYourCFGNow`                       — the correct APG math, adaptive
                                                       momentum, the momentum-buffer
                                                       lifecycle, the guidance limiter
  * `reForge-APGIsYourCFG`                           — the XYZ plot axes

Everything here hangs off `set_model_sampler_cfg_function`
(`backend/sampling/sampling_function.py:308`), which *replaces* the CFG combination —
the only hook that can, since APG is not expressible as a correction added to an
already-combined result.  That slot holds exactly one function, and Forge Neo's built-in
Rescale CFG (`modules/processing_scripts/rescale_cfg.py:75`) also claims it; whichever
script runs last wins, so this one logs a warning when it finds the slot occupied.
"""

import sys
import traceback
from functools import partial
from typing import Any

import gradio as gr
import torch

from lib_apg import core

from modules import script_callbacks, scripts, shared
from modules.processing import logger
from modules.prompt_parser import SdConditioning
from modules.ui_components import InputAccordion

from backend.sampling.condition import compile_conditions
from backend.sampling.sampling_function import calc_cond_uncond_batch

APG = "APG"
FDG = "FDG"
TRASCE = "TraSCE"
METHOD_TWO = "Method two"
NORMAL = "Normal"

METHODS = [APG, FDG, TRASCE, METHOD_TWO, NORMAL] if core.HAS_KORNIA_PYRAMID else [APG, TRASCE, METHOD_TWO, NORMAL]

MAHIRO = "MaHiRo"
NO_POST_CFG = "None"
POST_CFG = [NO_POST_CFG, MAHIRO]

#   name, eta, rescale threshold, momentum, adaptive momentum
PRESETS_BUILTIN = [
    ("SD 1.5", 0.0, 6.5, -0.5, 0.0),
    ("SD 1.5 CFG++", 0.0, 2.5, -0.45, 0.0),
    ("SD 2.1", 0.0, 7.5, -0.75, 0.0),
    ("SDXL", 0.0, 15.0, -0.5, 0.0),
    ("ComfyUI node defaults", 1.0, 15.0, 0.5, 0.18),
]

try:  # optional `apg_presets.py` in the extension root; 4- or 5-tuples
    import apg_presets

    PRESETS = PRESETS_BUILTIN + [tuple(p) + (0.0,) if len(p) == 4 else tuple(p) for p in apg_presets.presets_custom]
except Exception:
    PRESETS = PRESETS_BUILTIN

PRESET_PLACEHOLDER = "(APG presets)"


def _as_bool(value) -> bool:
    """XYZ plot axes hand booleans over as the strings "True"/"False", and `bool("False")`
    is `True`."""

    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


class _PassState:
    """Per-sampling-pass scratch space.

    A new one is built for every `process_before_every_sampling`, so the momentum buffer
    can never survive into the Hires. fix pass or the next image — the failure mode the
    Forge2 original had, where the running average leaked across generations.
    """

    def __init__(self, momentum: float):
        self.momentum = momentum
        self.buffer: "core.MomentumBuffer | None" = None
        self.last_sigma: float = float("-inf")
        self.first_t: float = 0.0
        self.edit_strength: float = 1.0
        self.last_scale: float = 1.0
        self.empty_cond = None

    def reset_buffer(self):
        self.buffer = core.MomentumBuffer(self.momentum)


class APGForNeo(scripts.Script):
    sorting_priority = 11.9

    #   class attributes, not instance: the hook runs long after
    #   `process_before_every_sampling` returned, and `postprocess` is not guaranteed to
    #   see the same instance
    invocations: int = 0
    diagnosed: bool = False
    warned_foreign_cfg: bool = False
    warned_fdg_fallback: bool = False
    active: bool = False

    def title(self):
        return "Adaptive Projected Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    #   ------------------------------------------------------------------ UI

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            method = gr.Radio(
                value=APG,
                choices=METHODS,
                label="CFG method",
                info="how the positive and negative predictions are combined; see the README",
            )

            with gr.Group() as g_apg:
                with gr.Row():
                    eta = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.0, label="eta (contrast)", info="how much of the guidance component parallel to the positive prediction is kept; 0 = drop it, which is what removes the over-saturation")
                    norm_threshold = gr.Slider(minimum=0.0, maximum=20.0, step=0.01, value=8.0, label="rescale threshold", info="cap on the length of the guidance update; model-dependent, 0 = no cap")
                with gr.Row():
                    momentum = gr.Slider(minimum=-1.5, maximum=1.0, step=0.01, value=-0.5, label="momentum", info="carry over the previous steps' guidance; negative (-0.75..-0.25) is the paper's recommendation")
                    adaptive_momentum = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, value=0.18, label="adaptive momentum", info="decay momentum to zero as sampling progresses; 0 = off, 0.18 is the ComfyUI node's default")

            with gr.Group(visible=False) as g_fdg:
                fdg_scale = gr.Textbox(
                    value="*1.1, 1.5",
                    label="Frequency-Decoupled Guidance scales (high frequency first; replaces the CFG scale)",
                    info='two or more numbers, e.g. "7.0, 1.5" or "7.0, 5.0, 1.5"; a "*" prefix multiplies the live CFG scale',
                    max_lines=1,
                    interactive=True,
                )

            with InputAccordion(False, label="Guidance limiter") as limiter_enabled:
                gr.HTML('<p style="opacity:0.7">Outside the sigma window the step runs unguided (CFG 1). Sigmas are in model space — around 14.6 at the start for SD/SDXL, 1.0 for flow-matching models. -1 disables that end.</p>')
                with gr.Row():
                    sigma_start = gr.Slider(minimum=-1.0, maximum=20.0, step=0.01, value=5.42, label="guidance above sigma")
                    sigma_end = gr.Slider(minimum=-1.0, maximum=20.0, step=0.01, value=0.28, label="guidance below sigma")

            with gr.Row():
                icg = gr.Slider(minimum=0.0, maximum=0.2, step=0.001, value=0.0, label="Independent Condition Guidance", info="blend the negative prediction towards noise; recovers diversity")
                icg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.4, label="ICG start", info="fraction of the run")
            with gr.Row():
                cfg_star = gr.Checkbox(value=False, label="CFG *", info="rescale the negative prediction onto the positive one before subtracting")
                tangential_damp = gr.Checkbox(value=False, label="Tangential damping", info="strip the negative's component that the positive knows nothing about")

            post_cfg = gr.Radio(value=NO_POST_CFG, choices=POST_CFG, label="post CFG method")

            with InputAccordion(False, label="CFG schedule + plain-CFG tweaks") as fade_enabled:
                with gr.Row():
                    low_cfg1 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.1, label="CFG 1 until step")
                    max_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.01, value=1.0, label="boost factor")
                with gr.Row():
                    boost_step = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.2, label="CFG boost start step")
                    min_scale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="fade factor")
                with gr.Row():
                    high_step = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.4, label="full boost at step")
                    heuristic = gr.Slider(minimum=0.0, maximum=16.0, step=0.1, value=0.0, label="Heuristic CFG (Normal only)")
                with gr.Row():
                    fade_step = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="CFG fade start step")
                    h_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="... start step for heuristic CFG")
                with gr.Row():
                    zero_step = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.7, label="full fade at step")
                    reinhard = gr.Slider(minimum=0.0, maximum=16.0, step=0.1, value=0.0, label="Reinhard CFG (Normal only)")
                with gr.Row():
                    high_cfg1 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="CFG 1 after step")
                    rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Rescale CFG (Normal only)")
                centre_mean = gr.Checkbox(value=False, label="centre conds to mean")

            with gr.Row():
                apply_to_hr = gr.Checkbox(value=True, label="Apply to the Hires. fix pass")
                preset = gr.Dropdown(
                    value=PRESET_PLACEHOLDER,
                    choices=[p[0] for p in PRESETS],
                    label="",
                    type="index",
                    scale=0,
                    allow_custom_value=True,
                )

        def set_preset(index):
            if index is not None and 0 <= index < len(PRESETS):
                _, p_eta, p_r, p_m, p_am = PRESETS[index]
            else:
                p_eta, p_r, p_m, p_am = 0.0, 8.0, -0.5, 0.0
            return APG, p_eta, p_r, p_m, p_am, PRESET_PLACEHOLDER

        preset.input(
            fn=set_preset,
            inputs=[preset],
            outputs=[method, eta, norm_threshold, momentum, adaptive_momentum, preset],
            show_progress=False,
        )

        method.change(
            fn=lambda chosen: [gr.update(visible=chosen == APG), gr.update(visible=chosen == FDG)],
            inputs=[method],
            outputs=[g_apg, g_fdg],
            show_progress=False,
        )

        self.infotext_fields = [
            (enabled, lambda d: "APG_method" in d),
            (method, "APG_method"),
            (eta, "APG_eta"),
            (norm_threshold, "APG_r"),
            (momentum, "APG_m"),
            (adaptive_momentum, "APG_adaptive_m"),
            (fdg_scale, "APG_FDG_scale"),
            (limiter_enabled, lambda d: "APG_limiter_start" in d),
            (sigma_start, "APG_limiter_start"),
            (sigma_end, "APG_limiter_end"),
            (icg, "APG_ICG"),
            (icg_start, "APG_ICG_start"),
            (cfg_star, "APG_CFGstar"),
            (tangential_damp, "APG_Tangential_Damping"),
            (post_cfg, "APG_post_cfg"),
            (fade_enabled, lambda d: "apg_fade_boostStep" in d),
            (centre_mean, "apg_fade_cntrMean"),
            (boost_step, "apg_fade_boostStep"),
            (high_step, "apg_fade_highStep"),
            (max_scale, "apg_fade_maxScale"),
            (fade_step, "apg_fade_fadeStep"),
            (zero_step, "apg_fade_zeroStep"),
            (min_scale, "apg_fade_minScale"),
            (low_cfg1, "apg_fade_lowCFG1"),
            (high_cfg1, "apg_fade_highCFG1"),
            (reinhard, "apg_fade_reinhard"),
            (rescale, "apg_fade_rescale"),
            (heuristic, "apg_fade_heuristic"),
            (h_start, "apg_fade_hStart"),
        ]

        return [
            enabled,
            method,
            eta,
            norm_threshold,
            momentum,
            adaptive_momentum,
            fdg_scale,
            limiter_enabled,
            sigma_start,
            sigma_end,
            icg,
            icg_start,
            cfg_star,
            tangential_damp,
            post_cfg,
            fade_enabled,
            centre_mean,
            low_cfg1,
            boost_step,
            high_step,
            max_scale,
            fade_step,
            zero_step,
            min_scale,
            high_cfg1,
            heuristic,
            h_start,
            reinhard,
            rescale,
            apply_to_hr,
        ]

    #   ------------------------------------------------------------------ settings

    ARG_NAMES = [
        "enabled",
        "method",
        "eta",
        "norm_threshold",
        "momentum",
        "adaptive_momentum",
        "fdg_scale",
        "limiter_enabled",
        "sigma_start",
        "sigma_end",
        "icg",
        "icg_start",
        "cfg_star",
        "tangential_damp",
        "post_cfg",
        "fade_enabled",
        "centre_mean",
        "low_cfg1",
        "boost_step",
        "high_step",
        "max_scale",
        "fade_step",
        "zero_step",
        "min_scale",
        "high_cfg1",
        "heuristic",
        "h_start",
        "reinhard",
        "rescale",
        "apply_to_hr",
    ]

    @classmethod
    def _settings(cls, p, args) -> dict:
        s = dict(zip(cls.ARG_NAMES, args))

        #   XYZ plot overrides, stashed on `p` by the axis callbacks below
        for field, value in getattr(p, "_apg_xyz", {}).items():
            if field in s:
                s[field] = value

        s["enabled"] = _as_bool(s["enabled"])
        s["method"] = str(s["method"])
        if s["method"] not in METHODS:
            logger.warning(f"APG: method {s['method']!r} is unavailable here, falling back to {NORMAL}")
            s["method"] = NORMAL

        s["fdg_scale"] = str(s["fdg_scale"])
        s["post_cfg"] = str(s["post_cfg"])
        if s["post_cfg"] not in POST_CFG:
            s["post_cfg"] = NO_POST_CFG

        for key in ("eta", "norm_threshold", "momentum", "adaptive_momentum", "sigma_start", "sigma_end", "icg", "icg_start", "low_cfg1", "boost_step", "high_step", "max_scale", "fade_step", "zero_step", "min_scale", "high_cfg1", "heuristic", "h_start", "reinhard", "rescale"):
            s[key] = float(s[key])
        for key in ("limiter_enabled", "cfg_star", "tangential_damp", "fade_enabled", "centre_mean", "apply_to_hr"):
            s[key] = _as_bool(s[key])

        if not s["fade_enabled"]:
            #   a flat schedule: weight 1.0 everywhere, and the three Normal-only knobs off
            s.update(low_cfg1=0.0, boost_step=0.0, high_step=0.0, max_scale=1.0, fade_step=1.0, zero_step=1.0, min_scale=1.0, high_cfg1=1.0, heuristic=0.0, reinhard=0.0, rescale=0.0, centre_mean=False)
        else:
            #   the fade factor is expressed relative to the boosted scale, so it has to be
            #   divided out before the two envelopes are multiplied together
            if s["max_scale"] > 0.0:
                s["min_scale"] = s["min_scale"] / s["max_scale"]

        return s

    #   ------------------------------------------------------------------ plumbing

    def process(self, p, *args, **kwargs):
        cls = APGForNeo

        #   an interrupted run never reaches `postprocess`; don't report its counters
        cls.invocations = 0
        cls.diagnosed = False
        cls.warned_foreign_cfg = False
        cls.warned_fdg_fallback = False

        s = self._settings(p, args)
        cls.active = s["enabled"]
        if not s["enabled"]:
            return

        params: dict[str, Any] = {"APG_method": s["method"]}

        if s["method"] == APG:
            params.update({"APG_eta": s["eta"], "APG_r": s["norm_threshold"], "APG_m": s["momentum"]})
            if s["adaptive_momentum"] > 0.0:
                params["APG_adaptive_m"] = s["adaptive_momentum"]
        elif s["method"] == FDG:
            params["APG_FDG_scale"] = s["fdg_scale"]

        if s["limiter_enabled"]:
            params.update({"APG_limiter_start": s["sigma_start"], "APG_limiter_end": s["sigma_end"]})
        if s["icg"] > 0.0:
            params.update({"APG_ICG": s["icg"], "APG_ICG_start": s["icg_start"]})
        if s["cfg_star"]:
            params["APG_CFGstar"] = True
        if s["tangential_damp"]:
            params["APG_Tangential_Damping"] = True
        if s["post_cfg"] != NO_POST_CFG:
            params["APG_post_cfg"] = s["post_cfg"]

        if s["fade_enabled"]:
            params.update(
                {
                    "apg_fade_cntrMean": s["centre_mean"],
                    "apg_fade_lowCFG1": s["low_cfg1"],
                    "apg_fade_boostStep": s["boost_step"],
                    "apg_fade_highStep": s["high_step"],
                    "apg_fade_maxScale": s["max_scale"],
                    "apg_fade_fadeStep": s["fade_step"],
                    "apg_fade_zeroStep": s["zero_step"],
                    #   undo the division above, so the value pastes back as it was typed
                    "apg_fade_minScale": s["min_scale"] * s["max_scale"],
                    "apg_fade_highCFG1": s["high_cfg1"],
                    "apg_fade_heuristic": s["heuristic"],
                    "apg_fade_hStart": s["h_start"],
                    "apg_fade_reinhard": s["reinhard"],
                    "apg_fade_rescale": s["rescale"],
                }
            )

        p.extra_generation_params.update(params)

    def process_before_every_sampling(self, p, *args, **kwargs):
        s = self._settings(p, args)
        if not s["enabled"]:
            return
        if getattr(p, "is_hr_pass", False) and not s["apply_to_hr"]:
            return

        unet = p.sd_model.forge_objects.unet.clone()

        if "sampler_cfg_function" in unet.model_options and not APGForNeo.warned_foreign_cfg:
            APGForNeo.warned_foreign_cfg = True
            logger.warning("APG replaces the CFG combination outright and has taken over a slot another script was using (Rescale CFG?); that script now does nothing")
        if s["post_cfg"] == MAHIRO and str(getattr(p, "mahiro", False)) == "True":
            logger.warning("APG: MaHiRo is selected here *and* enabled in the built-in CFG section; it will be applied twice")

        state = _PassState(s["momentum"])

        if s["method"] in (TRASCE, METHOD_TWO):
            state.empty_cond = self._empty_conditioning(p)
            if state.empty_cond is None:
                logger.error(f"APG: could not build the empty conditioning {s['method']} needs; falling back to {NORMAL}")
                s = dict(s, method=NORMAL)

        unet.add_sampler_pre_cfg_function(self._make_edit_strength_probe(state))
        unet.set_model_sampler_cfg_function(self._make_cfg_hook(s, state))
        if s["post_cfg"] != NO_POST_CFG:
            unet.set_model_sampler_post_cfg_function(self._make_post_cfg_hook(s, state))

        p.sd_model.forge_objects.unet = unet

    def postprocess(self, p, processed, *args):
        cls = APGForNeo

        if cls.active and cls.invocations == 0:
            logger.warning("APG was enabled but never modified a step; CFG is probably 1.0, or the guidance limiter excluded every step")

        cls.invocations = 0
        cls.diagnosed = False
        cls.warned_foreign_cfg = False
        cls.warned_fdg_fallback = False
        cls.active = False

    #   ------------------------------------------------------------------ hooks

    @staticmethod
    def _empty_conditioning(p):
        """The prediction anchor TraSCE and Method two need: the model's own idea of the
        prompt-free direction, which is *not* the negative prompt."""

        try:
            prompt = SdConditioning([""], is_negative_prompt=False, width=p.width, height=p.height)
            empty_cond = p.sd_model.get_learned_conditioning(prompt)
            if isinstance(empty_cond, list):
                empty_cond = torch.stack(empty_cond)
            return compile_conditions(empty_cond)
        except Exception:
            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def _make_edit_strength_probe(state: _PassState):
        """`sampling_function_inner` multiplies the CFG scale by the summed conditioning
        strengths (prompt composition), and taking over `sampler_cfg_function` skips that
        line.  This hook fires four lines earlier, where the conditioning list is still in
        scope, so the factor can be recovered instead of silently dropped."""

        def probe(model, cond, uncond, x, timestep, model_options):
            try:
                state.edit_strength = float(sum(item["strength"] if "strength" in item else 1 for item in cond))
            except Exception:
                state.edit_strength = 1.0
            return model, cond, uncond, x, timestep, model_options

        return probe

    @staticmethod
    def _progress() -> tuple[int, int]:
        step = int(getattr(shared.state, "sampling_step", 0))
        last = max(int(getattr(shared.state, "sampling_steps", 1)) - 1, 1)
        return step, last

    @classmethod
    def _make_cfg_hook(cls, s: dict, state: _PassState):
        method = s["method"]

        @torch.inference_mode()
        def apg_cfg_function(args):
            x = args["input"]
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            sigma = args["sigma"]
            cond_scale = float(args["cond_scale"])
            model = args["model"]
            options = args["model_options"]

            sigma_f = float(sigma.flatten()[0].item()) if torch.is_tensor(sigma) else float(sigma)
            step, last_step = cls._progress()

            #   before any early return, so it fires exactly once whatever happens next
            if not cls.diagnosed:
                cls.diagnosed = True
                logger.info(f"APG: {method}, cond{tuple(cond.shape)}/{cond.dtype}/{cond.device}, cfg {cond_scale}, sigma {sigma_f:.4f}")

            #   Forge drops the negative pass entirely at CFG 1.0 and leaves a zero tensor
            #   behind; `opts.skip_early_cond` and NGMS also force that mid-run
            if uncond is None or not torch.any(uncond):
                return x - cond

            #   guidance limiter: outside the sigma window, run the step unguided
            if s["limiter_enabled"]:
                if (s["sigma_start"] >= 0.0 and sigma_f > s["sigma_start"]) or (s["sigma_end"] >= 0.0 and sigma_f <= s["sigma_end"]):
                    state.last_scale = 1.0
                    return x - cond

            cls.invocations += 1

            #   fp32 for everything: the norms APG and Difference-style math take overflow
            #   fp16 on 16-channel latents
            xf = x.float()
            c = cond.float()
            u = uncond.float()

            #   Wan-VAE image models (Anima, Krea 2, Qwen) sample on (B, C, 1, H, W).  The
            #   math in `core` reduces over the last three dims, which on that shape is
            #   (T, H, W) — per channel instead of per sample — and kornia's pyramid
            #   refuses 5D outright.  Drop the single-frame axis here and put it back on
            #   the way out; real video (T > 1) is left as it arrives.
            frame_axis = c.ndim == 5 and c.shape[2] == 1
            if frame_axis:
                xf, c, u = xf.squeeze(2), c.squeeze(2), u.squeeze(2)

            def restore(t):
                return t.unsqueeze(2) if frame_axis else t

            if s["centre_mean"]:
                c = core.centre_to_mean(c)
                u = core.centre_to_mean(u)

            weight = core.fade_weight(step, last_step, s["low_cfg1"], s["boost_step"], s["high_step"], s["max_scale"], s["fade_step"], s["zero_step"], s["min_scale"], s["high_cfg1"])
            #   what `sampling_function_inner` would have combined with, had it combined
            scale = max(1.0, cond_scale * weight) * state.edit_strength
            state.last_scale = scale

            if scale > 1.0:
                if s["icg"] > 0.0 and step / last_step >= s["icg_start"]:
                    factor = s["icg"] * (1.0 if method == APG else 0.1) * min(1.0, sigma_f)
                    u = core.independent_condition_guidance(u, factor)
                if s["cfg_star"]:
                    u = core.cfg_star_uncond(c, u)
                if s["tangential_damp"]:
                    try:
                        u = core.tangential_damping(c, u)
                    except Exception:
                        logger.debug(traceback.format_exc())

            if method == APG:
                cls._prepare_momentum(state, s, model, sigma, sigma_f, c)
                denoised = core.normalized_guidance(c, u, scale, state.buffer, s["eta"], s["norm_threshold"])
                return restore((xf - denoised).to(cond.dtype))

            if method == FDG:
                scales = core.parse_fdg_scales(s["fdg_scale"], scale)
                if scales is None:
                    scales = [scale, scale]
                try:
                    denoised = core.laplacian_guidance(c, u, scales)
                except Exception as e:
                    logger.debug(traceback.format_exc())
                    if not cls.warned_fdg_fallback:
                        cls.warned_fdg_fallback = True
                        logger.warning(f"APG: FDG cannot run on a {tuple(c.shape)} latent ({type(e).__name__}); using plain CFG instead")
                    denoised = u + scale * (c - u)
                return restore((xf - denoised).to(cond.dtype))

            if method in (TRASCE, METHOD_TWO):
                if scale <= 1.0:
                    return x - cond
                #   the model still needs the latent in the shape it samples on
                bias, _ = calc_cond_uncond_batch(model, state.empty_cond, None, x, sigma, options)
                bias = bias.float()
                if frame_axis:
                    bias = bias.squeeze(2)
                if method == TRASCE:
                    denoised = core.trasce_guidance(bias, c, u, scale)
                else:
                    denoised = core.method_two_guidance(bias, c, u, scale)
                return restore((xf - denoised).to(cond.dtype))

            #   NORMAL — plain CFG plus the three magnitude tweaks, computed in eps space
            #   (`x - prediction`) because that is the space Rescale CFG's std comparison
            #   was tuned in upstream
            if scale <= 1.0:
                return x - cond

            c_eps = xf - c
            u_eps = xf - u
            noise = c_eps - u_eps

            heuristic = s["heuristic"] * weight
            reinhard = s["reinhard"] * weight

            if heuristic != 0.0 and heuristic != scale and step >= s["h_start"] * last_step:
                noise = core.heuristic_rescale(noise, u_eps, scale, heuristic)

            effective = scale
            if reinhard != 0.0 and reinhard != scale:
                noise, effective = core.reinhard_scale(noise, scale, reinhard)

            denoised = u_eps + effective * noise
            if s["rescale"] > 0.0:
                denoised = core.rescale_cfg(denoised, c_eps, s["rescale"])

            return restore(denoised.to(cond.dtype))

        return apg_cfg_function

    @staticmethod
    def _prepare_momentum(state: _PassState, s: dict, model, sigma, sigma_f: float, cond: torch.Tensor):
        """Momentum-buffer lifecycle, from the ComfyUI node.

        The buffer is a running sum over steps, so it is only meaningful within one
        continuous trajectory.  It is dropped when the latent changes shape (a new batch
        or resolution) and when sigma goes *up*, which is the generic form of the node's
        `t == 999` test: sigma only ever increases at the start of a fresh pass.
        """

        restart = state.buffer is None or sigma_f > state.last_sigma
        if not restart and torch.is_tensor(state.buffer.running_average) and state.buffer.running_average.shape != cond.shape:
            restart = True

        if restart:
            state.reset_buffer()
            state.first_t = 0.0

        state.last_sigma = sigma_f

        #   timestep, not sigma, is what the decay is expressed against; the predictors
        #   disagree wildly on its range (0..999 on eps models, 0..1 on Flux), so the
        #   decay runs on the fraction of *this* pass already sampled instead
        t = sigma_f
        try:
            predictor = getattr(model, "predictor", None)
            if predictor is not None:
                ts = predictor.timestep(sigma if torch.is_tensor(sigma) else torch.tensor([sigma]))
                t = float(ts.flatten()[0].item())
        except Exception:
            pass

        if state.first_t <= 0.0:
            state.first_t = max(t, 1e-6)

        progress = 1.0 - (t / state.first_t)
        state.buffer.momentum = core.adaptive_momentum_scale(s["momentum"], s["adaptive_momentum"], progress)

    @staticmethod
    def _make_post_cfg_hook(s: dict, state: _PassState):
        @torch.inference_mode()
        def apg_post_cfg(args):
            denoised = args["denoised"]
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]

            if uncond is None or not torch.any(uncond):
                return denoised

            if s["post_cfg"] == MAHIRO:
                #   the *effective* scale, so the CFG schedule reaches this too
                scale = state.last_scale if state.last_scale > 0.0 else float(args.get("cond_scale", 1.0))
                return core.mahiro_post_cfg(denoised.float(), cond.float(), uncond.float(), scale).to(denoised.dtype)

            return denoised

        return apg_post_cfg


#   ---------------------------------------------------------------------- XYZ plot


def _set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_apg_xyz"):
        p._apg_xyz = {}
    p._apg_xyz[field] = x


def _make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    def boolean(field):
        return xyz_grid.AxisOption(f"(APG) {field[1]}", str, partial(_set_value, field=field[0]), choices=lambda: ["True", "False"])

    axis = [
        xyz_grid.AxisOption("(APG) Enabled", str, partial(_set_value, field="enabled"), choices=lambda: ["True", "False"]),
        xyz_grid.AxisOption("(APG) Method", str, partial(_set_value, field="method"), choices=lambda: list(METHODS)),
        xyz_grid.AxisOption("(APG) eta", float, partial(_set_value, field="eta")),
        xyz_grid.AxisOption("(APG) Rescale threshold", float, partial(_set_value, field="norm_threshold")),
        xyz_grid.AxisOption("(APG) Momentum", float, partial(_set_value, field="momentum")),
        xyz_grid.AxisOption("(APG) Adaptive momentum", float, partial(_set_value, field="adaptive_momentum")),
        xyz_grid.AxisOption("(APG) FDG scales", str, partial(_set_value, field="fdg_scale")),
        boolean(("limiter_enabled", "Guidance limiter")),
        xyz_grid.AxisOption("(APG) Guidance sigma start", float, partial(_set_value, field="sigma_start")),
        xyz_grid.AxisOption("(APG) Guidance sigma end", float, partial(_set_value, field="sigma_end")),
        xyz_grid.AxisOption("(APG) ICG", float, partial(_set_value, field="icg")),
        xyz_grid.AxisOption("(APG) ICG start", float, partial(_set_value, field="icg_start")),
        boolean(("cfg_star", "CFG *")),
        boolean(("tangential_damp", "Tangential damping")),
        xyz_grid.AxisOption("(APG) post CFG", str, partial(_set_value, field="post_cfg"), choices=lambda: list(POST_CFG)),
    ]

    if not any(x.label.startswith("(APG)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def _on_before_ui():
    try:
        _make_axis_on_xyz_grid()
    except Exception:
        print(f"[-] Adaptive Projected Guidance: xyz_grid error:\n{traceback.format_exc()}", file=sys.stderr)


script_callbacks.on_before_ui(_on_before_ui)
