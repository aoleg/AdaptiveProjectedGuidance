"""Guidance math for the Adaptive Projected Guidance port.

Pure torch — no Forge, no gradio, no global state.  Everything here takes tensors and
returns new tensors; nothing is modified in place, because in Forge the two predictions
handed to a CFG hook are the same objects every other hook (and the caller) sees.

Sources:
  APG            https://arxiv.org/pdf/2410.02416   (Sadat, Hilliges, Weber)
  FDG            https://arxiv.org/pdf/2506.19713   (Sadat, Vontobel, Salehi, Weber)
  TraSCE         https://arxiv.org/abs/2412.07658   (Jain, Kobayashi, Shibuya, et al.)
  CFG*           https://arxiv.org/abs/2503.18137   (via the "CFG-Zero*" rescaling of the
                                                     unconditional prediction)
  Tangential     https://arxiv.org/pdf/2503.18137   (Kwon, Kim, Hsiao, Jeong, Uh)
  damping
"""

import torch
import torch.nn.functional as F

try:
    from kornia.geometry import pyrup
    from kornia.geometry.transform.pyramid import build_laplacian_pyramid

    HAS_KORNIA_PYRAMID = True
except Exception:  # pragma: no cover - depends on the installed kornia
    HAS_KORNIA_PYRAMID = False


#   the dimensions a latent's "image" lives in; guidance vectors are projected and
#   normalised per sample, never across the batch
SPATIAL = [-1, -2, -3]


class MomentumBuffer:
    """Running (exponentially weighted) sum of the guidance direction.

    `momentum` is re-assigned per step when adaptive momentum is on, which is why the
    coefficient lives on the buffer rather than being a constant of the update.
    """

    def __init__(self, momentum: float):
        self.momentum = float(momentum)
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        self.running_average = update_value + self.momentum * self.running_average

    @property
    def is_empty(self) -> bool:
        return not torch.is_tensor(self.running_average)


def adaptive_momentum_scale(momentum: float, adaptive_momentum: float, progress: float) -> float:
    """Decay the momentum coefficient towards zero as sampling progresses.

    From the ComfyUI node (`APG_ImYourCFGNow`), which expresses the decay against the
    discrete SD timestep as `(1000 - t)`.  `progress` here is the fraction of the
    schedule already sampled, so `progress * 1000` is the same quantity on eps models
    and stays meaningful on the flow-matching predictors, whose `timestep()` returns
    values in a completely different range.

    Momentum starts at the user's value and is pulled to exactly 0 partway through;
    `adaptive_momentum` is raised to the fourth power, so the useful range is small
    (0.18 puts the crossing point around 90% of the way through a run).
    """

    if adaptive_momentum <= 0.0 or momentum == 0.0:
        return momentum

    elapsed = 1000.0 * max(0.0, min(1.0, progress))
    decay = abs(momentum) * (adaptive_momentum**4) * elapsed

    if momentum < 0.0:
        return min(0.0, momentum + decay)
    return max(0.0, momentum - decay)


def project(v0: torch.Tensor, v1: torch.Tensor):
    """Split `v0` into the part along `v1` and the part orthogonal to it.

    float64 for the projection is what the paper's reference snippet does; the norms
    involved overflow fp16 on 16-channel latents, so the cast is not optional.
    """

    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=SPATIAL)
    v0_parallel = (v0 * v1).sum(dim=SPATIAL, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: "MomentumBuffer | None" = None,
    eta: float = 0.0,
    norm_threshold: float = 0.0,
):
    """Adaptive Projected Guidance — arXiv 2410.02416, Algorithm 1.

    Three edits to the ordinary CFG update `(cond - uncond)`:
      * momentum   — the update is a running sum over steps rather than this step alone
      * clipping   — its norm is capped at `norm_threshold`, per sample
      * projection — the component parallel to the conditional prediction (the one that
                     inflates saturation and contrast) is scaled by `eta`

    Note the projection is *outside* the `norm_threshold` test: at threshold 0 the
    clipping is skipped but the projection still happens.  The Forge2 original had the
    projection nested inside, which silently turned APG into "no guidance at all" for
    threshold 0 — this follows the ComfyUI node, which follows the paper.
    """

    diff = pred_cond - pred_uncond

    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0.0:
        diff_norm = diff.norm(p=2, dim=SPATIAL, keepdim=True)
        scale_factor = torch.minimum(torch.ones_like(diff), norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel

    return pred_cond + (guidance_scale - 1.0) * normalized_update


#   ---------------------------------------------------------------- FDG


def build_image_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = pyrup(img) + pyramid[i]
    return img


def laplacian_guidance(pred_cond: torch.Tensor, pred_uncond: torch.Tensor, guidance_scale=(1.0, 1.0)):
    """Frequency-Decoupled Guidance — arXiv 2506.19713.

    Both predictions are split into a Laplacian pyramid, CFG is applied at each level
    with its own scale (highest frequency first), and the pyramid is collapsed back.
    High guidance on the detail levels and low guidance on the base level buys the
    sharpness of a high CFG without the colour blow-out, which lives in the low
    frequencies.

    The per-level projection is the paper's, kept here for fidelity; with the default
    parallel weight of 1.0 it reassembles the difference unchanged.
    """

    if not HAS_KORNIA_PYRAMID:
        raise RuntimeError("kornia's build_laplacian_pyramid is unavailable")

    levels = len(guidance_scale)
    pred_cond_pyramid = build_laplacian_pyramid(pred_cond, levels)
    pred_uncond_pyramid = build_laplacian_pyramid(pred_uncond, levels)

    pred_guided_pyramid = []
    for p_cond, p_uncond, scale in zip(pred_cond_pyramid, pred_uncond_pyramid, guidance_scale):
        diff = p_cond - p_uncond
        diff_parallel, diff_orthogonal = project(diff, p_cond)
        diff = diff_parallel + diff_orthogonal
        pred_guided_pyramid.append(p_cond + (scale - 1.0) * diff)

    pred_guided = build_image_from_pyramid(pred_guided_pyramid)
    pred_guided = pred_guided[:, :, : pred_cond.shape[2], : pred_cond.shape[3]]

    return pred_guided.to(pred_cond.dtype)


def parse_fdg_scales(text: str, cond_scale: float):
    """"*1.1, 1.5" -> [cond_scale * 1.1, 1.5].  Highest frequency first.

    Returns None if fewer than two usable levels were given, so the caller can fall
    back to the live CFG scale rather than guess.
    """

    scales = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if token.startswith("*"):
                scales.append(cond_scale * float(token[1:]))
            else:
                scales.append(float(token))
        except ValueError:
            return None

    if len(scales) < 2:
        return None
    return scales


#   ---------------------------------------------------------------- other CFG forms


def trasce_guidance(bias: torch.Tensor, pred_cond: torch.Tensor, pred_uncond: torch.Tensor, guidance_scale: float):
    """TraSCE — arXiv 2412.07658.

    `bias + s * (cond - uncond)` instead of `uncond + s * (cond - uncond)`, where `bias`
    is the prediction for an *empty* prompt.  The trajectory is anchored to the
    unprompted model, so the negative prompt is free to act purely as a direction to
    move away from — which is what makes it able to erase a concept rather than merely
    trade it for whatever the negative happens to imply.  Useless with an empty
    negative prompt, by construction.
    """

    return bias + guidance_scale * (pred_cond - pred_uncond)


def method_two_guidance(bias: torch.Tensor, pred_cond: torch.Tensor, pred_uncond: torch.Tensor, guidance_scale: float):
    """The Forge2 original's "method two": the same empty-prompt anchor, applied as a
    symmetric push away from the average of the empty and negative predictions.  Softer
    than TraSCE at the same scale (the scale is halved internally)."""

    guidance_scale = guidance_scale * 0.5
    return (2.0 * guidance_scale + 1.0) * pred_cond - guidance_scale * (bias + pred_uncond)


def cfg_star_uncond(pred_cond: torch.Tensor, pred_uncond: torch.Tensor):
    """CFG* / CFG-Zero*: rescale the unconditional prediction by its optimal projection
    onto the conditional one before the CFG subtraction, so the two are on a comparable
    scale and the difference is closer to a pure direction."""

    batch = pred_cond.shape[0]
    cond_flat = pred_cond.reshape(batch, -1)
    uncond_flat = pred_uncond.reshape(batch, -1)

    dot_product = torch.sum(cond_flat * uncond_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(uncond_flat**2, dim=1, keepdim=True) + 1e-8
    st_star = dot_product / squared_norm

    return pred_uncond * st_star.reshape(batch, *([1] * (pred_uncond.ndim - 1)))


def tangential_damping(pred_cond: torch.Tensor, pred_uncond: torch.Tensor):
    """Tangential damping — arXiv 2503.18137.

    Project the unconditional prediction onto the leading right-singular direction of
    the {cond, uncond} pair, dropping the second component.  What survives is the part
    of the negative that shares structure with the positive, so the guidance difference
    carries less of the negative's own unrelated content.
    """

    all_noise = torch.stack((pred_cond, pred_uncond), dim=1).to(dtype=torch.float32)
    all_noise = all_noise.reshape(all_noise.size(0), all_noise.size(1), -1)

    _, _, Vh = torch.linalg.svd(all_noise, full_matrices=False)
    Vh_modified = Vh.clone()
    Vh_modified[:, 1] = 0

    noise_null_flat = pred_uncond.reshape(pred_uncond.size(0), 1, -1).to(dtype=torch.float32, device=Vh.device)

    x_Vh = torch.matmul(noise_null_flat, Vh.transpose(-2, -1))
    x_Vh_V = torch.matmul(x_Vh, Vh_modified)

    return x_Vh_V.reshape(*pred_uncond.shape).to(pred_cond.device, dtype=pred_cond.dtype)


def independent_condition_guidance(pred_uncond: torch.Tensor, factor: float):
    """ICG: blend the unconditional prediction towards noise of the same scale.

    Replacing the negative pass with noise makes the guidance direction depend on the
    positive prompt alone; a small factor loosens prompt adherence just enough to
    recover some of the diversity high CFG destroys.
    """

    icg_uncond = torch.randn_like(pred_uncond) * pred_uncond.std()
    return torch.lerp(pred_uncond, icg_uncond, factor)


def centre_to_mean(t: torch.Tensor):
    """Subtract each channel's own mean.  Removes the per-channel DC offset that drives
    the global colour cast at high guidance."""

    return t - t.mean(dim=(-1, -2), keepdim=True)


#   ---------------------------------------------------------------- plain CFG, embellished


def _quantile99(t: torch.Tensor) -> torch.Tensor:
    """`torch.quantile` refuses tensors above ~16.7M elements, which a large Hires latent
    reaches; subsample rather than fail."""

    v = t.flatten().float()
    if v.numel() > 16_000_000:
        v = v[:: (v.numel() // 8_000_000) + 1]
    return torch.quantile(v, 0.99)


def heuristic_rescale(noise_prediction: torch.Tensor, pred_uncond: torch.Tensor, cond_scale: float, heuristic: float):
    """Match the magnitude of a CFG-`cond_scale` result to that of a CFG-`heuristic`
    one, comparing 99th percentiles.  Above the live CFG it sharpens, below it softens,
    without changing the direction of the update."""

    base = pred_uncond + cond_scale * noise_prediction
    heur = pred_uncond + heuristic * noise_prediction

    base_q = _quantile99(base.abs())
    heur_q = _quantile99(heur.abs())

    if base_q == 0.0 or heur_q == 0.0:
        return noise_prediction
    return noise_prediction * (base_q / heur_q).to(noise_prediction.dtype)


def reinhard_scale(noise_prediction: torch.Tensor, cond_scale: float, reinhard: float):
    """Reinhard tonemap of the guidance magnitude (from ComfyUI).  Returns the
    normalised update and the per-element scale to apply to it — the largest guidance
    excursions are compressed, the small ones pass through."""

    multiplier = reinhard / cond_scale
    magnitude = (torch.linalg.vector_norm(noise_prediction, dim=1) + 1e-10)[:, None]
    noise_prediction = noise_prediction / magnitude

    mean = torch.mean(magnitude, dim=(1, 2, 3), keepdim=True)
    std = torch.std(magnitude, dim=(1, 2, 3), keepdim=True)
    top = (std * 3 + mean) * multiplier

    magnitude = magnitude * (1.0 / top)
    new_magnitude = magnitude / (magnitude + 1.0) * top

    return noise_prediction, cond_scale * new_magnitude


def rescale_cfg(denoised: torch.Tensor, pred_cond: torch.Tensor, amount: float):
    """RescaleCFG: pull the standard deviation of the guided result back towards that of
    the conditional prediction, `amount` of the way."""

    ro_pos = torch.std(pred_cond, dim=(1, 2, 3), keepdim=True)
    ro_cfg = torch.std(denoised, dim=(1, 2, 3), keepdim=True)

    if torch.any(ro_pos == 0.0) or torch.any(ro_cfg == 0.0):
        return denoised
    return torch.lerp(denoised, denoised * (ro_pos / ro_cfg), amount)


def mahiro_post_cfg(denoised: torch.Tensor, pred_cond: torch.Tensor, pred_uncond: torch.Tensor, cond_scale: float):
    """MaHiRo (yoinked-h, via ComfyUI): re-weight the guided result against the plain
    conditional prediction by how much the negative disagrees with it."""

    leap = pred_cond * cond_scale
    u_leap = pred_uncond * cond_scale
    merge = (leap + denoised) / 2

    normu = torch.sqrt(u_leap.abs()) * u_leap.sign()
    normm = torch.sqrt(merge.abs()) * merge.sign()
    sim = F.cosine_similarity(normu, normm).mean()
    simsc = 2 * (sim + 1)

    return (simsc * denoised + (4 - simsc) * leap) / 4


#   ---------------------------------------------------------------- CFG schedule


def fade_weight(
    step: int,
    last_step: int,
    low_cfg1: float,
    boost_step: float,
    high_step: float,
    max_scale: float,
    fade_step: float,
    zero_step: float,
    min_scale: float,
    high_cfg1: float,
) -> float:
    """Multiplier applied to the CFG scale at `step`, from the per-step schedule.

    All the step arguments are fractions of the run.  Two envelopes multiplied:
    a *boost* that ramps 1 -> `max_scale` between `boost_step` and `high_step`, and a
    *fade* that ramps 1 -> `min_scale` between `fade_step` and `zero_step`, with hard
    zero regions before `low_cfg1` and after `high_cfg1` (a zero weight means CFG 1,
    i.e. no guidance at all for that step).
    """

    last_step = max(int(last_step), 1)
    low_cfg1 = low_cfg1 * last_step
    boost_step = boost_step * last_step
    high_step = high_step * last_step
    high_cfg1 = high_cfg1 * last_step
    fade_step = fade_step * last_step
    zero_step = zero_step * last_step

    if step < low_cfg1:
        boost = 0.0
    elif step < boost_step:
        boost = 1.0
    elif step < high_step and high_step > boost_step:
        boost = 1.0 + (max_scale - 1.0) * ((step - boost_step) / (high_step - boost_step))
    else:
        boost = max_scale

    if step > high_cfg1:
        fade = 0.0
    else:
        if step < fade_step:
            fade = 1.0
        elif step < zero_step and zero_step > fade_step:
            fade = 1.0 - (step - fade_step) / (zero_step - fade_step)
        else:
            fade = 0.0
        #   0..1 -> min_scale..1
        fade = fade * (1.0 - min_scale) + min_scale

    return boost * fade
