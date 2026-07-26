# Adaptive Projected Guidance — for WebUI Forge Neo

A collection of alternatives to the CFG scale, for [WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

CFG (the "CFG Scale" slider) works by taking the model's prediction for your prompt,
subtracting its prediction for the negative prompt, and pushing hard in that direction.
It is a blunt instrument: turning it up gets you a picture that follows the prompt more
closely *and* is more saturated, more contrasty, more fried, and less varied. Everything
in this extension is a different answer to the same question — **how do you keep the
prompt adherence without the frying?**

Everything here replaces the usual CFG maths; one method is active at a time, with
optional modifiers on top.

## Install

Settings → Extensions → Install from URL, or clone into `extensions/`:

```bash
git clone https://github.com/aoleg/AdaptiveProjectedGuidance extensions/AdaptiveProjectedGuidance
```

Restart the webui. The panel appears as **Adaptive Projected Guidance** in txt2img and
img2img. Nothing happens until you tick it.

---

## The methods

### APG — Adaptive Projected Guidance

[arXiv:2410.02416](https://arxiv.org/pdf/2410.02416) — *Eliminating oversaturation and
artifacts of high guidance scales in diffusion models*, Sadat, Hilliges & Weber.

The paper's observation is that the push CFG applies has two parts. One part points
*along* what the model already predicted — it just makes everything more of what it
already is, and that is where the blown-out colour and crushed contrast come from. The
other part points *sideways*, and that is the part that actually moves the image towards
your prompt. APG separates the two and lets you keep the sideways push while turning the
along-the-prompt push down or off.

Two more refinements come with it: the push is capped in length so no single step can
lurch, and it is averaged with previous steps so the trajectory stays smooth.

**What you'll see:** you can run a much higher CFG scale than usual without the image
going neon and plastic. At normal CFG it reads as calmer colour, more retained texture in
skin and fabric, and slightly softer contrast. It does not make a bad prompt work better.

| Control | What it does |
|---|---|
| **eta (contrast)** | How much of the along-the-prompt push to keep. `0` drops it entirely (the paper's setting, and the strongest de-saturation). `1` keeps all of it, which makes APG behave exactly like ordinary CFG. Negative values push the other way and flatten contrast further. |
| **rescale threshold** | Cap on how far a single step may push. Very model-dependent; ~6.5 for SD 1.5, ~15 for SDXL. `0` removes the cap. Too low and the image goes flat and undercooked. |
| **momentum** | Blends this step's push with the running total of previous steps. Negative values (the paper recommends **-0.75 to -0.25**) subtract the recent history, which counteracts the drift that builds up over a run. Positive values reinforce it. |
| **adaptive momentum** | Fades momentum out to zero as the run progresses, so the early steps get the smoothing and the late detail steps are left alone. `0` is off; `0.18` (the ComfyUI node's default) crosses zero at roughly 90% of the way through. The value is raised to the fourth power internally, so small changes matter a lot. |

The **(APG presets)** dropdown fills in the three main values with settings known to work
for a given model family.

### FDG — Frequency-Decoupled Guidance

[arXiv:2506.19713](https://arxiv.org/pdf/2506.19713) — *Frequency-decoupled guidance for
diffusion models*, Sadat, Vontobel, Salehi & Weber.

The over-saturation problem lives in the *low* frequencies of an image — the broad
colour and brightness — while the sharpness and prompt detail you actually want from high
CFG live in the *high* frequencies. FDG splits both predictions into a stack of frequency
bands, applies a **different CFG scale to each band**, and puts the image back together.

**What you'll see:** the detail and prompt adherence of a high CFG with the colour balance
of a low one. This is the most immediately useful method here if your complaint about CFG
is "it looks fried".

Enter the scales high-frequency first: `7.0, 1.5` means detail bands at CFG 7 and the
broad colour at CFG 1.5. More than two levels is allowed — `7.0, 5.0, 1.5` for high,
middle and low. A `*` prefix means "multiply the live CFG scale", so `*1.1, 1.5` tracks
whatever the CFG Scale slider says. **These replace the CFG scale**; the slider itself
then only matters through the `*` entries.

Needs `kornia` with Laplacian pyramid support (Forge Neo ships a suitable version). If it
is missing, the FDG option is hidden rather than broken.

### TraSCE

[arXiv:2412.07658](https://arxiv.org/abs/2412.07658) — *Trajectory steering for concept
erasure*.

Ordinary CFG builds the image *starting from* the negative prompt's prediction and walking
away from it. That sounds right but has an odd consequence: whatever the negative prompt
implies but doesn't name still seeps into the result, because it is the foundation being
built on. TraSCE swaps that foundation for the model's prediction for an **empty** prompt,
and uses the negative purely as a direction to walk away from.

**What you'll see:** negative prompts that actually remove things. If you have been
fighting a concept that refuses to leave — a watermark style, an artist's look, a body
feature — this is the method to try. With an *empty* negative prompt it reduces exactly to
ordinary CFG, since the anchor and the negative become the same prediction — so it is only
worth selecting when you actually have something in the negative. It costs one extra model
evaluation per step (roughly 50% slower).

**Method two** is a middle ground between TraSCE and ordinary CFG. Where TraSCE builds
entirely from the empty-prompt prediction, Method two builds from the point halfway
between the empty prediction and the negative one — and pushes one scale step harder
(`midpoint + (CFG + 1) × (positive − midpoint)`). It keeps some of TraSCE's ability to
erase while staying closer to the composition ordinary CFG would have produced. Unlike
TraSCE it still does something with an empty negative prompt, where it reduces to plain
CFG at scale + 1. It costs the same extra model evaluation per step.

### Normal

Ordinary CFG, unchanged — there so you can use the schedule and the tweaks below without
one of the alternative guidance methods on top.

---

## Modifiers

These stack on top of whichever method is selected.

### CFG ★

Before subtracting the negative prediction, rescale it so it sits on the same scale as the
positive one. Early in a run the two predictions can differ in magnitude enough that the
subtraction is dominated by that difference rather than by any real disagreement about
content. Cheap, subtle; generally improves structure in the first few steps.

### Tangential damping

[arXiv:2503.18137](https://arxiv.org/pdf/2503.18137) — Kwon, Kim, Hsiao, Jeong & Uh.

Keeps only the part of the negative prediction that shares structure with the positive one
and discards the rest. The guidance difference then carries less of the negative prompt's
own unrelated content. Costs a small matrix decomposition per step.

### ICG — Independent Condition Guidance

Blends the negative prediction towards plain noise, which makes the guidance depend almost
entirely on your positive prompt. High CFG collapses variety — every seed converges on the
same "best" answer — and a small amount of ICG (0.01–0.05) gives some of that variety back.
Turn it on partway through the run (**ICG start**) so composition is settled first.

### Guidance limiter

Turns guidance off entirely outside a noise-level window. The first few steps of a run
decide the composition and the last few polish texture, and guidance is arguably
counter-productive in both. Values are model-space sigmas — a run starts around **14.6**
for SD/SDXL and **1.0** for flow-matching models (Flux, Chroma, Qwen) — so the defaults
(5.42 / 0.28) are SD-shaped. `-1` disables that end of the window. Outside the window the
step runs at CFG 1.

### Post-CFG: MaHiRo

Re-weights the guided result against the plain prompt prediction by how strongly the
negative disagrees with it. Forge Neo has this built in too (Settings → show the MaHiRo
toggle); use one or the other, not both — the extension warns you if both are on.

---

## CFG schedule + plain-CFG tweaks

A per-step envelope on the CFG scale, plus three magnitude tweaks that only apply to the
**Normal** method. The whole box has to be ticked for anything in it to take effect.

The idea behind scheduling CFG: the early steps set composition, where strong guidance
locks you into a rigid, over-determined layout; the late steps set texture, where strong
guidance fries detail. Pushing the guidance into the middle of the run often gets better
results than any single fixed value.

All the step positions are fractions of the run, so they follow your step count.

| Control | What it does |
|---|---|
| **CFG 1 until step** / **CFG 1 after step** | Hard "no guidance at all" regions at the start and end of the run. |
| **CFG boost start / full boost at** / **boost factor** | Ramp the CFG scale *up* to `boost factor` × its value between these two points. |
| **CFG fade start / full fade at** / **fade factor** | Ramp it back *down* to `fade factor` × its value. |
| **centre conds to mean** | Remove each colour channel's average offset from both predictions. Counteracts the global colour cast that builds up at high guidance. |
| **Heuristic CFG** (Normal only) | Rescale the result to the magnitude a different CFG value would have produced. Above the live CFG scale it sharpens, below it softens — without changing what the guidance is pointing at. `0` = off. |
| **Reinhard CFG** (Normal only) | Tonemap the guidance strength, compressing the largest excursions and leaving small ones alone. Reins in the worst artefacts without flattening everything. `0` = off. |
| **Rescale CFG** (Normal only) | The familiar rescale: pull the result's contrast back towards the prompt prediction's. Mostly for v-prediction checkpoints. `0` = off. |

---

## Notes

**It takes over the CFG combination.** APG cannot be expressed as a correction applied to
an already-combined result, so this extension claims Forge's single
`sampler_cfg_function` slot. Forge Neo's built-in **Rescale CFG** slider (Settings →
"Display the Rescale CFG Slider") uses the same slot, and whichever runs last wins. If you
want rescaling alongside these methods, use the **Rescale CFG** control inside this
extension's schedule box instead. The extension logs a warning when it displaces another
script's hook.

**Hires. fix** re-runs the whole guidance path; untick *Apply to the Hires. fix pass* to
leave the upscale alone.

**XYZ plot** axes are registered for everything worth sweeping, under `(APG) …`.

**Settings are written to the image metadata** and paste back from it, so you can recover
what produced a result.

**If nothing seems to happen**, check the console. The extension logs its configuration on
the first step of a run and warns if it was enabled but never modified a step — which
normally means the CFG Scale is 1.0, or the guidance limiter excluded every step.

## Credits

A port to Forge Neo, merged from three upstream implementations:

- **[AdaptiveProjectedGuidance](https://github.com/DenOfEquity/AdaptiveProjectedGuidance)**
  by DenOfEquity — the original Forge2 extension, and the source of the method zoo (FDG,
  TraSCE, CFG★, tangential damping, ICG, the CFG schedule).
- **[ComfyUI-APG_ImYourCFGNow](https://github.com/Extraltodeus/ComfyUI-APG_ImYourCFGNow)**
  by Extraltodeus — the reference APG implementation, adaptive momentum, the momentum
  buffer's lifecycle, and the guidance limiter.
- **[reForge-APGIsYourCFG](https://github.com/Panchovix/reForge-APGIsYourCFG)** — the XYZ
  plot integration.

Papers as linked per method above. MaHiRo is by yoinked-h, via ComfyUI.

### What changed in the port

- **APG's projection is no longer skipped at rescale threshold 0.** In the Forge2 original
  the projection sat inside the `threshold > 0` branch, so a threshold of 0 silently
  returned the unguided prediction. This follows the ComfyUI node, which follows the paper.
- **The momentum buffer is reset per sampling pass** and whenever the trajectory restarts
  or the latent changes shape. Previously it persisted across images and into the Hires.
  fix pass, so the second image of a batch was guided by the first one's history.
- **No monkey-patching.** The original replaced one of Forge's own sampling functions to
  implement the CFG schedule; that copy has since drifted from Neo's. The schedule is now
  applied to the guidance scale inside the extension's own hook.
- **Adaptive momentum, the guidance limiter and the XYZ axes** are merged in from the
  ComfyUI/reForge side, which the Forge2 version never had.
- **Adaptive momentum is measured against the current pass** rather than a hard-coded
  1000-step timeline, so it also works on flow-matching models, whose timesteps run 0–1.
  On an eps model this matches the ComfyUI node to within one timestep.
- **All guidance maths runs in fp32 or better** — the sums involved lose too much
  precision in fp16 on 16-channel latents — and nothing is written into the prediction
  tensors in place, since Forge shares them with every other hook.
- **Prompt composition (`AND`) is honoured.** Taking over the CFG slot skips the line where
  Forge multiplies the scale by the summed conditioning strengths; the extension recovers
  that factor from a pre-CFG probe.
- **SLG (SD3)** was dropped: Forge Neo has no skip-layer support in its backend, so the
  option could never have done anything.
