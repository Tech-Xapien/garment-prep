# Deferred Optimizations

Throughput ideas we have **analyzed and deliberately NOT implemented yet**, with the
reason for each. See [`PIPELINE.md`](PIPELINE.md) §5 for the profiling that motivates
them. All three would require **restructuring the pipeline**, so they are tracked here
as future work rather than dropped in piecemeal.

Context: the pipeline is **CPU-bound** — on the shared prod box the GPU sits ~30–50%
utilized while the CPU (decode + canvas + PNG encode) is the bottleneck. The temptation
is to move pixel work onto the idle GPU, but that GPU is **not idle from the platform's
point of view** — it runs the **vton backend** at the same time.

---

## 1. Full GPU ensemble (decode + resize + parse + encode on the GPU) — NOT NOW

**What it is:** a Triton ensemble / DALI pipeline that takes JPEG bytes in and returns
PNG bytes out entirely on the GPU (nvJPEG decode → resize/normalize → TRT parse → CUDA
canvas → nvImageCodec encode). The CPU would do almost nothing and throughput would
become GPU-bound — potentially 3–8× current.

**Why deferred:** this is a beast. It would push the SegFormer parse **plus** all the
decode/resize/encode work onto the **tensor cores + GPU engines that vton is already
using**. We are co-tenant on one Blackwell card with the live vton backend; saturating
the GPU with garment-prep pixel work risks starving vton and hurting the product's main
path. Until garment-prep has its **own** GPU headroom (or a dedicated card), we keep the
heavy pixel work on the CPU where it can't contend with vton's tensor cores. The current
CPU-bound design is, in effect, a **deliberate GPU-usage cap**.

**Revisit when:** garment-prep gets a dedicated GPU, or vton's GPU utilization leaves a
reliable margin, or throughput demand exceeds what CPU scaling can supply.

---

## 2. WEBP output encode — NOT NOW

**What it is:** replace PNG encode (~48 ms/img) with WEBP (~30 ms/img, ~7× smaller
files) — faster on the bottleneck stage *and* far smaller callback payloads.

**Why deferred:** it changes the **output format** on the downstream contract. We are not
yet sure the downstream consumers (asset-service upload, vton preprocessing, any stored
`processed.png` reader) handle WEBP without issue — the complete-callback path today
assumes PNG (`Content-Type: image/png`, key `…/processed.png`). Shipping WEBP before
confirming every consumer decodes it could silently break garments downstream.

**Revisit when:** we've confirmed (or updated) every downstream reader to accept WEBP,
and decided lossy-vs-lossless. Low code cost once that's cleared — it's a format flag in
`imaging.encode`, gated behind an `OUTPUT_FORMAT` env.

---

## 3. Decode-at-scale by source resolution — MOST PROMISING, PENDING RESTRUCTURE

**What it is:** many source garments are larger (e.g. 2200×2400) than anything we output
(≤ 928×1664 canvas). JPEG supports near-free 1/2, 1/4, 1/8 **DCT-scaled decoding**, so we
can decode straight to the resolution we actually need instead of decoding full-res and
throwing pixels away. Cuts decode CPU (~26→~12 ms on large images) and speeds the
subsequent resizes.

```python
img = Image.open(io.BytesIO(raw))
img.draft("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT))   # libjpeg decodes at the largest 1/2^n >= target
```

**Why not shipped yet:** it's the **lowest-risk** of the three (no format change, no GPU
contention — we were going to downscale high-res images anyway), but doing it *correctly*
means threading a "target decode size" through decode → crop-coordinate scaling and
verifying quality across the resolution mix, i.e. a pipeline restructure rather than a
one-liner. It's the first one to pick up when we next restructure.

**Safety:** no-op for sources already ≤ the canvas (draft never decodes below target), so
small images are unaffected.

---

## Summary

| Idea | Gain | Risk | Blocker |
|------|------|------|---------|
| GPU ensemble | 3–8× | high | co-tenant with vton on one GPU — would contend for tensor cores |
| WEBP encode | ~1.6× encode, 7× smaller files | medium | downstream must accept WEBP |
| Decode-at-scale | ~1.2–1.5× on large sources | low | needs pipeline restructure + quality check |

Order when we resume: **decode-at-scale → WEBP (after downstream sign-off) → GPU ensemble
(only with dedicated GPU headroom).**
