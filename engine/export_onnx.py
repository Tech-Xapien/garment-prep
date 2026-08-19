"""Export FASHN SegFormer -> ONNX with preprocessing + argmax fused into the graph.

Design goals (throughput-first):
  * Input is UINT8 CHW (3,576,384). Gateway only does an INTER_AREA resize on CPU;
    /255 + ImageNet normalization run on the GPU inside the graph.
  * Output is an INT32 class map (576,384) — argmax done on the GPU, so we ship
    ~0.9 MB/image instead of an 18-channel float logits tensor (~16 MB).
  * Batch axis is dynamic; spatial size is static (ideal for a single TRT profile).

Run this ONCE on any machine with torch + transformers (neither is in the runtime
image), then feed the ONNX to build_trt.sh. The ONNX is portable; the .plan is not.

Usage:
    python engine/export_onnx.py --model fashn-ai/fashn-human-parser \
        --out engine/artifacts/segformer.onnx
"""
from __future__ import annotations

import argparse

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation

# Must match FashnHumanParser training preprocessing.
INPUT_H, INPUT_W = 576, 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ParserGraph(nn.Module):
    """Wraps SegFormer so the ONNX graph is: uint8 -> normalize -> logits -> argmax."""

    def __init__(self, model: SegformerForSemanticSegmentation):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.float() / 255.0
        x = (x - self.mean) / self.std
        logits = self.model(pixel_values=x).logits          # (B,18,H/4,W/4)
        logits = nn.functional.interpolate(
            logits, size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False
        )
        return logits.argmax(dim=1).to(torch.int32)          # (B,576,384)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fashn-ai/fashn-human-parser")
    ap.add_argument("--out", default="engine/artifacts/segformer.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = SegformerForSemanticSegmentation.from_pretrained(args.model).eval()
    graph = ParserGraph(model).eval()

    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, dtype=torch.uint8)
    torch.onnx.export(
        graph,
        (dummy,),
        args.out,
        input_names=["pixel_values"],
        output_names=["seg"],
        dynamic_axes={"pixel_values": {0: "batch"}, "seg": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter: no onnxscript dep, TRT-friendly graph
    )
    print(f"Exported ONNX -> {args.out}  (input uint8 [B,3,{INPUT_H},{INPUT_W}], output int32 seg)")


if __name__ == "__main__":
    main()
