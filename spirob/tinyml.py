from __future__ import annotations

from pathlib import Path
import torch


class _ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, imu, mask, actuator):
        out = self.model(imu, mask, actuator)
        return (
            out["success_prob"],
            out["quality"],
            out["family_logits"],
            out["size"],
            out["action_logits"],
            out["confidence"],
            out["trust"],
        )


def export_torchscript(model, sample: dict, path: str | Path) -> Path:
    path = Path(path)
    wrapper = _ExportWrapper(model).eval()
    with torch.no_grad():
        ts = torch.jit.trace(wrapper, (sample["imu"], sample["mask"], sample["actuator"]))
    ts.save(str(path))
    return path


def export_onnx(model, sample: dict, path: str | Path) -> Path:
    path = Path(path)
    wrapper = _ExportWrapper(model).eval()

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (sample["imu"], sample["mask"], sample["actuator"]),
                str(path),
                input_names=["imu", "mask", "actuator"],
                output_names=[
                    "success_prob",
                    "quality",
                    "family_logits",
                    "size",
                    "action_logits",
                    "confidence",
                    "trust",
                ],
                dynamic_axes={
                    "imu": {1: "time", 2: "imus"},
                    "mask": {1: "time", 2: "imus"},
                    "actuator": {1: "time"},
                    "success_prob": {0: "batch"},
                    "quality": {0: "batch"},
                    "family_logits": {0: "batch"},
                    "size": {0: "batch"},
                    "action_logits": {0: "batch"},
                    "confidence": {0: "batch"},
                    "trust": {0: "batch", 1: "imus"},
                },
                opset_version=17,
                export_params=True,
                do_constant_folding=True,
                dynamo=False,   # important: use legacy exporter
            )
        return path
    except Exception as exc:
        fallback = path.with_suffix(path.suffix + ".unavailable.txt")
        fallback.write_text(f"ONNX export unavailable: {exc}\n", encoding="utf-8")
        return fallback


def quantize_onnx_dynamic(onnx_path: str | Path, out_path: str | Path) -> Path:
    onnx_path = Path(onnx_path)
    out_path = Path(out_path)

    if onnx_path.suffix != ".onnx" or not onnx_path.exists():
        out_path = out_path.with_suffix(out_path.suffix + ".unavailable.txt")
        out_path.write_text(
            "INT8 quantization unavailable because a real ONNX model was not produced.\n",
            encoding="utf-8",
        )
        return out_path

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
        return out_path
    except Exception as exc:
        out_path = out_path.with_suffix(out_path.suffix + ".unavailable.txt")
        out_path.write_text(f"INT8 quantization unavailable: {exc}\n", encoding="utf-8")
        return out_path