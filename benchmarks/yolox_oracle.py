"""Compare RXLA YOLOX raw heads with the official PyTorch implementation."""

from pathlib import Path
from typing import Annotated
import os
import subprocess
import sys
import types

import numpy as np
import torch
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    plugin: Annotated[Path, Arg(help="trusted CUDA or CPU PJRT plugin")]
    checkpoint: Annotated[Path, Arg(help="official yolox_tiny.pth")]
    weights: Annotated[Path, Arg(help="converted RXLA safetensors")]
    output: Annotated[Path, Arg(help="directory for RXLA raw heads")]
    size: Annotated[int, Arg(help="positive multiple-of-32 input size")] = 64
    atol: Annotated[float, Arg(help="maximum accepted absolute error")] = 5e-5
    yolox_source: Annotated[
        Path, Arg(help="checkout containing the official-compatible fcyolox package")
    ] = Path("/home/me/code/projects/fcyolox/src")


def official_model(source: Path, checkpoint: Path) -> torch.nn.Module:
    # Model imports do not use OpenCV or icecream during a forward pass. Stubs
    # keep the oracle independent of those optional demo-only dependencies.
    sys.path.insert(0, os.fspath(source))
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    icecream = types.ModuleType("icecream")
    icecream.ic = lambda *values: values  # type: ignore[attr-defined]
    sys.modules.setdefault("icecream", icecream)

    from fcyolox.models.yolo_head import YOLOXHead
    from fcyolox.models.yolo_pafpn import YOLOPAFPN
    from fcyolox.models.yolox import YOLOX

    model = YOLOX(
        backbone=YOLOPAFPN(depth=0.33, width=0.375),
        head=YOLOXHead(num_classes=80, width=0.375),
    )
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(loaded.get("model", loaded), strict=True)
    model.eval()
    return model


def pytorch_outputs(model: torch.nn.Module, size: int) -> list[tuple[str, np.ndarray]]:
    captured: dict[str, list[torch.Tensor]] = {
        "reg": [],
        "obj": [],
        "cls": [],
    }
    handles = []
    backbone: dict[str, torch.Tensor] = {}
    for stage_name in ("stem", "dark2", "dark3", "dark4", "dark5"):
        handles.append(
            getattr(model.backbone.backbone, stage_name).register_forward_hook(
                lambda _module, _inputs, output, stage_name=stage_name: backbone.update(
                    {stage_name: output.detach()}
                )
            )
        )
    features: list[torch.Tensor] = []
    handles.append(
        model.backbone.register_forward_hook(
            lambda _module, _inputs, output: features.extend(
                value.detach() for value in output
            )
        )
    )
    for name, modules in (
        ("reg", model.head.reg_preds),
        ("obj", model.head.obj_preds),
        ("cls", model.head.cls_preds),
    ):
        for module in modules:
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, name=name: captured[name].append(
                        output.detach()
                    )
                )
            )
    elements = size * size * 3
    nhwc = ((torch.arange(elements, dtype=torch.int64) * 17 + 13) % 256).float()
    nchw = nhwc.reshape(1, size, size, 3).permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode():
        model(nchw)
    for handle in handles:
        handle.remove()
    output = [
        (name, backbone[name].permute(0, 2, 3, 1).contiguous().numpy())
        for name in ("stem", "dark2", "dark3", "dark4", "dark5")
    ]
    output.extend(
        (f"feature{i}", feature.permute(0, 2, 3, 1).contiguous().numpy())
        for i, feature in enumerate(features)
    )
    output.extend(
        (
            f"head{i}",
            torch.cat(
                [captured["reg"][i], captured["obj"][i], captured["cls"][i]], 1
            )
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy(),
        )
        for i in range(3)
    )
    return output


def main() -> None:
    args = Args.parse_args()
    if args.size <= 0 or args.size % 32:
        raise ValueError("size must be a positive multiple of 32")
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "rxla-safetensors",
            "--release",
            "--features",
            "model",
            "--example",
            "yolox_oracle",
            "--",
            os.fspath(args.plugin),
            os.fspath(args.weights),
            os.fspath(args.output),
            "--size",
            str(args.size),
        ],
        check=True,
    )
    expected = pytorch_outputs(official_model(args.yolox_source, args.checkpoint), args.size)
    failed = False
    for name, reference in expected:
        actual = np.fromfile(args.output / f"{name}.f32", dtype="<f4").reshape(
            reference.shape
        )
        difference = np.abs(actual - reference)
        maximum = float(difference.max())
        mean = float(difference.mean())
        print(
            f"tensor={name} shape={reference.shape} max_abs={maximum:.8g} "
            f"mean_abs={mean:.8g}"
        )
        failed |= not np.isfinite(difference).all() or maximum > args.atol
    if failed:
        raise SystemExit(f"YOLOX oracle exceeded atol={args.atol}")


if __name__ == "__main__":
    main()
