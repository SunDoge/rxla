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


def pytorch_assignments(
    model: torch.nn.Module, outputs: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decoded = []
    objectness = []
    classes = []
    x_shifts = []
    y_shifts = []
    expanded_strides = []
    for index, stride in enumerate((8, 16, 32)):
        raw = torch.from_numpy(outputs[f"head{index}"])
        _, height, width, channels = raw.shape
        flat = raw.reshape(1, height * width, channels)
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        grid = torch.stack((x, y), -1).reshape(1, height * width, 2).float()
        boxes = flat[..., :4].clone()
        boxes[..., :2] = (boxes[..., :2] + grid) * stride
        boxes[..., 2:4] = boxes[..., 2:4].exp() * stride
        decoded.append(boxes)
        objectness.append(flat[..., 4:5])
        classes.append(flat[..., 5:])
        x_shifts.append(grid[..., 0])
        y_shifts.append(grid[..., 1])
        expanded_strides.append(torch.full((1, height * width), float(stride)))

    boxes = torch.cat(decoded, 1)
    objectness_logits = torch.cat(objectness, 1)
    class_logits = torch.cat(classes, 1)
    x_shifts_tensor = torch.cat(x_shifts, 1)
    y_shifts_tensor = torch.cat(y_shifts, 1)
    strides_tensor = torch.cat(expanded_strides, 1)
    ground_truth = torch.tensor(
        [[20.0, 20.0, 12.0, 16.0], [44.0, 42.0, 18.0, 14.0]]
    )
    ground_truth_classes = torch.tensor([0.0, 7.0])
    (
        matched_classes,
        foreground,
        matched_ious,
        matched_indices,
        _count,
    ) = model.head.get_assignments(
        0,
        2,
        ground_truth,
        ground_truth_classes,
        boxes[0],
        strides_tensor,
        x_shifts_tensor,
        y_shifts_tensor,
        class_logits,
        objectness_logits,
    )
    anchors = boxes.shape[1]
    dense_boxes = torch.zeros((anchors, 4))
    dense_objectness = foreground.float()
    dense_classes = torch.zeros((anchors, 80))
    dense_boxes[foreground] = ground_truth[matched_indices]
    dense_classes[foreground, matched_classes.long()] = matched_ious
    return (
        dense_boxes.numpy(),
        dense_objectness.numpy(),
        dense_classes.numpy(),
    )


def pytorch_losses(
    outputs: dict[str, np.ndarray],
    assignments: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    head_tensors = [
        torch.tensor(outputs[f"head{index}"], requires_grad=True)
        for index in range(3)
    ]
    losses = pytorch_loss_from_heads(head_tensors, assignments)
    losses[0].backward()
    return (
        losses.detach().numpy(),
        [head.grad.detach().numpy() for head in head_tensors],
    )


def pytorch_loss_from_heads(
    head_tensors: list[torch.Tensor],
    assignments: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> torch.Tensor:
    decoded = []
    raw_boxes = []
    objectness_logits = []
    class_logits = []
    l1_targets = []
    boxes_target = torch.from_numpy(assignments[0]).unsqueeze(0)
    foreground = torch.from_numpy(assignments[1]).reshape(1, -1, 1)
    classes_target = torch.from_numpy(assignments[2]).unsqueeze(0)
    anchor_offset = 0
    for raw, stride in zip(head_tensors, (8, 16, 32), strict=True):
        batch, height, width, channels = raw.shape
        flat = raw.reshape(batch, height * width, channels)
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        grid = torch.stack((x, y), -1).reshape(1, height * width, 2).float()
        decoded_box = flat[..., :4].clone()
        decoded_box[..., :2] = (decoded_box[..., :2] + grid) * stride
        decoded_box[..., 2:4] = decoded_box[..., 2:4].exp() * stride
        decoded.append(decoded_box)
        raw_boxes.append(flat[..., :4])
        objectness_logits.append(flat[..., 4:5])
        class_logits.append(flat[..., 5:])
        count = height * width
        target = boxes_target[:, anchor_offset : anchor_offset + count]
        l1_targets.append(
            torch.cat(
                [
                    target[..., :2] / stride - grid,
                    torch.log(target[..., 2:4] / stride + 1e-8),
                ],
                -1,
            )
        )
        anchor_offset += count

    decoded_boxes = torch.cat(decoded, 1)
    raw_boxes_tensor = torch.cat(raw_boxes, 1)
    objectness_tensor = torch.cat(objectness_logits, 1)
    classes_tensor = torch.cat(class_logits, 1)
    l1_target = torch.cat(l1_targets, 1)
    mask = foreground.bool().reshape(-1)
    normalizer = foreground.sum().clamp(min=1.0)
    prediction = decoded_boxes.reshape(-1, 4)[mask]
    target = boxes_target.reshape(-1, 4)[mask]
    top_left = torch.maximum(prediction[:, :2] - prediction[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
    bottom_right = torch.minimum(prediction[:, :2] + prediction[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
    intersection = (bottom_right - top_left).prod(1) * (top_left < bottom_right).all(1)
    union = prediction[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection
    iou = intersection / (union + 1e-16)
    box_loss = (1.0 - iou.square()).sum() / normalizer
    objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        objectness_tensor, foreground, reduction="sum"
    ) / normalizer
    class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        classes_tensor.reshape(-1, 80)[mask],
        classes_target.reshape(-1, 80)[mask],
        reduction="sum",
    ) / normalizer
    l1_loss = torch.nn.functional.l1_loss(
        raw_boxes_tensor.reshape(-1, 4)[mask],
        l1_target.reshape(-1, 4)[mask],
        reduction="sum",
    ) / normalizer
    total = 5.0 * box_loss + objectness_loss + class_loss + l1_loss
    return torch.stack(
        [total, box_loss, objectness_loss, class_loss, l1_loss]
    )


def pytorch_model_gradients(
    model: torch.nn.Module,
    size: int,
    assignments: tuple[np.ndarray, np.ndarray, np.ndarray],
    paths: tuple[str, ...],
) -> dict[str, np.ndarray]:
    captured: dict[str, list[torch.Tensor]] = {
        "reg": [],
        "obj": [],
        "cls": [],
    }
    handles = []
    for name, modules in (
        ("reg", model.head.reg_preds),
        ("obj", model.head.obj_preds),
        ("cls", model.head.cls_preds),
    ):
        for module in modules:
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, name=name: captured[name].append(
                        output
                    )
                )
            )

    model.zero_grad(set_to_none=True)
    elements = size * size * 3
    nhwc = ((torch.arange(elements, dtype=torch.int64) * 17 + 13) % 256).float()
    nchw = nhwc.reshape(1, size, size, 3).permute(0, 3, 1, 2).contiguous()
    model(nchw)
    for handle in handles:
        handle.remove()
    heads = [
        torch.cat(
            [captured["reg"][index], captured["obj"][index], captured["cls"][index]],
            1,
        )
        .permute(0, 2, 3, 1)
        .contiguous()
        for index in range(3)
    ]
    pytorch_loss_from_heads(heads, assignments)[0].backward()
    parameters = dict(model.named_parameters())
    gradients = {}
    for path in paths:
        gradient = parameters[path].grad
        if gradient is None:
            raise RuntimeError(f"parameter did not receive a gradient: {path}")
        gradients[path] = gradient.detach().numpy()
    return gradients


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
    model = official_model(args.yolox_source, args.checkpoint)
    expected = pytorch_outputs(model, args.size)
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
    expected_by_name = dict(expected)
    expected_assignment = pytorch_assignments(model, expected_by_name)
    anchors = sum((args.size // stride) ** 2 for stride in (8, 16, 32))
    actual_assignment = (
        np.fromfile(args.output / "assignment_boxes.f32", dtype="<f4").reshape(
            anchors, 4
        ),
        np.fromfile(
            args.output / "assignment_objectness.f32", dtype="<f4"
        ).reshape(anchors),
        np.fromfile(args.output / "assignment_classes.f32", dtype="<f4").reshape(
            anchors, 80
        ),
    )
    assignment_names = ("boxes", "objectness", "classes")
    for name, actual, reference in zip(
        assignment_names, actual_assignment, expected_assignment, strict=True
    ):
        difference = np.abs(actual - reference)
        maximum = float(difference.max(initial=0.0))
        print(
            f"assignment={name} foreground={int(actual_assignment[1].sum())} "
            f"max_abs={maximum:.8g}"
        )
        failed |= not np.array_equal(actual != 0, reference != 0) or maximum > args.atol
    expected_loss, expected_gradients = pytorch_losses(
        expected_by_name, expected_assignment
    )
    actual_loss = np.fromfile(args.output / "loss.f32", dtype="<f4")
    loss_difference = np.abs(actual_loss - expected_loss)
    print(
        f"loss actual={actual_loss.tolist()} expected={expected_loss.tolist()} "
        f"max_abs={float(loss_difference.max()):.8g}"
    )
    failed |= (
        actual_loss.shape != expected_loss.shape
        or not np.isfinite(actual_loss).all()
        or float(loss_difference.max()) > args.atol * 5
    )
    for index, reference in enumerate(expected_gradients):
        actual = np.fromfile(
            args.output / f"head{index}_gradient.f32", dtype="<f4"
        ).reshape(reference.shape)
        difference = np.abs(actual - reference)
        maximum = float(difference.max(initial=0.0))
        mean = float(difference.mean())
        print(
            f"gradient=head{index} shape={reference.shape} "
            f"max_abs={maximum:.8g} mean_abs={mean:.8g}"
        )
        failed |= not np.isfinite(difference).all() or maximum > args.atol * 5
    model_gradient_paths = (
        "backbone.backbone.stem.conv.conv.weight",
        "backbone.backbone.dark5.1.conv1.conv.weight",
        "backbone.backbone.dark5.1.conv2.conv.weight",
        "head.cls_preds.0.weight",
    )
    expected_model_gradients = pytorch_model_gradients(
        model, args.size, expected_assignment, model_gradient_paths
    )
    for path, reference in expected_model_gradients.items():
        actual = np.fromfile(
            args.output / f"model_gradient_{path.replace('.', '_')}.f32",
            dtype="<f4",
        ).reshape(reference.shape)
        difference = np.abs(actual - reference)
        maximum = float(difference.max(initial=0.0))
        mean = float(difference.mean())
        print(
            f"model_gradient={path} shape={reference.shape} "
            f"max_abs={maximum:.8g} mean_abs={mean:.8g}"
        )
        failed |= not np.isfinite(difference).all() or maximum > args.atol * 5
    if failed:
        raise SystemExit(f"YOLOX oracle exceeded atol={args.atol}")


if __name__ == "__main__":
    main()
