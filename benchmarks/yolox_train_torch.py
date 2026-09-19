"""Benchmark an official-compatible PyTorch YOLOX-tiny training step."""

import os
from pathlib import Path
import sys
import time
import types
from typing import Annotated, Literal

import torch
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    checkpoint: Annotated[Path, Arg(help="official yolox_tiny.pth")]
    size: Annotated[int, Arg("--size")] = 416
    batch_size: Annotated[int, Arg("--batch-size")] = 8
    warmup: Annotated[int, Arg("--warmup")] = 5
    iterations: Annotated[int, Arg("--iterations")] = 20
    targets_per_image: Annotated[int, Arg("--targets-per-image")] = 12
    learning_rate: Annotated[float, Arg("--learning-rate")] = 0.00015625
    precision: Annotated[Literal["f32", "bf16", "f16"], Arg("--precision")] = "f32"
    execution: Annotated[Literal["eager", "compile"], Arg("--execution")] = "eager"
    l1: Annotated[bool, Arg("--l1")] = False
    profile: Annotated[
        bool, Arg("--profile", help="bracket timed iterations with CUDA profiler API")
    ] = False
    yolox_source: Annotated[
        Path, Arg("--yolox-source")
    ] = Path("/home/me/code/projects/fcyolox/src")


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def load_model(source: Path, checkpoint: Path) -> torch.nn.Module:
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
    return model


def optimizer_for(model: torch.nn.Module, learning_rate: float) -> torch.optim.SGD:
    normalizations = tuple(
        layer
        for name, layer in torch.nn.__dict__.items()
        if "Norm" in name and isinstance(layer, type)
    )
    no_decay = []
    decay = []
    biases = []
    for module in model.modules():
        if hasattr(module, "bias") and isinstance(module.bias, torch.nn.Parameter):
            biases.append(module.bias)
        if isinstance(module, normalizations):
            if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                no_decay.append(module.weight)
        elif hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
            decay.append(module.weight)
    optimizer = torch.optim.SGD(
        no_decay, lr=learning_rate, momentum=0.9, nesterov=True
    )
    optimizer.add_param_group({"params": decay, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": biases})
    return optimizer


class U8InputModel(torch.nn.Module):
    """Keep the deployment input boundary identical to RXLA's compiled graph."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.model(images.float(), targets)


def main() -> None:
    args = Args.parse_args()
    if (
        args.size <= 0
        or args.size % 32
        or args.batch_size <= 0
        or args.warmup < 0
        or args.iterations <= 0
        or not 1 <= args.targets_per_image <= 120
        or args.learning_rate < 0
    ):
        raise ValueError("invalid benchmark arguments")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable")

    torch.backends.cudnn.benchmark = True
    inner_model = load_model(args.yolox_source, args.checkpoint).train().cuda()
    inner_model.head.use_l1 = args.l1
    model = U8InputModel(inner_model).cuda()
    optimizer = optimizer_for(model, args.learning_rate)
    if args.execution == "compile":
        # YOLOX caches head grids across calls. CUDA-graph mode overwrites these
        # user-visible tensors, so use Inductor's default mode for this model.
        model = torch.compile(model)
    autocast_dtype = {
        "f32": torch.float32,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
    }[args.precision]
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "f16")
    count = args.batch_size * args.size * args.size * 3
    images = (
        ((torch.arange(count, dtype=torch.int64) * 17 + 13) % 256)
        .to(torch.uint8)
        .reshape(args.batch_size, args.size, args.size, 3)
        .permute(0, 3, 1, 2)
        .contiguous()
        .cuda()
    )
    targets = torch.zeros((args.batch_size, 120, 5), device="cuda")
    scale = args.size / 64.0
    for index in range(args.targets_per_image):
        targets[:, index, 0] = index % 80
        targets[:, index, 1] = (8 + index * 11 % 48) * scale
        targets[:, index, 2] = (8 + index * 17 % 48) * scale
        targets[:, index, 3] = (6 + index % 13) * scale
        targets[:, index, 4] = (7 + index * 3 % 12) * scale

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=args.precision != "f32"
        ):
            loss = model(images, targets)["total_loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    loss = torch.tensor(float("nan"), device="cuda")
    for _ in range(args.warmup):
        loss = step()
    torch.cuda.synchronize()
    samples = []
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iterations):
        started = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    total_ms = sum(samples)
    print(
        f"pytorch={args.execution} precision={args.precision} size={args.size} batch={args.batch_size} "
        f"iterations={args.iterations} targets_per_image={args.targets_per_image} "
        f"l1={args.l1} loss={loss.item():.6f}"
    )
    print(
        f"mean_ms={total_ms / len(samples):.3f} "
        f"p50_ms={percentile(samples, 0.5):.3f} "
        f"p95_ms={percentile(samples, 0.95):.3f} "
        f"images_per_second={args.batch_size * args.iterations / (total_ms / 1e3):.1f}"
    )


if __name__ == "__main__":
    main()
