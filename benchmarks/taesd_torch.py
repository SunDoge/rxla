"""PyTorch oracle and benchmark for RXLA's TAESD decoder."""

from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    weights: Annotated[Path, Arg(help="TAESD safetensors checkpoint")]
    device: Annotated[Literal["cpu", "cuda"], Arg("--device")] = "cuda"
    latent: Annotated[int, Arg("--latent")] = 64
    warmup: Annotated[int, Arg("--warmup")] = 2
    iterations: Annotated[int, Arg("--iterations")] = 10
    output: Annotated[Path | None, Arg("--output")] = None
    compare: Annotated[Path | None, Arg("--compare")] = None


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
        )
        self.skip = nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(value) + self.skip(value))


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(4, 64, 3, padding=1), nn.ReLU()]
        for stage, blocks in enumerate((3, 3, 3, 1)):
            layers.extend(TinyBlock() for _ in range(blocks))
            final = stage == 3
            if not final:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(nn.Conv2d(64, 3 if final else 64, 3, padding=1, bias=final))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.tanh(value / 3) * 3
        return self.layers(value) * 2 - 1


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def main() -> None:
    args = Args.parse_args()
    if args.latent <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("latent and iterations must be positive; warmup must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable")
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    model = Decoder().eval()
    checkpoint = load_file(args.weights)
    decoder = {name.removeprefix("decoder."): value for name, value in checkpoint.items() if name.startswith("decoder.")}
    model.load_state_dict(decoder, strict=True)
    model.to(args.device)
    count = args.latent * args.latent * 4
    latent_nhwc = ((torch.arange(count, dtype=torch.float32) % 257) - 128) / 64
    latent = latent_nhwc.reshape(1, args.latent, args.latent, 4).permute(0, 3, 1, 2).to(args.device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            image = model(latent)
            synchronize(args.device)
        samples = []
        for _ in range(args.iterations):
            synchronize(args.device)
            start = perf_counter()
            image = model(latent)
            synchronize(args.device)
            samples.append((perf_counter() - start) * 1e3)
    download_start = perf_counter()
    image = image.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype("<f4", copy=False)
    download_ms = (perf_counter() - download_start) * 1e3
    if args.output is not None:
        image.tofile(args.output)
    comparison = ""
    if args.compare is not None:
        expected = np.fromfile(args.compare, dtype="<f4")
        if expected.size != image.size:
            raise SystemExit(f"comparison has {expected.size} values, expected {image.size}")
        difference = np.abs(expected - image.reshape(-1))
        comparison = f" max_abs={difference.max():.9g} mean_abs={difference.mean():.9g}"
    print(
        f"backend=pytorch device={args.device} latent={args.latent} image={args.latent * 8} "
        f"warmup={args.warmup} iterations={args.iterations} mean_ms={sum(samples) / len(samples):.3f} "
        f"p50_ms={percentile(samples, 0.5):.3f} p95_ms={percentile(samples, 0.95):.3f} "
        f"output_download_ms={download_ms:.3f} "
        f"checksum={image.sum(dtype=np.float64):.9f} squared_sum={np.square(image.astype(np.float64)).sum():.9f}"
        f"{comparison}"
    )


if __name__ == "__main__":
    main()
