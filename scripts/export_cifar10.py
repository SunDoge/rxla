"""Download CIFAR-10 with torchvision and export Rust-friendly binary shards."""

from pathlib import Path
from typing import Annotated

from torchvision.datasets import CIFAR10
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    root: Annotated[Path, Arg("--root")] = Path(".cache/torchvision")
    output: Annotated[Path, Arg("--output")] = Path(".cache/cifar-10-binary")
    source_url: Annotated[str | None, Arg("--source-url")] = None


def write_shard(path: Path, images: object, labels: list[int]) -> None:
    with path.open("wb") as stream:
        for image, label in zip(images, labels, strict=True):
            stream.write(bytes([label]))
            stream.write(image.transpose(2, 0, 1).tobytes())


def main() -> None:
    args = Args.parse_args()
    if args.source_url is not None:
        CIFAR10.url = args.source_url
    train = CIFAR10(root=args.root, train=True, download=True)
    test = CIFAR10(root=args.root, train=False, download=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for shard in range(5):
        start = shard * 10_000
        write_shard(
            args.output / f"data_batch_{shard + 1}.bin",
            train.data[start : start + 10_000],
            train.targets[start : start + 10_000],
        )
    write_shard(args.output / "test_batch.bin", test.data, test.targets)
    print(args.output)


if __name__ == "__main__":
    main()
