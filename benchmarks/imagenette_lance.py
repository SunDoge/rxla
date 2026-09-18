"""Convert an ImageFolder dataset to Lance and benchmark common access patterns."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal, Union

import lance
import pyarrow as pa
from PIL import Image
from pydantic import Field
from typed_args import Arg, Subparsers, TypedArgs


class BuildArgs(TypedArgs):
    """Build a self-contained Lance dataset from an ImageFolder."""

    command: Literal["build"] = "build"
    source: Annotated[
        Path, Arg(help="ImageFolder root containing train/ and val/", metavar="SOURCE")
    ]
    output: Annotated[Path, Arg(help="output Lance dataset", metavar="OUTPUT")]
    force: Annotated[bool, Arg("--force")] = False
    batch_size: Annotated[int, Arg("--batch-size", metavar="ROWS")] = 256


class BenchmarkArgs(TypedArgs):
    """Benchmark an existing Lance dataset against its ImageFolder source."""

    command: Literal["benchmark"] = "benchmark"
    source: Annotated[Path, Arg(help="source ImageFolder root", metavar="SOURCE")]
    dataset: Annotated[Path, Arg(help="existing Lance dataset", metavar="DATASET")]
    random_samples: Annotated[int, Arg("--random-samples", metavar="COUNT")] = 2_048
    repeats: Annotated[int, Arg("--repeats", metavar="COUNT")] = 3
    seed: Annotated[int, Arg("--seed", metavar="INTEGER")] = 42
    window_sizes: Annotated[
        list[int], Arg("--window-sizes", nargs="+", metavar="ROWS")
    ] = [32, 128, 512]
    json_output: Annotated[
        Path | None, Arg("--json-output", metavar="PATH")
    ] = None


Command = Annotated[
    Union[BuildArgs, BenchmarkArgs],
    Field(discriminator="command"),
    Subparsers(title="Commands"),
]


class Args(TypedArgs):
    command: Command


@dataclass(frozen=True)
class Sample:
    path: Path
    relative_path: str
    sample_id: str
    split: str
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Measurement:
    name: str
    samples: int
    bytes: int
    seconds_median: float
    seconds_all: list[float]
    samples_per_second: float
    mib_per_second: float


def discover(root: Path) -> list[Sample]:
    class_names = sorted(
        {
            directory.name
            for split in ("train", "val")
            for directory in (root / split).iterdir()
            if directory.is_dir()
        }
    )
    class_ids = {name: index for index, name in enumerate(class_names)}
    samples = []
    for split in ("train", "val"):
        for path in sorted((root / split).glob("*/*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            samples.append(
                Sample(
                    path=path,
                    relative_path=relative,
                    sample_id=hashlib.sha256(relative.encode()).hexdigest()[:32],
                    split=split,
                    class_id=class_ids[path.parent.name],
                    class_name=path.parent.name,
                )
            )
    if not samples:
        raise ValueError(f"no ImageFolder samples found below {root}")
    return samples


def schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("relative_path", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("class_id", pa.int16(), nullable=False),
            pa.field("class_name", pa.string(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("height", pa.int32(), nullable=False),
            lance.blob_field("image", nullable=False),
        ]
    )


def record_batches(samples: list[Sample], batch_size: int) -> Iterator[pa.RecordBatch]:
    dataset_schema = schema()
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        encoded = [sample.path.read_bytes() for sample in chunk]
        dimensions = []
        for payload in encoded:
            with Image.open(BytesIO(payload)) as image:
                dimensions.append(image.size)
        yield pa.RecordBatch.from_arrays(
            [
                pa.array([sample.sample_id for sample in chunk]),
                pa.array([sample.relative_path for sample in chunk]),
                pa.array([sample.split for sample in chunk]),
                pa.array([sample.class_id for sample in chunk], type=pa.int16()),
                pa.array([sample.class_name for sample in chunk]),
                pa.array([size[0] for size in dimensions], type=pa.int32()),
                pa.array([size[1] for size in dimensions], type=pa.int32()),
                lance.blob_array(encoded),
            ],
            schema=dataset_schema,
        )


def convert(samples: list[Sample], output: Path, batch_size: int) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reader = pa.RecordBatchReader.from_batches(schema(), record_batches(samples, batch_size))
    lance.write_dataset(
        reader,
        output,
        data_storage_version="2.2",
        enable_stable_row_ids=True,
        max_rows_per_group=1_024,
        max_rows_per_file=16_384,
    )
    return time.perf_counter() - started


def measure(
    name: str, count: int, operation: Callable[[], int], repeats: int
) -> Measurement:
    timings = []
    observed_bytes = 0
    for _ in range(repeats):
        started = time.perf_counter()
        observed_bytes = operation()
        timings.append(time.perf_counter() - started)
    median = statistics.median(timings)
    return Measurement(
        name=name,
        samples=count,
        bytes=observed_bytes,
        seconds_median=median,
        seconds_all=timings,
        samples_per_second=count / median,
        mib_per_second=observed_bytes / median / (1024 * 1024),
    )


def benchmark(
    samples: list[Sample],
    output: Path,
    random_samples: int,
    repeats: int,
    seed: int,
    window_sizes: list[int],
) -> list[Measurement]:
    dataset = lance.dataset(output)
    if dataset.count_rows() != len(samples):
        raise ValueError("Lance row count does not match the source dataset")
    selected = random.Random(seed).sample(
        range(len(samples)), min(random_samples, len(samples))
    )

    def window_plan(window_size: int) -> list[list[int]]:
        windows = [
            list(range(start, min(start + window_size, len(samples))))
            for start in range(0, len(samples), window_size)
        ]
        random.Random(seed).shuffle(windows)
        remaining = len(selected)
        plan = []
        for window in windows:
            if remaining == 0:
                break
            plan.append(window[:remaining])
            remaining -= len(plan[-1])
        return plan

    window_plans = {size: window_plan(size) for size in window_sizes}

    def directory_sequential() -> int:
        return sum(len(sample.path.read_bytes()) for sample in samples)

    def lance_sequential() -> int:
        total = 0
        batches = dataset.scanner(
            columns=["image"],
            batch_size=256,
            batch_readahead=8,
            blob_handling="all_binary",
        ).to_batches()
        for batch in batches:
            total += sum(len(value.as_py()) for value in batch.column(0))
        return total

    def directory_random() -> int:
        return sum(len(samples[index].path.read_bytes()) for index in selected)

    def lance_random() -> int:
        return sum(
            len(payload)
            for _, payload in dataset.read_blobs(
                "image", indices=selected, preserve_order=False
            )
            if payload is not None
        )

    def decode(payload: bytes) -> None:
        with Image.open(BytesIO(payload)) as image:
            image.convert("RGB").load()

    def directory_random_decode() -> int:
        total = 0
        for index in selected:
            payload = samples[index].path.read_bytes()
            decode(payload)
            total += len(payload)
        return total

    def lance_random_decode() -> int:
        total = 0
        for _, payload in dataset.read_blobs(
            "image", indices=selected, preserve_order=False
        ):
            if payload is not None:
                decode(payload)
                total += len(payload)
        return total

    def lance_window_shuffle(window_size: int, decode_images: bool) -> int:
        total = 0
        shuffle_buffer: list[bytes] = []
        rng = random.Random(seed)
        buffer_limit = window_size * 4

        def consume_one() -> None:
            nonlocal total
            index = rng.randrange(len(shuffle_buffer))
            payload = shuffle_buffer[index]
            shuffle_buffer[index] = shuffle_buffer[-1]
            shuffle_buffer.pop()
            if decode_images:
                decode(payload)
            total += len(payload)

        for indices in window_plans[window_size]:
            for _, payload in dataset.read_blobs(
                "image", indices=indices, preserve_order=True
            ):
                if payload is not None:
                    shuffle_buffer.append(payload)
            while len(shuffle_buffer) >= buffer_limit:
                consume_one()
        while shuffle_buffer:
            consume_one()
        return total

    def lance_window_local_shuffle(window_size: int) -> int:
        total = 0
        rng = random.Random(seed)
        for indices in window_plans[window_size]:
            payloads = [
                payload
                for _, payload in dataset.read_blobs(
                    "image", indices=indices, preserve_order=True
                )
                if payload is not None
            ]
            rng.shuffle(payloads)
            for payload in payloads:
                decode(payload)
                total += len(payload)
        return total

    def lance_scattered_same_samples(window_size: int, decode_images: bool) -> int:
        indices = [index for window in window_plans[window_size] for index in window]
        random.Random(seed).shuffle(indices)
        total = 0
        for _, payload in dataset.read_blobs(
            "image", indices=indices, preserve_order=True
        ):
            if payload is not None:
                if decode_images:
                    decode(payload)
                total += len(payload)
        return total

    def lance_metadata_filter() -> int:
        rows = 0
        for batch in dataset.scanner(
            columns=["sample_id", "class_id", "width", "height"],
            filter="split = 'train' AND width >= 320",
            batch_size=1_024,
        ).to_batches():
            rows += batch.num_rows
        return rows

    # One untimed pass makes these explicitly warm-cache, steady-state measurements.
    warmups: list[Callable[[], int]] = [
        directory_sequential,
        lance_sequential,
        directory_random,
        lance_random,
        directory_random_decode,
        lance_random_decode,
        lance_metadata_filter,
    ]
    for window_size in window_sizes:
        warmups.extend(
            [
                lambda size=window_size: lance_window_shuffle(size, False),
                lambda size=window_size: lance_scattered_same_samples(size, False),
                lambda size=window_size: lance_scattered_same_samples(size, True),
                lambda size=window_size: lance_window_local_shuffle(size),
                lambda size=window_size: lance_window_shuffle(size, True),
            ]
        )
    for operation in warmups:
        operation()

    metadata_rows = lance_metadata_filter()
    measurements = [
        measure("directory_sequential", len(samples), directory_sequential, repeats),
        measure("lance_sequential", len(samples), lance_sequential, repeats),
        measure("directory_random", len(selected), directory_random, repeats),
        measure("lance_random", len(selected), lance_random, repeats),
        measure(
            "directory_random_decode", len(selected), directory_random_decode, repeats
        ),
        measure("lance_random_decode", len(selected), lance_random_decode, repeats),
        measure("lance_metadata_filter", metadata_rows, lance_metadata_filter, repeats),
    ]
    for window_size in window_sizes:
        measurements.extend(
            [
                measure(
                    f"lance_window_{window_size}",
                    len(selected),
                    lambda size=window_size: lance_window_shuffle(size, False),
                    repeats,
                ),
                measure(
                    f"lance_scattered_{window_size}",
                    len(selected),
                    lambda size=window_size: lance_scattered_same_samples(size, False),
                    repeats,
                ),
                measure(
                    f"lance_scattered_{window_size}_decode",
                    len(selected),
                    lambda size=window_size: lance_scattered_same_samples(size, True),
                    repeats,
                ),
                measure(
                    f"lance_window_{window_size}_local_decode",
                    len(selected),
                    lambda size=window_size: lance_window_local_shuffle(size),
                    repeats,
                ),
                measure(
                    f"lance_window_{window_size}_pool_decode",
                    len(selected),
                    lambda size=window_size: lance_window_shuffle(size, True),
                    repeats,
                ),
            ]
        )
    return measurements


def main() -> None:
    args = Args.parse_args()
    command = args.command
    if isinstance(command, BuildArgs):
        if command.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        if command.output.exists():
            if not command.force:
                raise FileExistsError(
                    f"{command.output} already exists; pass --force to replace it"
                )
            shutil.rmtree(command.output)
        samples = discover(command.source)
        conversion_seconds = convert(samples, command.output, command.batch_size)
        print(
            json.dumps(
                {
                    "source": str(command.source),
                    "output": str(command.output),
                    "samples": len(samples),
                    "conversion_seconds": conversion_seconds,
                },
                indent=2,
            )
        )
        return

    if command.random_samples <= 0 or command.repeats <= 0:
        raise ValueError("random-samples and repeats must be positive")
    if not command.window_sizes or any(size <= 0 for size in command.window_sizes):
        raise ValueError("window-sizes must contain positive integers")
    if not command.dataset.exists():
        raise FileNotFoundError(
            f"{command.dataset} does not exist; run the build subcommand first"
        )
    samples = discover(command.source)
    measurements = benchmark(
        samples,
        command.dataset,
        command.random_samples,
        command.repeats,
        command.seed,
        command.window_sizes,
    )
    report = {
        "environment": {
            "python": platform.python_version(),
            "lance": lance.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
        },
        "source": str(command.source),
        "output": str(command.dataset),
        "samples": len(samples),
        "source_bytes": sum(sample.path.stat().st_size for sample in samples),
        "lance_bytes": sum(
            path.stat().st_size for path in command.dataset.rglob("*") if path.is_file()
        ),
        "warm_cache": True,
        "measurements": [asdict(measurement) for measurement in measurements],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if command.json_output is not None:
        command.json_output.parent.mkdir(parents=True, exist_ok=True)
        command.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
