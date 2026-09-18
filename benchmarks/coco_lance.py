"""Build and benchmark a detection-oriented COCO Lance dataset."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import lance
import pyarrow as pa
from PIL import Image
from pydantic import Field
from typed_args import Arg, Subparsers, TypedArgs


class BuildArgs(TypedArgs):
    """Build COCO splits as one self-contained Lance dataset."""

    command: Literal["build"] = "build"
    root: Annotated[Path, Arg(help="COCO 2017 root", metavar="ROOT")]
    output: Annotated[Path, Arg(help="output Lance dataset", metavar="OUTPUT")]
    splits: Annotated[
        list[str], Arg("--splits", nargs="+", metavar="NAME")
    ] = ["train2017", "val2017"]
    force: Annotated[bool, Arg("--force")] = False
    batch_size: Annotated[int, Arg("--batch-size", metavar="ROWS")] = 256


class BenchmarkArgs(TypedArgs):
    """Benchmark an existing COCO Lance dataset."""

    command: Literal["benchmark"] = "benchmark"
    dataset: Annotated[Path, Arg(help="existing Lance dataset", metavar="DATASET")]
    samples: Annotated[int, Arg("--samples", metavar="COUNT")] = 2_048
    read_window: Annotated[int, Arg("--read-window", metavar="ROWS")] = 128
    shuffle_buffer: Annotated[int, Arg("--shuffle-buffer", metavar="ROWS")] = 512
    repeats: Annotated[int, Arg("--repeats", metavar="COUNT")] = 3
    seed: Annotated[int, Arg("--seed", metavar="INTEGER")] = 42
    json_output: Annotated[Path | None, Arg("--json-output", metavar="PATH")] = None


Command = Annotated[
    Union[BuildArgs, BenchmarkArgs],
    Field(discriminator="command"),
    Subparsers(title="Commands"),
]


class Args(TypedArgs):
    command: Command


@dataclass(frozen=True)
class Measurement:
    name: str
    samples: int
    bytes: int
    seconds_median: float
    seconds_all: list[float]
    samples_per_second: float
    mib_per_second: float


INSTANCE_TYPE = pa.struct(
    [
        pa.field("annotation_id", pa.int64(), nullable=False),
        pa.field("category_id", pa.int16(), nullable=False),
        pa.field("bbox_xywh", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("bbox_xyxy", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("area", pa.float32(), nullable=False),
        pa.field("is_crowd", pa.bool_(), nullable=False),
        pa.field("segmentation_json", pa.large_string(), nullable=False),
    ]
)


def schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("coco_image_id", pa.int64(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("file_name", pa.string(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("height", pa.int32(), nullable=False),
            pa.field("box_count", pa.int32(), nullable=False),
            pa.field("crowd_count", pa.int32(), nullable=False),
            pa.field("category_ids", pa.list_(pa.int16()), nullable=False),
            pa.field("instances", pa.list_(INSTANCE_TYPE), nullable=False),
            lance.blob_field("image", nullable=False),
        ],
        metadata={"rxla.format": "coco-detection", "rxla.schema-version": "1"},
    )


def load_coco(root: Path, split: str) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    annotation_path = root / "annotations" / f"instances_{split}.json"
    document = json.loads(annotation_path.read_text())
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in document["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    return sorted(document["images"], key=lambda image: image["id"]), annotations


def encode_instance(annotation: dict[str, Any]) -> dict[str, Any]:
    x, y, width, height = annotation["bbox"]
    return {
        "annotation_id": annotation["id"],
        "category_id": annotation["category_id"],
        "bbox_xywh": [x, y, width, height],
        "bbox_xyxy": [x, y, x + width, y + height],
        "area": annotation["area"],
        "is_crowd": bool(annotation["iscrowd"]),
        "segmentation_json": json.dumps(annotation["segmentation"], separators=(",", ":")),
    }


def record_batches(
    root: Path,
    split: str,
    images: list[dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    dataset_schema = schema()
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        encoded_images = [(root / split / image["file_name"]).read_bytes() for image in chunk]
        encoded_instances = [
            [encode_instance(annotation) for annotation in annotations[image["id"]]]
            for image in chunk
        ]
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(
                    [
                        hashlib.sha256(f"coco:{split}:{image['id']}".encode()).hexdigest()[:32]
                        for image in chunk
                    ]
                ),
                pa.array([image["id"] for image in chunk], type=pa.int64()),
                pa.array([split] * len(chunk)),
                pa.array([image["file_name"] for image in chunk]),
                pa.array([image["width"] for image in chunk], type=pa.int32()),
                pa.array([image["height"] for image in chunk], type=pa.int32()),
                pa.array([len(instances) for instances in encoded_instances], type=pa.int32()),
                pa.array(
                    [sum(instance["is_crowd"] for instance in instances) for instances in encoded_instances],
                    type=pa.int32(),
                ),
                pa.array(
                    [
                        sorted({instance["category_id"] for instance in instances})
                        for instances in encoded_instances
                    ],
                    type=pa.list_(pa.int16()),
                ),
                pa.array(encoded_instances, type=pa.list_(INSTANCE_TYPE)),
                lance.blob_array(encoded_images),
            ],
            schema=dataset_schema,
        )


def build(command: BuildArgs) -> dict[str, Any]:
    if command.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if command.output.exists():
        if not command.force:
            raise FileExistsError(f"{command.output} exists; pass --force to replace it")
        shutil.rmtree(command.output)
    if not command.splits or len(set(command.splits)) != len(command.splits):
        raise ValueError("splits must be nonempty and distinct")
    sources = []
    for split in command.splits:
        images, annotations = load_coco(command.root, split)
        sources.append((split, images, annotations))
    missing = [
        f"{split}/{image['file_name']}"
        for split, images, _ in sources
        for image in images
        if not (command.root / split / image["file_name"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} annotated images are missing; first: {missing[0]}")
    started = time.perf_counter()

    def batches() -> Iterator[pa.RecordBatch]:
        for split, images, annotations in sources:
            yield from record_batches(
                command.root, split, images, annotations, command.batch_size
            )

    reader = pa.RecordBatchReader.from_batches(schema(), batches())
    dataset = lance.write_dataset(
        reader,
        command.output,
        data_storage_version="2.2",
        enable_stable_row_ids=True,
        max_rows_per_group=1_024,
        max_rows_per_file=16_384,
    )
    return {
        "splits": {
            split: {
                "images": len(images),
                "annotations": sum(map(len, annotations.values())),
            }
            for split, images, annotations in sources
        },
        "images": sum(len(images) for _, images, _ in sources),
        "annotations": sum(
            len(values)
            for _, _, annotations in sources
            for values in annotations.values()
        ),
        "seconds": time.perf_counter() - started,
        "version": dataset.version,
    }


def measure(
    name: str, count: int, operation: Callable[[], int], repeats: int
) -> Measurement:
    timings = []
    byte_count = 0
    for _ in range(repeats):
        started = time.perf_counter()
        byte_count = operation()
        timings.append(time.perf_counter() - started)
    median = statistics.median(timings)
    return Measurement(
        name=name,
        samples=count,
        bytes=byte_count,
        seconds_median=median,
        seconds_all=timings,
        samples_per_second=count / median,
        mib_per_second=byte_count / median / (1024 * 1024),
    )


def benchmark(command: BenchmarkArgs) -> dict[str, Any]:
    if min(command.samples, command.read_window, command.shuffle_buffer, command.repeats) <= 0:
        raise ValueError("samples, read-window, shuffle-buffer and repeats must be positive")
    if command.shuffle_buffer < command.read_window:
        raise ValueError("shuffle-buffer must be at least read-window")
    dataset = lance.dataset(command.dataset)
    row_count = dataset.count_rows()
    sample_count = min(command.samples, row_count)
    starts = list(range(0, row_count, command.read_window))
    random.Random(command.seed).shuffle(starts)
    plan = []
    remaining = sample_count
    for start in starts:
        take = min(command.read_window, row_count - start, remaining)
        plan.append(list(range(start, start + take)))
        remaining -= take
        if remaining == 0:
            break
    scattered = [index for window in plan for index in window]
    random.Random(command.seed).shuffle(scattered)

    def consume(payload: bytes) -> int:
        with Image.open(BytesIO(payload)) as image:
            image.convert("RGB").load()
        return len(payload)

    def scattered_decode() -> int:
        return sum(
            consume(payload)
            for _, payload in dataset.read_blobs("image", indices=scattered, preserve_order=True)
            if payload is not None
        )

    def window_pool_decode() -> int:
        pool: list[bytes] = []
        rng = random.Random(command.seed)
        total = 0

        def pop() -> int:
            index = rng.randrange(len(pool))
            payload = pool[index]
            pool[index] = pool[-1]
            pool.pop()
            return consume(payload)

        for indices in plan:
            pool.extend(
                payload
                for _, payload in dataset.read_blobs("image", indices=indices, preserve_order=True)
                if payload is not None
            )
            while len(pool) >= command.shuffle_buffer:
                total += pop()
        while pool:
            total += pop()
        return total

    def metadata_query() -> int:
        return sum(
            batch.num_rows
            for batch in dataset.scanner(
                columns=["sample_id", "box_count", "category_ids"],
                filter="box_count >= 10 AND crowd_count = 0",
                batch_size=1_024,
            ).to_batches()
        )

    def category_query() -> int:
        return sum(
            batch.num_rows
            for batch in dataset.scanner(
                columns=["sample_id", "box_count"],
                filter="array_has(category_ids, 1)",
                batch_size=1_024,
            ).to_batches()
        )

    for operation in (
        scattered_decode,
        window_pool_decode,
        metadata_query,
        category_query,
    ):
        operation()
    matched = metadata_query()
    category_matched = category_query()
    measurements = [
        measure("scattered_decode", sample_count, scattered_decode, command.repeats),
        measure("window_pool_decode", sample_count, window_pool_decode, command.repeats),
        measure("metadata_query", matched, metadata_query, command.repeats),
        measure(
            "category_query", category_matched, category_query, command.repeats
        ),
    ]
    box_counts = dataset.to_table(columns=["box_count"])["box_count"].to_pylist()
    return {
        "environment": {
            "python": platform.python_version(),
            "lance": lance.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
        },
        "dataset": str(command.dataset),
        "rows": row_count,
        "fragments": len(dataset.get_fragments()),
        "dataset_bytes": sum(
            path.stat().st_size
            for path in command.dataset.rglob("*")
            if path.is_file()
        ),
        "boxes": sum(box_counts),
        "empty_images": sum(count == 0 for count in box_counts),
        "max_boxes_per_image": max(box_counts),
        "read_window": command.read_window,
        "shuffle_buffer": command.shuffle_buffer,
        "measurements": [asdict(measurement) for measurement in measurements],
    }


def main() -> None:
    command = Args.parse_args().command
    if isinstance(command, BuildArgs):
        report = build(command)
    else:
        if not command.dataset.exists():
            raise FileNotFoundError(command.dataset)
        report = benchmark(command)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if isinstance(command, BenchmarkArgs) and command.json_output is not None:
        command.json_output.parent.mkdir(parents=True, exist_ok=True)
        command.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
