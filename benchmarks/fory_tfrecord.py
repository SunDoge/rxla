"""Export COCO samples as TFRecord-framed Apache Fory Standard Rows."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import pyfory
from typed_args import Arg, TypedArgs
from tfrecord_slow.writer import TfrecordWriter


@dataclass
class DetectionRecord:
    sample_id: str
    width: pyfory.Int32
    height: pyfory.Int32
    image: bytes
    boxes: list[pyfory.Float32]
    labels: list[pyfory.Int32]


class Args(TypedArgs):
    root: Annotated[Path, Arg(help="COCO 2017 root", metavar="ROOT")]
    output: Annotated[Path, Arg(help="output TFRecord shard", metavar="OUTPUT")]
    samples: Annotated[int, Arg("--samples", metavar="COUNT")] = 4_096
    seed: Annotated[int, Arg("--seed", metavar="INTEGER")] = 42
    force: Annotated[bool, Arg("--force")] = False


def masked_crc32c_available() -> None:
    try:
        import crc32c  # noqa: F401
    except ImportError as error:
        raise RuntimeError("install tfrecord-slow with its crc32c extra") from error


def main() -> None:
    args = Args.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")

    masked_crc32c_available()
    document = json.loads(
        (args.root / "annotations" / "instances_train2017.json").read_text()
    )
    annotations: dict[int, list[dict]] = {}
    for annotation in document["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    images = random.Random(args.seed).sample(
        document["images"], min(args.samples, len(document["images"]))
    )

    encoder = pyfory.encoder(DetectionRecord)
    schema = str(encoder.schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoded_bytes = 0
    image_bytes = 0
    boxes = 0
    with TfrecordWriter.create(str(args.output), buffering=8 << 20) as writer:
        for image in images:
            instances = annotations.get(image["id"], [])
            coordinates = []
            labels = []
            for instance in instances:
                x, y, width, height = instance["bbox"]
                coordinates.extend((x, y, x + width, y + height))
                labels.append(instance["category_id"])
            jpeg = (args.root / "train2017" / image["file_name"]).read_bytes()
            row = encoder.to_row(
                DetectionRecord(
                    sample_id=f"coco:train2017:{image['id']}",
                    width=image["width"],
                    height=image["height"],
                    image=jpeg,
                    boxes=coordinates,
                    labels=labels,
                )
            ).to_bytes()
            writer.write(row)
            encoded_bytes += len(row)
            image_bytes += len(jpeg)
            boxes += len(instances)
    seconds = time.perf_counter() - started

    report = {
        "codec": "apache-fory-standard-row-1.7",
        "container": "tfrecord",
        "schema": schema,
        "schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
        "samples": len(images),
        "boxes": boxes,
        "image_bytes": image_bytes,
        "row_bytes": encoded_bytes,
        "tfrecord_bytes": args.output.stat().st_size,
        "seconds": seconds,
        "samples_per_second": len(images) / seconds,
        "mib_per_second": image_bytes / seconds / (1024 * 1024),
    }
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
