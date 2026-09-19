"""Run an RXLA-exported YOLOX StableHLO program with JAX on COCO.

The script deliberately keeps COCO postprocessing outside the model program:
JAX executes the exact StableHLO inference graph, while Python performs image
loading, decode/NMS, and writes the standard JSON consumed by pycocotools.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import torch
from jax._src import xla_bridge
from PIL import Image
from torchvision.ops import batched_nms
from typed_args import Arg, TypedArgs


COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)
SIZE = 416


class Args(TypedArgs):
    fixture: Annotated[Path, Arg(help="directory exported by yolox_coco_eval")]
    annotations: Annotated[Path, Arg(help="COCO instances JSON")]
    images: Annotated[Path, Arg(help="directory containing validation images")]
    output: Annotated[Path, Arg(help="standard COCO detection-result JSON")]
    backend: Annotated[str, Arg("--backend")] = "cuda"
    max_images: Annotated[int | None, Arg("--max-images")] = None
    confidence: Annotated[float, Arg("--confidence")] = 0.01
    iou_threshold: Annotated[float, Arg("--iou-threshold")] = 0.65
    log_every: Annotated[int, Arg("--log-every")] = 100
    use_fixture_input: Annotated[bool, Arg("--use-fixture-input")] = False


def load_arguments(directory: Path, device: jax.Device) -> list[jax.Array]:
    shapes = [
        tuple(int(dim) for dim in line.split(",") if dim)
        for line in (directory / "arguments.txt").read_text().splitlines()
    ]
    arguments = []
    for index, shape in enumerate(shapes):
        host = np.fromfile(directory / f"arg-{index:04}.f32", dtype="<f4").reshape(shape)
        arguments.append(jax.device_put(host, device))
    return arguments


def letterbox_bgr(path: Path) -> tuple[np.ndarray, float]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        ratio = min(SIZE / image.width, SIZE / image.height)
        width = int(image.width * ratio)
        height = int(image.height * ratio)
        resized = np.asarray(
            image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
        )
    output = np.full((SIZE, SIZE, 3), 114.0, dtype=np.float32)
    output[:height, :width] = resized[..., ::-1]
    return output[None], ratio


def decode(
    heads: list[np.ndarray], confidence: float, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = []
    scores = []
    classes = []
    for head, stride in zip(heads, (8, 16, 32), strict=True):
        _, height, width, channels = head.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        rows = head[0]
        centers = (rows[..., :2] + np.stack((grid_x, grid_y), axis=-1)) * stride
        sizes = np.exp(rows[..., 2:4]) * stride
        xyxy = np.concatenate((centers - sizes * 0.5, centers + sizes * 0.5), axis=-1)
        objectness = 1.0 / (1.0 + np.exp(-rows[..., 4:5]))
        class_scores = objectness * (1.0 / (1.0 + np.exp(-rows[..., 5:])))
        ys, xs, class_ids = np.nonzero(class_scores >= confidence)
        boxes.append(xyxy[ys, xs])
        scores.append(class_scores[ys, xs, class_ids])
        classes.append(class_ids.astype(np.int64))
    if not scores or sum(score.size for score in scores) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    all_boxes = np.concatenate(boxes)
    all_scores = np.concatenate(scores)
    all_classes = np.concatenate(classes)
    kept = batched_nms(
        torch.from_numpy(all_boxes),
        torch.from_numpy(all_scores),
        torch.from_numpy(all_classes),
        iou_threshold,
    )[:300].numpy()
    return all_boxes[kept], all_scores[kept], all_classes[kept]


def main() -> None:
    args = Args.parse_args()
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    backend = xla_bridge.get_backend(args.backend)
    device = backend.devices()[0]
    module = (args.fixture / "module.mlir").read_text()
    executable = backend.compile_and_load(module, [device])
    arguments = load_arguments(args.fixture, device)
    dynamic_input = int((args.fixture / "dynamic-input.txt").read_text())
    visible_outputs = int((args.fixture / "visible-outputs.txt").read_text())
    if visible_outputs != 3:
        raise ValueError(f"expected three YOLOX heads, got {visible_outputs}")

    document = json.loads(args.annotations.read_text())
    records = document["images"]
    if args.max_images is not None:
        records = records[: args.max_images]
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        image, ratio = letterbox_bgr(args.images / record["file_name"])
        if not args.use_fixture_input:
            arguments[dynamic_input] = jax.device_put(image, device)
        elif index != 1 or len(records) != 1:
            raise ValueError("--use-fixture-input requires exactly one image")
        outputs = executable.execute(arguments)
        heads = [np.asarray(output) for output in outputs[:visible_outputs]]
        boxes, scores, classes = decode(heads, args.confidence, args.iou_threshold)
        boxes /= ratio
        for box, score, class_id in zip(boxes, scores, classes, strict=True):
            x0, y0, x1, y1 = box.tolist()
            results.append(
                {
                    "image_id": record["id"],
                    "category_id": COCO_CATEGORY_IDS[int(class_id)],
                    "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                    "score": float(score),
                }
            )
        if index % args.log_every == 0 or index == len(records):
            elapsed = time.perf_counter() - started
            print(f"images={index}/{len(records)} detections={len(results)} elapsed_s={elapsed:.1f}")
    args.output.write_text(json.dumps(results, separators=(",", ":")))


if __name__ == "__main__":
    main()
