"""Evaluate a standard COCO detection-result JSON file."""

from pathlib import Path
from typing import Annotated

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    annotations: Annotated[Path, Arg(help="COCO instances JSON")]
    detections: Annotated[Path, Arg(help="COCO detection-result JSON")]


def main() -> None:
    args = Args.parse_args()
    ground_truth = COCO(args.annotations)
    # pycocotools only recognizes `str`, not `os.PathLike`, as a JSON path.
    predictions = ground_truth.loadRes(str(args.detections))
    evaluator = COCOeval(ground_truth, predictions, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()
