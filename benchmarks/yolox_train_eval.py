"""Run uninterrupted YOLOX training and pause at checkpoints for EMA COCO AP."""

from __future__ import annotations

import math
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Annotated

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rxla_log import Evaluation, Metric, RunFinished, RunLog, RunStarted
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    plugin: Annotated[Path, Arg(help="trusted CUDA PJRT plugin")]
    train_records: Annotated[Path, Arg(help="full COCO train TFRecord")]
    coco_root: Annotated[Path, Arg(help="COCO root containing annotations and val2017")]
    output: Annotated[Path, Arg(help="new run output directory")]
    epochs: Annotated[int, Arg("--epochs")] = 300
    batch_size: Annotated[int, Arg("--batch-size")] = 8
    gradient_accumulation: Annotated[int, Arg("--gradient-accumulation")] = 8
    samples: Annotated[int, Arg("--samples")] = 118_287
    eval_every_epochs: Annotated[int, Arg("--eval-every-epochs")] = 10
    log_every: Annotated[int, Arg("--log-every")] = 100
    run_id: Annotated[str | None, Arg("--run-id")] = None
    confidence: Annotated[float, Arg("--confidence")] = 0.01
    iou_threshold: Annotated[float, Arg("--iou-threshold")] = 0.65
    skip_build: Annotated[bool, Arg("--skip-build")] = False
    max_eval_images: Annotated[int | None, Arg("--max-eval-images")] = None
    short_schedule: Annotated[bool, Arg("--short-schedule")] = False
    keep_detections: Annotated[bool, Arg("--keep-detections")] = False
    compile_cache: Annotated[Path, Arg("--compile-cache")] = Path(
        ".cache/yolox/compile-cache-accuracy"
    )


CHECKPOINT = re.compile(r"^checkpoint_step=(\d+) output=(.+)$")


def command(*parts: object) -> list[str]:
    return [str(part) for part in parts]


def evaluate(
    evaluator_binary: Path,
    args: Args,
    checkpoint: Path,
    step: int,
    directory: Path,
    export_stablehlo: bool,
) -> tuple[list[float], float]:
    detections = directory / f"detections-step-{step}.json"
    invocation = command(
        evaluator_binary,
        args.plugin,
        checkpoint,
        args.coco_root / "annotations/instances_val2017.json",
        args.coco_root / "val2017",
        detections,
        "--ema",
        "--confidence",
        args.confidence,
        "--iou-threshold",
        args.iou_threshold,
        "--log-every",
        500,
        "--preallocate",
        "false",
    )
    if export_stablehlo:
        invocation.extend(["--export-jax", str(directory / "stablehlo-inference")])
    if args.max_eval_images is not None:
        invocation.extend(["--max-images", str(args.max_eval_images)])
    started = time.perf_counter()
    subprocess.run(invocation, check=True)

    prediction_rows = json.loads(detections.read_text())
    if not prediction_rows:
        print("COCO evaluator: no detections; all AP/AR metrics are zero")
        if not args.keep_detections:
            detections.unlink()
        return [0.0] * 12, time.perf_counter() - started
    ground_truth = COCO(str(args.coco_root / "annotations/instances_val2017.json"))
    predictions = ground_truth.loadRes(prediction_rows)
    coco = COCOeval(ground_truth, predictions, "bbox")
    coco.evaluate()
    coco.accumulate()
    coco.summarize()
    stats = [float(value) for value in coco.stats]
    if not args.keep_detections:
        detections.unlink()
    return stats, time.perf_counter() - started


def log_evaluation(
    log: RunLog,
    directory: Path,
    run_id: str,
    step: int,
    stats: list[float],
    seconds: float,
) -> None:
    names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = [Metric(name=name, scalar=value) for name, value in zip(names, stats)]
    metrics.append(Metric(name="evaluation_seconds", scalar=seconds, unit="s"))
    event = Evaluation(step=step, dataset="coco2017-val", metrics=metrics)
    timestamp = time.time_ns()
    log.write(event, timestamp)
    with RunLog(directory / f"evaluation-step-{step}.mcap") as snapshot:
        snapshot.write(
            RunStarted(
                run_id=run_id,
                model="yolox-tiny-evaluation",
                producer="benchmarks/yolox_train_eval.py",
            ),
            timestamp,
        )
        snapshot.write(event, timestamp)
        snapshot.write(
            RunFinished(run_id=run_id, final_step=step, succeeded=True),
            timestamp,
        )


def main() -> None:
    args = Args.parse_args()
    if min(
        args.epochs,
        args.batch_size,
        args.gradient_accumulation,
        args.samples,
        args.eval_every_epochs,
        args.log_every,
    ) <= 0:
        raise ValueError("counts and intervals must be positive")
    if os.name != "posix":
        raise RuntimeError("checkpoint evaluation currently requires POSIX SIGSTOP/SIGCONT")
    root = Path(__file__).resolve().parent.parent
    args.plugin = args.plugin.resolve()
    args.train_records = args.train_records.resolve()
    args.coco_root = args.coco_root.resolve()
    args.output = args.output.resolve()
    args.compile_cache = args.compile_cache.resolve()
    if args.max_eval_images is not None and args.max_eval_images <= 0:
        raise ValueError("--max-eval-images must be positive")
    if not args.skip_build:
        subprocess.run(
            command(
                "cargo",
                "build",
                "-p",
                "rxla-safetensors",
                "--release",
                "--features",
                "model,disk-cache",
                "--example",
                "yolox_coco_train_step",
                "--example",
                "yolox_coco_eval",
            ),
            cwd=root,
            check=True,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    training_binary = root / "target/release/examples/yolox_coco_train_step"
    evaluator_binary = root / "target/release/examples/yolox_coco_eval"
    for binary in (training_binary, evaluator_binary):
        if not binary.is_file():
            raise FileNotFoundError(binary)

    effective_batch = args.batch_size * args.gradient_accumulation
    steps_per_epoch = math.ceil(args.samples / effective_batch)
    checkpoint_every = steps_per_epoch * args.eval_every_epochs
    total_steps = steps_per_epoch * args.epochs
    run_id = args.run_id or f"yolox-{time.time_ns()}"
    training_state = args.output / "training-final.safetensors"
    model = args.output / "model-ema.safetensors"
    training_log = args.output / "training.mcap"
    evaluation_log = args.output / "evaluation.mcap"
    stablehlo = args.output / "stablehlo-training"

    invocation = command(
        training_binary,
        args.plugin,
        "--epochs",
        args.epochs,
        "--batch-size",
        args.batch_size,
        "--gradient-accumulation",
        args.gradient_accumulation,
        "--checkpoint-every",
        checkpoint_every,
        "--training-output",
        training_state,
        "--output",
        model,
        "--event-log",
        training_log,
        "--run-id",
        run_id,
        "--stablehlo-dir",
        stablehlo,
        "--compile-cache",
        args.compile_cache,
        "--log-every",
        args.log_every,
        "--preallocate",
        "false",
        "tfrecord",
        args.train_records,
    )
    if args.short_schedule:
        source = invocation[-2:]
        del invocation[-2:]
        invocation.extend(
            [
                "--warmup-steps",
                "0",
                "--no-augmentation-steps",
                "0",
                "--minimum-rate-steps",
                "0",
                *source,
            ]
        )

    with RunLog(evaluation_log, chunk_size=1) as log:
        log.write(
            RunStarted(
                run_id=run_id,
                model="yolox-tiny-evaluation",
                producer="benchmarks/yolox_train_eval.py",
                metadata=[
                    Metric(name="total_steps", integer=total_steps),
                    Metric(name="eval_every_steps", integer=checkpoint_every),
                ],
            ),
            time.time_ns(),
        )
        process = subprocess.Popen(
            invocation,
            cwd=root,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        evaluated: set[int] = set()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                match = CHECKPOINT.fullmatch(line.rstrip("\n"))
                if match is None:
                    continue
                step = int(match.group(1))
                checkpoint = Path(match.group(2))
                if not checkpoint.is_absolute():
                    checkpoint = root / checkpoint
                os.kill(process.pid, signal.SIGSTOP)
                try:
                    stats, seconds = evaluate(
                        evaluator_binary,
                        args,
                        checkpoint,
                        step,
                        args.output,
                        export_stablehlo=not evaluated,
                    )
                    log_evaluation(log, args.output, run_id, step, stats, seconds)
                    log.flush()
                    evaluated.add(step)
                finally:
                    os.kill(process.pid, signal.SIGCONT)
            status = process.wait()
            if status != 0:
                raise subprocess.CalledProcessError(status, invocation)

            stats, seconds = evaluate(
                evaluator_binary,
                args,
                training_state,
                total_steps,
                args.output,
                export_stablehlo=not evaluated,
            )
            log_evaluation(log, args.output, run_id, total_steps, stats, seconds)
            log.write(
                RunFinished(run_id=run_id, final_step=total_steps, succeeded=True),
                time.time_ns(),
            )
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


if __name__ == "__main__":
    main()
