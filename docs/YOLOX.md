# YOLOX training and inference

RXLA implements the official YOLOX-tiny topology, checkpoint-compatible model
paths, image preprocessing, decode/NMS, SimOTA matching, reverse-mode training,
and resident grouped momentum-SGD updates. CPU and CUDA use the same model and training code;
only the PJRT plugin path changes.

## Prepare official weights

Download the official `yolox_tiny.pth`, then convert it once with the locked uv
environment:

```sh
uv run python benchmarks/convert_yolox.py \
  yolox_tiny.pth yolox_tiny.safetensors
```

The converter records the `rxla-yolox-v1` format marker and maps PyTorch
BatchNorm state names to RXLA's model schema.

The deterministic training oracle checks the converted checkpoint, model
topology, SimOTA matcher, losses, reverse-mode head gradients, selected
full-model parameter gradients, BatchNorm state, and one complete Nesterov SGD
update against the official-compatible PyTorch implementation without image
resize or decoding differences:

```sh
uv run --no-sync python benchmarks/yolox_oracle.py \
  /path/to/xla_cuda_plugin.so \
  yolox_tiny.pth yolox_tiny.safetensors .cache/yolox/oracle
```

At the default 64×64 oracle size, all five CSPDarknet stages, three PAFPN
features, and three raw heads are checked. It then compares dense SimOTA box,
objectness, and class assignments plus total, IoU, objectness, class, and L1
losses. On the CUDA 13 PJRT plugin, the largest observed raw-head absolute error
was `2.10e-5`, the assignments selected the same two foreground anchors, and
the largest loss error was `4.30e-6`. The largest absolute gradient error over
all three raw heads was `2.54e-7`. A second graph runs the complete model in
training mode and checks matcher outputs, loss, selected parameter gradients,
BatchNorm state, and the resulting parameter updates. Gradients and updates are
sampled at the stem, immediately before and after the SPP max-pool branches,
and at the first class predictor. Their largest observed relative L2 errors
were both `7.59e-4` (`rtol=2e-3`); the training loss differed by `1.44e-5`.
The optimizer comparison uses the training configuration's `0.9` momentum,
Nesterov update, and `5e-4` decay for non-BatchNorm weights. It now checks the
resident momentum buffers themselves as well as the resulting parameters.
Their largest observed relative L2 error was `7.59e-4`. The stem running mean
and variance differed by at most `9.54e-7` and `2.45e-4`, respectively. RXLA
applies the same Bessel correction as PyTorch when updating running variance
while retaining population variance for the current batch's normalization.

Passing `--training-steps N` additionally emits a multi-step diagnostic. Every
RXLA step exports its detached device SimOTA assignment and PyTorch replays the
same assignment, separating optimizer/state drift from discrete matcher
changes. This is deliberately not a strict gate: nonlinear training trajectories
amplify backend convolution rounding. At 416x416, step-one selected gradients
were within `4.90e-5` relative L2, while step-two and step-three losses differed
by `0.00457` and `0.06859`; final selected momentum buffers differed by
1.8%-4.7%. At the artificial 64x64, batch-one oracle size the amplification is
larger. A separate three-step scalar Nesterov runtime test checks loss,
parameter, and momentum against the closed-form recurrence at every step, so
the diagnostic drift is not silently accepted as optimizer semantics.

## Inference

```sh
cargo run -p rxla-safetensors --release --features model \
  --example yolox_infer -- \
  /path/to/xla_cuda_plugin.so \
  yolox_tiny.safetensors input.jpg detections.png
```

The example letterboxes to 416×416, executes the three YOLOX heads, performs
decode and class-aware NMS, and writes an annotated image.

Training quality is checked with the standard COCO evaluator rather than loss
alone. Export predictions from an inference checkpoint, then compute the twelve
official bbox metrics with `pycocotools`:

```sh
cargo run -p rxla-safetensors --release --features model \
  --example yolox_coco_eval -- \
  /path/to/xla_cuda_plugin.so yolox_tiny.safetensors \
  /data/datasets/coco2017/annotations/instances_val2017.json \
  /data/datasets/coco2017/val2017 detections.json

uv run --no-sync python benchmarks/coco_eval.py \
  /data/datasets/coco2017/annotations/instances_val2017.json detections.json
```

The evaluator defaults to YOLOX-tiny's official 416×416 test size, `0.01`
confidence threshold, `0.65` class-aware NMS threshold, and at most 300 retained
detections per image. `--max-images` provides a short pipeline smoke test, but
only all 5,000 `val2017` images produce comparable COCO AP. Both inference
checkpoints and resumable training snapshots are accepted; evaluating a
training snapshot measures its raw model parameters, while the normal
`--output` checkpoint measures the EMA parameters used for deployment.
Pass `--ema` to evaluate the packed EMA directly from a resumable training
snapshot, without first rewriting it as an inference checkpoint.

Training intentionally does not run COCO AP in the hot loop. Loss components,
learning rate, foreground count, and data-pipeline timing are cheap health
signals; decode, NMS, and all 5,000 validation images belong in an asynchronous
checkpoint evaluator. The final metrics still come from the standard
`pycocotools.COCOeval`, not from a framework-specific approximation.

For an independent execution check, the evaluator can export the exact
StableHLO module and first-image ABI (including EMA parameters and BatchNorm
state), then JAX can compile that module without tracing an alternate model:

```sh
cargo run -p rxla-safetensors --release --features model \
  --example yolox_coco_eval -- \
  /path/to/xla_cpu_plugin.so training.safetensors \
  /data/datasets/coco2017/annotations/instances_val2017.json \
  /data/datasets/coco2017/val2017 rxla-detections.json \
  --ema --max-images 1 --export-jax /tmp/yolox-jax

JAX_PLATFORMS=cpu uv run --no-sync python \
  benchmarks/yolox_stablehlo_coco.py /tmp/yolox-jax \
  /data/datasets/coco2017/annotations/instances_val2017.json \
  /data/datasets/coco2017/val2017 jax-detections.json \
  --backend cpu --max-images 1 --use-fixture-input
```

`--use-fixture-input` isolates StableHLO execution and postprocessing by using
the exact pixels exported by Rust. Without it, the Python runner preprocesses
images with Pillow; interpolation differences can change boxes near confidence
or NMS thresholds. Either detection JSON can be passed to
`benchmarks/coco_eval.py` for the official twelve COCO bbox metrics.

As a baseline, the converted official `yolox_tiny.pth` produced 249,721
post-NMS detections over all 5,000 images and reached `AP=0.328`, `AP50=0.504`,
and `AP75=0.347`. The primary AP exactly matches the published YOLOX-tiny
`32.8` result. This closes the inference-side accuracy loop; training changes
must retain the one-step numerical oracle and are judged end to end by this
same full-validation AP path.

## Training on COCO

```sh
cargo run -p rxla-safetensors --release --features model \
  --example yolox_coco_train_step -- \
  /path/to/xla_cuda_plugin.so \
  yolox_tiny.safetensors \
  --size 416 --batch-size 8 --steps 100 \
  --output yolox_tiny_trained.safetensors \
  coco \
  /data/datasets/coco2017/annotations/instances_train2017.json \
  /data/datasets/coco2017/train2017 \
  391895
```

The model is compiled once. One stateful XLA program executes the model forward,
device-side SimOTA assignment, loss, backward pass, BatchNorm updates, and SGD
update. SimOTA receives at most 120 padded targets per image, matching the
official training transform, and lowers center filtering, pairwise IoU,
classification cost, dynamic-k selection, and conflict resolution to StableHLO.
Assignments are detached before entering the differentiable loss. The output
checkpoint excludes optimizer moments but includes parameters and model
BatchNorm state, so it can be passed directly to `yolox_infer`.

The default training size is YOLOX-tiny's 416×416. Multiscale training is also
enabled by default and selects a 32-pixel multiple in 320..640 every ten
optimizer updates; pass `--random-resize false` only for controlled experiments.
The learning-rate scheduler follows the official trainer's update ordering.
SGD is initialized with the zero warmup rate, so the first optimizer update is
a zero-rate update. The trainer then installs `update_lr(1)` for the second
update and repeats that post-step ordering throughout training.

The default total learning rate follows the official trainer and scales
`basic_lr_per_img = 0.01 / 64` by the effective batch size. For example,
`--batch-size 8 --gradient-accumulation 8` represents batch 64 and uses `0.01`.
An explicit `--learning-rate` overrides the resulting total optimizer rate
rather than the per-image constant.

Gradient accumulation is part of the compiled stateful transform, not a host
approximation. Each microbatch adds to resident F32 gradient buffers. Only the
last microbatch divides the sum by `--gradient-accumulation`, applies weight
decay and Nesterov momentum, updates parameters, and advances model EMA. The
learning-rate schedule, random-size schedule, `--steps`, `--run-steps`, and
checkpoint `completed_steps` therefore count optimizer updates. Every
microbatch uses its own training-time BatchNorm statistics for forward and
backward, as on one local rank of the official distributed run. Persistent
running mean and variance are committed only on the final microbatch, so batch
8 accumulated eight ways advances them once per global update rather than
silently applying momentum eight times. Resumable snapshots include the
accumulation counter and buffers and are written only at optimizer-step
boundaries.

The initial checkpoint is optional. Omitting both the positional weights path
and `--resume` initializes every parameter from the model schema, including
Kaiming-uniform convolution weights and the official 0.01 predictor priors.
This is the normal path for reproducing training from scratch; supplying
converted weights is a fine-tuning experiment and its BatchNorm statistics may
temporarily adapt away from their pretrained validation distribution.

Freshly initialized models use the official YOLOX policies: convolution weights
use PyTorch-compatible Kaiming-uniform initialization, BatchNorm uses
`epsilon=1e-3` and `momentum=0.03`, and objectness/classification predictor
biases start at `logit(0.01)`.

The example accepts a static `--batch-size`; variable numbers of targets are
padded and masked before transfer. JPEG decode and augmentation run across the
batch with Rayon. Dataset shuffling, augmentation, and prefetch remain in the
data pipeline rather than the YOLO model definition.

CUDA training can use convolution-only automatic mixed precision with
`--compute-dtype bf16`. Convolution inputs and F32 master weights are cast for
the cuDNN operation and its adjoints, then converted back at the layer boundary.
BatchNorm, SimOTA, losses, model state, momentum, and parameter updates remain
F32. This is deliberately narrower than whole-program dtype demotion.

Training defaults to `--convolution-precision default`, allowing XLA/cuDNN to
use accelerator-native reduced-precision contraction algorithms such as TF32
for F32 convolutions. Use `--convolution-precision highest` for the strict
StableHLO precision contract used by the numerical oracle. Precision is part of
`Conv2dOptions` and propagates through both input- and weight-gradient
convolutions rather than being a process-global flag.

The PyTorch training benchmark accepts the same
`--convolution-precision default|highest` option. It controls cuDNN TF32 so
performance and evolving-loss comparisons do not accidentally compare RXLA's
strict F32 convolutions with PyTorch's default TF32 convolutions. Dynamic
SimOTA assignments are discontinuous, so multi-step loss trajectories remain a
diagnostic rather than a numerical oracle; the fixed-assignment one-step oracle
is the authoritative gradient and update check.

For sequential training, COCO can instead be stored as standard TFRecord
framing with Apache Fory Standard Row payloads. The JPEG remains compressed;
Fory provides typed, borrowed views of its bytes and the box/label arrays.

```sh
uv run --no-sync python benchmarks/fory_tfrecord.py \
  /data/datasets/coco2017 \
  /data/datasets/coco2017-fory-4096.tfrecord \
  --samples 4096

cargo run -p rxla-safetensors --release --features model \
  --example yolox_coco_train_step -- \
  /path/to/xla_cuda_plugin.so yolox_tiny.safetensors \
  --compute-dtype bf16 \
  --size 416 --batch-size 8 \
  --shuffle-buffer 1024 --seed 7 \
  --steps 100 \
  tfrecord /data/datasets/coco2017-fory-4096.tfrecord
```

The exporter applies the same COCO annotation policy as official YOLOX: crowd
annotations are omitted, coordinates are clipped to the image, and invalid or
zero-area annotations are discarded. The training reader requires the adjacent
`.tfrecord.json` manifest and checks this policy in addition to the container,
Fory version, and schema hash before reading any records. Older exports without
`annotation_policy: "yolox-coco-v1"` must be regenerated rather than silently
training on different targets. Each
TFRecord length and payload CRC32C is then verified while streaming.
The reader remains sequential; a deterministic bounded pool above the storage
layer supplies randomized samples without seeking. This is also the sampling
boundary for compound augmentations: each training Mosaic consumes four samples
from the pool without replacement, applies the official random-center layout
and YOLOX-tiny's `0.5..1.5` affine scale range. Tiny disables MixUp by default;
when enabled explicitly, it draws another nonempty sample from the same pool
with `0.5..1.5` jitter, optional flip, crop, and a 50/50 blend. HSV
perturbation and a 0.5-probability horizontal flip then run on the composite.
Post-affine boxes follow the current YOLOX 0.3 training path: affine itself
retains clipped boxes, then `TrainTransform` removes boxes whose scaled minimum
side is at most one pixel. If that would remove every label, it falls back to
the pre-HSV/pre-flip image and labels. The stricter width/area/aspect candidate
filter belonged to YOLOX 0.1's older `random_perspective` path and is not mixed
into the current recipe.
The final no-augmentation phase consumes one image, disables Mosaic/MixUp, and
enables the official raw-box L1 loss. Its weight is a runtime scalar, so the
phase transition reuses the same compiled forward/backward/SGD program. Decoded
RGB images are converted to the BGR channel order used by the official OpenCV
input pipeline before entering the model.

Both COCO-file and TFRecord sources roll over at end of input, so `--steps` can
span any number of epochs instead of stopping after one physical pass. A COCO
`image_id` cursor applies only to the first partial epoch; later epochs restart
at the beginning. TFRecord input reopens the same validated file. A genuinely
empty source still terminates with an error rather than spinning forever, and
the bounded shuffle pool remains continuous across storage rollovers.

Long runs can write and resume the complete training transformation, rather
than only inference weights. `--training-output state.safetensors` includes
parameters, BatchNorm state, SGD momentum, packed model EMA, and the EMA update
counter. `--resume state.safetensors` validates the global schedule, model and
data-pipeline configuration before restoring exact dtypes and shapes. Use
`--run-steps N` to stop after `N` updates while retaining the global epoch
schedule:

```sh
# COCO train2017 has 118,287 images. Batch 8 accumulated eight ways represents
# the official effective batch 64. --epochs derives 554,700 optimizer steps
# and the exact official phase boundaries from the source manifest.
yolox_coco_train_step PLUGIN WEIGHTS \
  --batch-size 8 --gradient-accumulation 8 \
  --epochs 300 --run-steps 1000 \
  --checkpoint-every 1000 \
  --training-output step-1000.safetensors \
  tfrecord train.tfrecord

# Continue the same schedule and deterministic sample stream.
yolox_coco_train_step PLUGIN --resume step-1000.safetensors \
  --batch-size 8 --gradient-accumulation 8 --epochs 300 \
  --training-output final.safetensors \
  tfrecord train.tfrecord
```

`--epochs` uses the COCO image count or TFRecord manifest sample count and
`ceil(samples / effective_batch)` iterations per epoch. It reproduces the
official five warmup epochs, starts Mosaic-off plus L1 at epoch 285 (16 epochs
remain, matching the trainer's actual boundary), and keeps the minimum LR tail
at 15 epochs. `--steps` remains available for controlled experiments; in that
mode its values are optimizer updates and the two final phases default to the
same 15/300 fraction unless overridden separately. The 4,096-sample TFRecord
used by microbenchmarks is intentionally not an accuracy-training dataset;
accuracy runs need a manifest containing all of train2017.

Long invocations can combine `--training-output state.safetensors` with
`--checkpoint-every N`. Each boundary writes an immutable sibling such as
`state.step-10000.safetensors`, while successful completion writes the requested
`state.safetensors`. Publication is atomic and no-clobber, so an interrupted
multi-hour run loses at most `N` updates and never exposes a partially written
checkpoint. Existing milestone files are rejected rather than overwritten.

Training can additionally emit indexed Protobuf/MCAP events and the exact
StableHLO for every multi-scale variant compiled by the run:

```sh
yolox_coco_train_step PLUGIN \
  --epochs 300 --batch-size 8 --gradient-accumulation 8 \
  --event-log train.mcap --run-id yolox-tiny-coco-v4 \
  --stablehlo-dir stablehlo-training \
  --training-output training-final.safetensors \
  --output model-ema.safetensors \
  tfrecord /data/datasets/coco2017-train2017.tfrecord
```

The event stream contains run lifecycle, typed loss/timing metrics, input size,
foreground/target counts, and published checkpoints. Logs use the standalone
`rxla-log` schema and can be read by both Rust and Python. StableHLO files are
named by compiled input size and written with no-clobber semantics.

For accuracy runs, the Python supervisor keeps one training process resident,
pauses it at milestone checkpoints, evaluates the packed EMA on COCO val2017,
then resumes it:

```sh
uv run --frozen python benchmarks/yolox_train_eval.py \
  /path/to/cuda_pjrt_plugin.so \
  /data/datasets/coco2017-train2017.tfrecord \
  /data/datasets/coco2017 \
  .cache/yolox/runs/yolox-tiny-v4 \
  --epochs 300 --eval-every-epochs 10 --log-every 100
```

It produces separate `training.mcap` and `evaluation.mcap` logs, resumable
milestone snapshots, the final EMA model, training StableHLO variants, and the
exact inference StableHLO plus ABI fixture used by the first evaluation.
Evaluation uses non-preallocating PJRT clients so the paused trainer and
evaluator can coexist on one GPU without each reserving 75% of VRAM. Raw COCO
detection JSON is deleted after metrics are recorded unless
`--keep-detections` is supplied.

An interrupted supervised run resumes from an immutable milestone snapshot;
the original run ID is required so training and evaluation segments retain one
logical identity:

```sh
uv run --frozen python benchmarks/yolox_train_eval.py \
  /path/to/cuda_pjrt_plugin.so \
  /data/datasets/coco2017-train2017.tfrecord \
  /data/datasets/coco2017 \
  .cache/yolox/runs/yolox-tiny-v4 \
  --epochs 300 --eval-every-epochs 10 --log-every 100 \
  --run-id yolox-tiny-coco-v4 \
  --resume .cache/yolox/runs/yolox-tiny-v4/training-final.step-18490.safetensors
```

Resume preserves prior artifacts and writes new
`training-resume-step-N.mcap` and `evaluation-resume-step-N.mcap` segments.
The Rust trainer validates the snapshot's full schedule and optimizer metadata;
the supervisor rejects mutable/final snapshots whose filename does not encode
the completed milestone. Existing StableHLO exports are reused.

Create that full accuracy dataset explicitly with:

```sh
uv run --no-sync python benchmarks/fory_tfrecord.py \
  /data/datasets/coco2017 \
  /data/datasets/coco2017-train2017.tfrecord \
  --samples 118287
```

To reconstruct the shuffle pool and variable Mosaic/MixUp consumption exactly,
resume replays the sampling plan from the beginning without decoding or
augmenting images. The source and its ordering must therefore remain unchanged.
This costs sequential compressed-record reads but does not repeat device work.
In a CUDA check, a one-step snapshot resumed for step two produced all 630
parameter, BatchNorm, optimizer, and EMA tensors bit-identically to an
uninterrupted two-step run; replaying the first TFRecord step took 1.7 ms.

## Verified baseline

### Accuracy status

Inference is matched end to end: the converted official checkpoint reproduces
the published `AP=0.328` on all 5,000 COCO val2017 images. Training is not yet
claimed to reproduce that AP. The fixed-input oracle establishes one-update
model, loss, gradient, BatchNorm, momentum-SGD, and EMA semantics, while a
from-scratch run establishes that the real COCO input pipeline remains finite
and resumable. Neither check substitutes for a complete training curve.

A pre-v3 full-dataset diagnostic used the 300-epoch schedule, effective batch
64 (`batch=8`, eight accumulated microbatches), default multi-scale training,
Mosaic, and model EMA. At update 1,000 the loss had fallen from `17.845` to the
`12`-`15` range; after resuming to update 2,000 it was typically `10`-`12`.
Both the raw model and EMA rounded to `AP=0.000` on val2017, with
`AR@100=0.002`. That run exposed a one-update LR-order mismatch: it used
`update_lr(1)` for the first optimizer step, while official YOLOX initializes
SGD at the zero warmup rate and installs `update_lr(1)` only after that step.
Its snapshots are intentionally rejected by v3 and its AP is not accuracy
evidence. The next accuracy gate is a matched PyTorch/RXLA curve at fixed
milestones through the end of warmup, followed by a complete 300-epoch COCO AP
run.

The clean v3 run has now completed the five-epoch warmup boundary at update
9,245. It executed 73,960 accumulated microbatches (591,680 augmented output
images) with all eleven 320..640 multi-scale variants, reached the scheduled
`9.99784e-3` learning rate, and remained finite. On the complete 5,000-image
val2017 set, the raw model reached `AP=0.034`, `AP50=0.082`, and `AR@100=0.160`;
the exported EMA reached `AP=0.049`, `AP50=0.109`, and `AR@100=0.190`. The EMA
gain and nonzero localization metrics establish genuine early convergence.
They do not yet satisfy or replace the final 300-epoch AP gate.

The supervised v4 accuracy run exercises the production checkpoint/evaluation
loop every ten epochs. At update 18,490 (epoch 10), its EMA reached
`AP=0.1061`, `AP50=0.2169`, `AP75=0.0932`, and `AR@100=0.2720` on all 5,000
val2017 images. Evaluation took 58.5 seconds on the same RTX 5080 while the
trainer remained paused with its CUDA context resident. The milestone produced
an immutable resumable snapshot, a finalized evaluation MCAP, and the exact
inference StableHLO/ABI fixture; the trainer then resumed the unchanged
300-epoch run. At update 36,980 (epoch 20), the next periodic evaluation
reached `AP=0.1654`, `AP50=0.3056`, `AP75=0.1682`, and `AR@100=0.3355` in
58.0 seconds. The second immutable snapshot and finalized MCAP were published,
and the paused trainer resumed again. At update 55,470 (epoch 30), EMA reached
`AP=0.1972`, `AP50=0.3501`, `AP75=0.2017`, and `AR@100=0.3678`; evaluation
took 56.9 seconds and the trainer resumed at update 55,500. These are
intermediate convergence points, not the final AP claim.

The remaining intentional data-order difference is accepted by the training
quality contract. Official YOLOX permutes one primary sample per output and
draws the other three Mosaic images uniformly with replacement. The sequential
TFRecord path instead draws all four from a bounded shuffled pool without
replacement. Their marginal image distribution is similar, but their exact
sample correlations and epoch coverage differ. Consequently the end-to-end
gate is statistical rather than bit-exact: using the same complete COCO
val2017 evaluator, the final EMA must be within one absolute AP point of the
official YOLOX-tiny `32.8` result (at least `31.8 AP`). The fixed-input oracle
remains strict for operator, loss, gradient, SGD, BatchNorm, and EMA semantics.

On an RTX 5080 with JAX's CUDA 13 PJRT plugin, device SimOTA selects exactly the
same oracle assignments as the official-compatible PyTorch implementation and
the retained Rust host reference. At 416×416 and batch size 8, a 20-step run
over the Fory TFRecord with default Mosaic, no MixUp, and two final
no-augmentation/L1 steps processed 160 output images in 869.0 ms, including
first-step execution warmup: 184.1 images/s. The unified
forward/SimOTA/loss/backward/SGD graph took 36.4–37.2 ms after warmup. A bounded
three-batch producer hid the 39–51 ms Mosaic preparation time; measured consumer
wait remained 0.0 ms. Compilation took 12.4 seconds. Timings vary with image
contents, target counts, host CPU, and PJRT/XLA versions.

For comparison, the preceding host-SimOTA design reached 124.9 images/s on the
same hardware. It required an additional approximately 8 ms inference pass,
9–11 ms of host matching, a raw-head download, and dense-target uploads per
step. Moving matching into XLA improved measured end-to-end throughput by about
47% while preserving oracle assignments and losses.

With Mosaic, affine, MixUp, HSV, and flip all enabled, the corresponding
twenty-step run took 1.482 seconds for 160 output images (107.9 images/s),
including first-step device warmup and two final no-augmentation/L1 steps.
Augmentation prepared a typical batch in 63–66 ms, but a three-batch bounded
prefetch kept measured consumer wait at 0.0 ms throughout the run.

A dedicated synthetic benchmark now runs the same U8 input pattern, 12 targets
per image, device SimOTA, loss, BatchNorm updates, Nesterov SGD, warmup count,
and per-step synchronization in RXLA and the official-compatible PyTorch model.
At 416×416, batch 8, 10 warmups, and 100 measured F32 steps, RXLA with default
convolution precision measured `16.871 ms` (`474.2 images/s`), while
`torch.compile` measured `21.453 ms` (`372.9 images/s`). A shorter 20-step BF16
run measured `12.901 ms` (`620.1 images/s`) for RXLA, `16.918 ms` (`472.9
images/s`) for `torch.compile`, and `24.189 ms` for eager PyTorch. The Python
loss still has a graph break at its per-image `Tensor.item()` SimOTA loop, while
RXLA compiles matching and optimization into the same program.

The same sparse-target graph completed a seeded 100-step BF16 run over the
Fory COCO stream at `502.1 images/s`. Consumer wait remained `0 ms`; late
steady device observations were `10.8–12.4 ms`. This is within the observed
run-to-run range of the earlier `513.6 images/s` host-pipeline result rather
than evidence of a separate end-to-end gain.

Host augmentation now parallelizes affine warp rows, HSV pixels, MixUp blend
rows, and letterbox conversion in addition to batch and Mosaic JPEG work. In a
paired 200-step BF16 run at 416×416, batch 8, three augmentation workers, and
the same executable-cache hit, the preceding scalar-pixel implementation
reached `388.1 images/s`; the parallel implementation reached `444.9
images/s`, a 14.6% end-to-end improvement. Typical Mosaic preparation fell
from `51–59 ms` to `26–39 ms` per batch, while the ordered bounded pipeline
retained deterministic sample order.

The real trainer now enables the official YOLOX parameter EMA by default,
including its `0.9998 * (1 - exp(-updates / 2000))` decay ramp. SGD and EMA are
one stateful device program, with no parameter download between them; the
export path writes EMA parameters under their canonical model names. The EMA
values are packed into one logical F32 vector for the hot update and split only
when a checkpoint is requested. On the same seeded 100-step BF16 workload, the
initial per-parameter implementation reached `447.6 images/s`; packing
parameters raised that to `470.6 images/s`. After folding floating model state
into the same packed operation, a subsequent run reached `490.8 images/s`,
versus `492.6 images/s` with `--disable-model-ema`. Late observed device time
was `10.9 ms` with packed EMA and `11.1 ms` in that no-EMA run, so the remaining
total-throughput difference is below the noise of these short end-to-end runs.
BatchNorm running statistics are initialized from their exact pre-forward
values, share the packed EMA update, and are exported under canonical state
paths; non-floating counters and RNG state are not averaged.

Metric observation is decoupled from device execution: the real trainer only
downloads and validates its six scalar metrics on the first, final, and
`--log-every` steps. Intermediate PJRT submissions retain their device-side
state dependencies without forcing a host synchronization. With
`--log-every 100`, the same seeded BF16/EMA workload reached `503.1 images/s`
instead of the preceding fully synchronized `490.8 images/s`. A matching
no-EMA asynchronous run reached `502.2 images/s`, so packed parameter and
BatchNorm EMA had no measurable throughput cost in this pair of short runs.

Device SimOTA normally retains the official per-image padding capacity of 120.
For datasets or augmentation policies with a known lower object-count ceiling,
`--target-capacity` reduces that static matching dimension; excess targets are
explicitly capped in the same way as the official 120-entry preprocessing
limit. On the RTX 5080 BF16 synthetic workload at 416×416, batch 8, and 12 valid
targets per image, capacity 120 measured `12.466 ms` (`641.8 images/s`), while
capacity 64 measured `11.779 ms` (`679.2 images/s`), a 5.8% throughput gain.
Capacity 32 measured `11.880 ms`, so smaller shapes are not monotonically faster
under XLA code generation. The default remains 120 for recipe compatibility;
this knob is fixed per compiled variant and does not create an automatic bucket
cross-product with multi-scale image sizes.

The trainer reports both the largest unpadded per-image target count and the
number of targets dropped by the selected capacity. This makes a lower capacity
an observable, dataset-specific optimization instead of a silent semantic
change. In a deterministic 200-step COCO/Mosaic run at 416x416, batch 8, and
BF16, the observed maximum was 67 targets per image. Capacity 64 dropped three
targets and reached `466.3 images/s`; capacity 80 dropped none and reached
`464.4 images/s`; the matching capacity-120 run reached `451.7 images/s`.
Capacity 80 therefore improved this short workload by 2.8% without changing its
targets, but it is not a replacement for the recipe-compatible default: a
different dataset, seed, or augmentation stream can have a higher maximum.

Host augmentation uses two independent controls. `--augmentation-workers`
sets the number of batches prepared concurrently, while `--prefetch` bounds the
ready-batch queue. Each batch already parallelizes its JPEG decodes and pixel
transforms through Rayon's global pool. Earlier in development, before those
inner transforms were parallel, changing 16 workers/16 prefetched batches to
4/32 improved a deterministic 200-step workload from `466.9`-`467.5` to
`478.5`-`482.7 images/s`.

That outer concurrency stopped helping after the inner pipeline was optimized.
The trainer now reports cumulative mean wait, augmentation, staging, image
upload, and device-training times so starvation can be measured directly. In
400-step BF16/COCO runs, 4/32, 2/16, and 1/8 respectively reached `547.0`,
`547.1`, and `546.7 images/s`; every configuration spent only `0.004 ms` per
step waiting for prepared data. The default is therefore one outer worker and
eight prefetched batches. This preserves throughput while avoiding nested
batch-level contention and bounding queued image memory more tightly. Both
values remain explicit CLI overrides for slower storage or different hosts.

CPU sampling then identified generic `image::imageops::resize` as the largest
remaining host hotspot: it accounted for 14,444 leaf samples, ahead of JPEG
entropy decoding (6,773), HSV (5,204), and affine interpolation (4,124).
Mosaic, MixUp, and letterbox resizing now use `fast_image_resize`'s SIMD RGB8
linear interpolation. A fixed resize reference agrees with OpenCV
`INTER_LINEAR` within one integer LSB per channel. In interleaved 400-step
BF16/COCO runs, the old implementation reached `467.0` and `495.0 images/s`;
the SIMD implementation reached `531.6` and `534.1 images/s`. This is a 7.4%
gain even against the faster old run and a 10.8% gain between pair averages.
The follow-up CPU profile no longer contained the former generic resize
hotspot; JPEG, HSV, and affine interpolation became the leading host costs.

The remaining scalar color and affine paths were then specialized without
changing their sampling contract. HSV conversion now operates directly in
OpenCV's byte-scale H/S/V domains, avoids redundant normalization and
`fmodf`, and matches fixed OpenCV augmentation references within two integer
LSBs. Its sampled leaf cost fell from 5,657 to 3,684 samples (-34.9%). The
affine warp uses a raw RGB8 bilinear sampler that is byte-identical to the
previous generic `image` implementation at in-bounds coordinates;
its leaf samples fell from 4,282 to 3,393 (-20.8%), while total `roundf`
samples fell from 2,584 to 1,151 before the final positive-value rounding was
inlined. Two 400-step runs after both changes reached `547.3` and `547.7
images/s`, versus `531.6` and `534.1 images/s` for the SIMD-resize baseline,
about 2.8% higher by pair average. JPEG entropy decoding is now the largest
sampled host hotspot.

RGB-to-HSV now also uses OpenCV's 12-bit fixed-point divisor tables and
rounding rules rather than an approximate floating-point conversion. Against
OpenCV 5.0, one million seeded random RGB pixels matched all three HSV bytes
exactly; the former implementation matched only 97.38% after rounding and had
a hue wraparound outlier. The remaining float HSV-to-RGB path matched 99.20%
of full adjusted RGB pixels exactly and every tested channel within one byte.
Two 400-step end-to-end runs reached `560.7` and `554.2 images/s`, statistically
unchanged from the prior 558.6 average, so this is retained as an official
augmentation-semantics correction rather than a speed claim.

Replacing the decoder was measured before adding another dependency. The
current `image` feature set already selects `zune-jpeg 0.5.15` with its x86
SIMD implementation. On 512 real COCO train JPEGs, three warm-cache sequential
passes through `image`/zune reached `759.8`, `763.3`, and `762.6 images/s`;
`zenjpeg 0.8.4` reached `731.3`, `730.3`, and `724.7 images/s` while producing
the same total RGB byte count. The prerelease decoder was therefore rejected:
it was 4%-5% slower in this workload and would not address the augmentation
kernel costs. Decoder choice should be re-evaluated with end-to-end concurrent
loading rather than inferred from single-image library benchmarks.

The known-JPEG training path does call the existing zune backend directly. It
avoids generic format detection, `DynamicImage`, and `image`'s defensive copy
of the encoded byte slice. On the same 512-image microbenchmark this reached
`788.2`, `787.0`, and `792.9 images/s`, 2%-3% above the wrapper. Two complete
400-step COCO runs reached `554.4` and `549.4 images/s` (551.9 average), versus
the previous corrected-affine pair's 546.6 average. This is a modest roughly
1% end-to-end gain, as expected now that decode overlaps device execution.

Affine border sampling also follows OpenCV's constant-border interpolation:
each out-of-range neighbor contributes the value 114 independently. The former
implementation replaced an entire output pixel with 114 whenever its sampling
center fell just outside the source, discarding valid in-range neighbors. A
fixed translated/sheared matrix now matches `cv2.warpAffine` byte-for-byte,
including all four image boundaries. Two real 400-step runs after the fix
reached `542.1` and `551.1 images/s`, so the corrected edge semantics have no
measurable throughput penalty.

Affine inverse-coordinate terms that depend only on the output row are now
hoisted out of the pixel loop, removing two multiplications and one subtraction
per output pixel without accumulating incremental-coordinate rounding error.
The OpenCV boundary reference remains byte-identical. Two complete 400-step
runs reached `544.7` and `562.6 images/s` (553.7 average); the roughly 0.3%
difference from the direct-JPEG pair average is within run-to-run variance, so
this is retained as a local kernel simplification rather than claimed as an
end-to-end speedup.

A fresh steady-state Nsight CPU sample after those changes still placed JPEG
entropy decoding first (`decode_mcu_block`, 5,556 leaf samples). Affine
bilinear interpolation followed at 3,930 samples and HSV adjustment at 3,092.
The affine sampler was checking bounds and recomputing the byte address for all
four neighbors separately for each RGB channel. It now loads each neighbor as
one RGB triplet, reducing twelve checks/address calculations to four while
preserving the exact interpolation order and OpenCV byte reference. Two
400-step runs reached `555.8` and `561.4 images/s` (558.6 average), roughly
0.9% above the preceding pair average.

Mosaic tile placement now copies the selected RGB rows directly into the
canvas. The former `crop_imm(...).to_image()` followed by `replace()` allocated
four temporary images and copied every selected tile twice; those two generic
image operations accounted for 581 and 612 leaf samples in the steady profile.
The direct-copy version has a rectangle-level unit test and preserves the same
coordinates. Two 400-step runs reached `555.5` and `562.0 images/s` (558.8
average), so the eliminated allocation and memory traffic do not change
throughput while the host pipeline remains overlapped with the device.

`--random-resize` enables YOLOX's ten-step multi-scale schedule. It selects a
square 32-pixel multiple in the inclusive `--random-size-min` to
`--random-size-max` range (official defaults `10..20`), precompiles only the
unique static sizes that this run will use, and switches at ten-step
boundaries. As in the official trainer, Mosaic, affine, HSV, and flip remain at
the base `--size`; the selected variant then performs PyTorch-compatible
half-pixel bilinear resize in the device graph and scales target coordinates by
the same ratio. This keeps a sampled augmentation independent of the current
network size, unlike the former implementation that regenerated augmentation
directly at each selected size. Parameter, BatchNorm, momentum, EMA, and
counter buffers migrate between the size-specialized programs on the same PJRT
client without a host tensor round-trip. A real CUDA/COCO check trained ten
steps at 64×64, migrated the device input to 96×96, and completed the eleventh
update with finite loss and 21 foreground anchors. The training snapshot marker
is `rxla-yolox-training-v3`; older snapshots predate the exact official
post-step learning-rate ordering and are rejected instead of silently resuming
a different optimization trajectory. A separate real check verified the
official empty-match behavior:
after augmentation removed every box, SimOTA produced zero foreground and the
normalized objectness-only loss continued instead of treating the batch as an
error. Multi-scale is enabled by default for recipe compatibility; pass
`--random-resize false` only for controlled diagnostics. Each distinct static
size needs its own compiled program; two tiny size variants took 43.7 seconds
in this check.
Building the example with
`--features model,disk-cache` and passing a trusted `--compile-cache DIR`
persists PJRT-native executables per exact StableHLO/shape. In a fresh-process
two-size test, the first run compiled both programs in `43.762 s`; the second
reported two disk hits, zero backend misses, and restored them in `0.664 s`.
The two entries occupied 6.7 MiB. Deserialized CUDA executables still paid lazy
first-execution setup, but compile-plus-run startup fell from roughly 43.9 s to
1.1 s.

Run the comparable RXLA side with:

```sh
cargo run -p rxla-safetensors --release --features model \
  --example yolox_train_benchmark -- \
  /path/to/xla_cuda_plugin.so yolox_tiny.safetensors \
  --size 416 --batch-size 8 --warmup 10 --iterations 100 \
  --targets-per-image 12 --compute-dtype f32
```

The input boundary preserves decoded images as U8 through host batching and
PCIe transfer, then performs the exact U8-to-F32 conversion in the compiled
graph. Target classes likewise cross the boundary as compact I32 indices rather
than host one-hot vectors. For batch 8, 120 padded targets, and 80 COCO classes,
this reduces their per-step upload from 307,200 to 3,840 bytes. SimOTA gathers
the selected class cost directly and only materializes one-hot classes for the
final matched anchors; this also removes the class-selection `dot_general` that
caused unnecessary GEMM autotuning. Mosaic exposes its four independent JPEG
decode/resize jobs to the same Rayon pool as the batch, rather than serializing
them inside each output image. Before the native max-pool adjoint, this let a 100-step real-data run
sustain 295.6 images/s (2.707 seconds for 800 images), 37.9% above the U8-only
result. Steady device steps were then 22.6–24.8 ms, observed Mosaic preparation
was 22.6–28.3 ms, and consumer wait remained 0.0 ms.

The first Nsight profile appeared to expose a convolution problem: RXLA launched
about 1,154 kernels per steady step with 15.21 ms aggregate GPU kernel time,
versus about 2,132 kernels and 7.70 ms for the profiled PyTorch/Inductor step.
About 239 `conv2d_grouped_direct` launches consumed 61.2% of RXLA kernel time.
The name was misleading. YOLOX-tiny has `depthwise=false`; the launches came
from RXLA's old max-pool adjoint, which expanded each spatial window position
into a transpose-convolution scatter. The 5x5, 9x9, and 13x13 SPP windows alone
created 275 such operations per training graph.

Max-pool reverse mode now remains a first-class RXLA IR operation and lowers to
one StableHLO `select_and_scatter` per pool. It also follows PyTorch/JAX-style
single-winner tie semantics instead of dividing a cotangent among equal maxima.
On the same real-data BF16 run, steady training steps fell to 12.2–16.3 ms. A
100-step run processed 800 output images in 2.618 seconds (305.6 images/s end to
end); JPEG/Mosaic preparation and producer waits, rather than the device graph,
now set that throughput. An 11-step Nsight capture measured 4.07 ms aggregate
GPU kernel time
and about 402 kernel launches per step. Only one grouped-direct launch remained
per step, consuming less than 0.01 ms. The native lowering therefore removed
the actual bottleneck instead of tuning unrelated convolutions.

Host augmentation is now split into deterministic sequential sampling and an
ordered parallel transformation stage. Multiple batches may decode and augment
concurrently, while a bounded reorder buffer preserves the exact seeded sample
order. On the same 100-step workload, 16 augmentation workers with 16 prefetched
batches raised end-to-end throughput from 305.6 to 513.6 images/s. Final
measurements for 1, 2, 4, 8, 16, and 24 workers were respectively 339.7, about
417, 429.1, 480.4, 513.6, and 497.7 images/s. Those measurements predated the
parallel pixel transforms, which made high outer concurrency redundant; the
current single-worker/eight-prefetch defaults and their paired measurements are
documented above. The prepared sample IDs remained identical across every run.

BF16 XLA autotune logs also confirm that forward and weight-gradient
convolutions remain BF16, while input gradients are represented as
reversed/dilated forward convolutions and ultimately autotuned as
`__cudnn$convForward`; PyTorch uses dedicated dgrad kernels. Rewriting the
reversal to JAX's explicit `stablehlo.reverse` form was mathematically correct
but slowed steady training
to 32.6--37.1 ms, so it was rejected. Autotune inspection showed that 12 BF16
shapes then became `__cudnn$convBackwardInput`, but an Nsight capture exposed
the regression: nine profiled steps spent 38.85 ms in 954
`dgrad2d_grouped_direct` launches, plus about 33 ms in FFT and GEMV support
kernels. The retained reversed-window forward formulation is substantially
faster for this graph. Closing the gap therefore requires a better convolution
plan rather than merely making XLA classify the operation as dgrad; a
depthwise-only kernel would not help this model.

Two narrower canonicalization experiments were also rejected. Using the dgrad
form only for 13x13, 3x3, ungrouped input gradients measured 23.4--24.8 ms and
278.7 images/s, still below the retained end-to-end baseline. Explicitly
transposing checkpoint-native OIHW parameters to JAX-style HWIO before every
convolution measured 22.9--24.4 ms and 272.7 images/s. This indicates that XLA
already canonicalizes the OIHW dimension numbers adequately; neither a
model-shape heuristic nor extra layout operations justify permanent API or IR
complexity.

RXLA now independently preserves genuinely grouped input gradients as one
`feature_group_count` StableHLO convolution instead of expanding a Rust loop.
CPU scalar-gradient and higher-order tests verify that lowering, but it is an
IR/correctness improvement rather than the explanation for YOLOX-tiny's speed.

The earlier real-data experiment that appeared to make `DEFAULT` convolution
precision slower was confounded by the old max-pool adjoint and host pipeline.
Repeating it after those fixes showed the opposite: strict synthetic F32 steps
fell from `26.075 ms` at `HIGHEST` to `17.108 ms` at `DEFAULT`. Nsight explains
the difference: the strict graph spent about 19.4 ms of GPU kernel time per
profiled step in SIMT/FFT convolution algorithms, while PyTorch used TF32
tensor-core kernels. `DEFAULT` is now an explicit per-convolution policy and is
selected by the training example; the model API retains `HIGHEST` as its
accuracy-first default.

Two other XLA configuration alternatives remain rejected. Enabling
`--xla_gpu_experimental_enable_conv_fusion=true` slowed them to 36--37 ms
(196 images/s), and forcing NCHW with `--xla_gpu_force_conv_nchw=true` reached
28.4--29.2 ms (235 images/s). The plugin already defaults convolution autotuning
to level 4; the former cuDNN frontend switch is no longer present in this XLA
version. The retained NHWC path is therefore the best measured general XLA
lowering rather than an accidentally disabled autotuning setup.

Disabling CUDA command buffers remains workload-dependent. On the sparse-target
strict BF16 benchmark, 100 measured steps improved from `12.719 ms` (`629.0
images/s`) to `12.459 ms` (`642.1 images/s`). Nsight showed about 173
`cuGraphLaunch` calls per step, with graph launches and node-parameter updates
accounting for roughly 3.3 ms of host CUDA API time. The seeded real COCO run,
however, regressed from `502.1` to `489.9 images/s`. Command buffers therefore
remain the default. Both training and synthetic benchmark examples expose
`--direct-launches` for hardware- and workload-specific measurement; internally
it uses `Compiler::without_cuda_command_buffers()`.

For bounded Nsight captures, `--cuda-profile --cuda-runtime /path/to/libcudart.so`
starts collection after the first training step. Use it with
`nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`.

The exported checkpoint was then loaded by the 416×416 inference example on
both CPU and CUDA. CUDA inference measured 9.4 ms after compilation and produced
the same 15 decoded detections as CPU for the validation image used in the test.
