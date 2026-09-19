# Structured logging

RXLA uses Protobuf messages inside MCAP rather than TensorBoard event files or
an application-specific record stream. The format is independently useful to
Rust and Python training programs and does not depend on tensors, PJRT, or the
RXLA runtime.

## Layers

- Protobuf defines stable event schemas and cross-language field evolution.
- MCAP supplies topics, timestamps, sequence numbers, chunks, compression,
  checksums, indexes, metadata, and binary attachments.
- `rxla-log` wraps both with typed Rust and Python APIs.

Core topics include run lifecycle, training steps, evaluation results, system
metrics, profile spans, checkpoints, and diagnostics. Extensible scalar values
use the typed `Metric` message. Images, StableHLO modules, profiler traces, and
other large values are MIME-typed MCAP attachments rather than metric fields.

## Rust

```rust,ignore
use rxla_log::{RunLog, proto::{Metric, TrainStep}};

let mut log = RunLog::create("train.mcap")?;
log.write(
    &TrainStep::new(7, 1.25)
        .learning_rate(0.01)
        .metric(Metric::scalar("box_loss", 0.25)),
    123,
)?;
log.attach("sample.jpg", "image/jpeg", image_bytes, 124)?;
log.finish()?;
```

`RunLog::flush` closes the current MCAP chunk and makes it durable without
finishing the run. `read_flushed_events` reads those complete, CRC-validated
chunks while the writer is still active (or after an interrupted run):

```rust,ignore
let bytes = std::fs::read("train.mcap")?;
let steps = rxla_log::read_flushed_events::<TrainStep>(&bytes)?;
```

The latest YOLOX training event can also be inspected directly:

```console
cargo run -p rxla-log --example latest_train_step -- path/to/training.mcap
```

## Python

```python
from rxla_log import Metric, RunLog, TrainStep

with RunLog("train.mcap") as log:
    log.write(
        TrainStep(
            step=7,
            loss=1.25,
            learning_rate=0.01,
            metrics=[Metric(name="box_loss", scalar=0.25)],
        ),
        123,
    )
```

The canonical schema is `crates/rxla-log/proto/rxla/log/v1/events.proto`.
`cargo run -p xtask -- generate` regenerates checked-in Prost, Python, and
descriptor-set artifacts with a pinned vendored protoc. Normal Cargo builds and
wheel builds therefore do not require protoc or another language toolchain.

Rust and Python each read a checked-in MCAP fixture produced by the other
implementation. These tests cover both the Protobuf wire representation and
the MCAP schema/channel conventions.
