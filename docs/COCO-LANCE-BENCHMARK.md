# COCO 2017 Lance dataset benchmark

This experiment converts COCO 2017 detection data into a self-contained Lance
2.2 dataset. It tests the layout intended for future rxla training loaders:
encoded images and structured annotations live together, while `split` remains
ordinary metadata instead of defining separate physical datasets.

Build the combined dataset with the repository's uv environment:

```bash
uv run --no-sync python benchmarks/coco_lance.py \
  build \
  /data/datasets/coco2017 \
  /data/datasets/coco2017.lance \
  --splits train2017 val2017 \
  --force
```

Run the recorded benchmark independently:

```bash
uv run --no-sync python benchmarks/coco_lance.py \
  benchmark \
  /data/datasets/coco2017.lance \
  --samples 4096 \
  --read-window 128 \
  --shuffle-buffer 512 \
  --repeats 3 \
  --json-output benchmarks/coco-lance-results.json
```

## Dataset layout

The combined dataset contains 123,287 images and 896,782 annotations:

| Split | Images | Boxes |
|---|---:|---:|
| `train2017` | 118,287 | 860,001 |
| `val2017` | 5,000 | 36,781 |

The source ZIP files were integrity-checked before extraction. Conversion took
97.65 seconds and produced 8 Lance fragments totalling 20,336,520,696 bytes.
The original train and validation image ZIP payloads total 20,152,447,128
bytes; this comparison excludes the compressed annotations ZIP and extracted
filesystem metadata.

Each image row contains:

- a stable ID derived from the source namespace, split, and COCO image ID;
- split, filename, dimensions, box count, and crowd count;
- distinct category IDs for cheap category filtering;
- structured instances with both `xywh` and `xyxy` boxes, area, crowd flag,
  annotation ID, and losslessly retained polygon/RLE segmentation JSON;
- the encoded image in a Lance Blob v2 column.

This is deliberately one dataset with a `split` column. A split is dataset
metadata, not a storage boundary. That makes adding `test`, `unlabelled`,
`hard`, or cross-validation partitions possible without copying image data or
changing the loader abstraction. Temporary one-split datasets can still be
useful when isolating storage experiments.

## Warm-cache results

The machine used Python 3.14.6, Lance 11.0.0, and PyArrow 25.0.1. Every
operation receives an untimed warmup; values are medians of three runs. Image
measurements include JPEG decoding to RGB, so encoded throughput is shown only
as a useful reference.

| Operation | Samples/matches | Median | Samples/s | Encoded MiB/s |
|---|---:|---:|---:|---:|
| One scattered Lance request + decode | 4,096 | 5.314 s | 770.8 | 119.9 |
| 128-row requests + 512-image shuffle pool + decode | 4,096 | 5.771 s | 709.7 | 110.4 |
| `box_count >= 10 AND crowd_count = 0` | 22,956 | 11.21 ms | 2.05M | n/a |
| Category ID 1 appears in image | 66,808 | 14.72 ms | 4.54M | n/a |

The selected image sample is identical for both decode plans and contains
668,000,211 encoded bytes. The manual window/pool plan is 8.6% slower here.
This reverses the small Imagenette experiment because a single large
`read_blobs` request gives Lance enough information to plan/coalesce scattered
access internally, whereas the manual plan submits many Python calls.

The result does **not** mean locality is irrelevant. HDDs, remote object stores,
cold caches, and bounded-memory loaders have different constraints. It means
the loader should expose locality as policy rather than imposing 128-row reads:

- prefer one planned Lance request when the sample set and memory budget allow;
- use fewer, larger physical windows for high-latency storage;
- keep model batch size independent from I/O request size and shuffle depth;
- overlap reading, decode, augmentation, collation, and device transfer before
  judging end-to-end training throughput.

## Self-collected datasets

The same physical dataset can grow append-only while split membership remains
a column. A practical schema should keep a stable `sample_id` and provenance
fields such as capture/session/group IDs. Train/validation assignment should be
computed at the group level to prevent adjacent video frames, bursts, or the
same object from leaking across splits. Hard-sample mining can then be another
metadata column or side table instead of rebuilding sequential shards.

For more complex annotations, the current nested instance column is convenient
for per-image training reads. A separate annotation table keyed by `sample_id`
can be added later for global instance-level analytics without changing image
storage.

Raw timings and environment versions are recorded in
[`benchmarks/coco-lance-results.json`](../benchmarks/coco-lance-results.json).

## Rust and Rayon follow-up

The Rust follow-up uses Lance's native `ReadBlobsBuilder`, the `image` crate's
JPEG decoder, and Rayon. It does not cross the Python boundary. Build it with
the protoc already managed by mise:

```bash
cd benchmarks/coco_lance_rust
PROTOC=/home/me/.local/share/mise/installs/protoc/35.1/bin/protoc \
  cargo build --release
./target/release/coco-lance-bench \
  /data/datasets/coco2017.lance \
  --samples 4096 \
  --repeats 3 \
  --json-output ../coco-lance-rust-results.json
```

On 4,096 random images totalling 664,971,737 encoded bytes:

| Operation | Threads | Median | Images/s |
|---|---:|---:|---:|
| Native Lance Blob v2 read | internal | 64.1 ms | 63,948 |
| Rust JPEG decode | 1 | 5.427 s | 754.7 |
| Rayon JPEG decode | 24 | 382.6 ms | 10,704 |
| Native Lance read + Rayon decode | 24 | 464.3 ms | 8,823 |

The serial Rust result is close to the Python/Pillow end-to-end result, while
Rayon makes the complete path about 11.4 times faster than the recorded Python
path. This confirms that JPEG decoding and lack of concurrency—not Lance blob
I/O—caused the roughly 770 images/s result.

Thread scaling for the complete read-and-decode operation was:

| Rayon threads | Median | Images/s |
|---:|---:|---:|
| 4 | 1.446 s | 2,833 |
| 8 | 781.0 ms | 5,245 |
| 16 | 546.5 ms | 7,495 |
| 24 | 464.3 ms | 8,823 |

Scaling remains useful through 24 threads, although it is sublinear after 8
threads. A real loader should use a bounded Rayon pool so decoding does not
compete with model-side CPU work.

There is an important integration cost: Lance 11 does not expose a lightweight
read-only dataset crate. `lance-file` and `lance-io` are split out, but the
high-level `Dataset::open` and planned `read_blobs` API is in the umbrella
`lance` crate, which unconditionally includes DataFusion, indexing, and text
search dependencies even with default features disabled. The benchmark is
therefore a standalone crate outside the rxla workspace. Adding Lance to rxla's
public runtime dependency graph would currently impose an unreasonable clean
build cost.

Raw Rust timings are recorded in
[`benchmarks/coco-lance-rust-results.json`](../benchmarks/coco-lance-rust-results.json).
