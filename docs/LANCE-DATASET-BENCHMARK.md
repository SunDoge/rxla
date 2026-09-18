# Imagenette Lance dataset benchmark

This experiment converts the local ImageFolder dataset at
`/data/datasets/imagenette2` into a self-contained Lance 2.2 dataset with
stable row IDs and Blob v2 image storage. It measures storage overhead and
three data-loader primitives before adding batching, augmentation, pinned
memory, or device transfer.

Run it with the repository's uv environment:

```bash
uv run --no-sync python benchmarks/imagenette_lance.py \
  build \
  /data/datasets/imagenette2 \
  /data/datasets/imagenette2.lance \
  --force
```

Benchmark the existing dataset independently:

```bash
uv run --no-sync python benchmarks/imagenette_lance.py \
  benchmark \
  /data/datasets/imagenette2 \
  /data/datasets/imagenette2.lance \
  --random-samples 2048 \
  --repeats 3 \
  --json-output benchmarks/imagenette-lance-results.json
```

## Dataset

- 13,394 JPEG images across `train` and `val`.
- Source payload: 1,568,676,055 bytes.
- Lance dataset: 1,569,758,399 bytes.
- Storage overhead: 1,082,344 bytes, or 0.069%.
- Conversion: 7.84 seconds in the recorded run.

Each Lance row contains a stable path-derived sample ID, relative path, split,
class ID/name, image dimensions, and the encoded JPEG in a Blob v2 column. The
conversion validates every image header while recording its dimensions.

## Warm-cache results

All operations receive one untimed warmup. Reported values are medians of three
runs on the same local machine, so these are page-cache and Python API costs,
not cold-storage or object-store measurements.

| Operation | Samples | Median | Samples/s | Encoded MiB/s |
|---|---:|---:|---:|---:|
| Directory sequential bytes | 13,394 | 225.9 ms | 59,300 | 6,623 |
| Lance sequential bytes | 13,394 | 909.7 ms | 14,723 | 1,644 |
| Directory random bytes | 2,048 | 37.2 ms | 55,086 | 5,995 |
| Lance random bytes | 2,048 | 136.8 ms | 14,972 | 1,629 |
| Directory random read + JPEG decode | 2,048 | 1,825.9 ms | 1,122 | 122.1 |
| Lance random read + JPEG decode | 2,048 | 1,983.9 ms | 1,032 | 112.3 |
| Lance metadata filter | 8,336 matched | 5.82 ms | 1.43M rows/s | n/a |

The metadata query selects `train` rows whose width is at least 320 while
projecting only sample ID, class ID, width, and height.

## Interpretation

Reading already-cached individual files is substantially faster than
materializing the same encoded bytes through Lance Blob v2. The directory has
almost no work beyond Linux page-cache lookup and Python `read_bytes`; Lance
also resolves descriptors and reconstructs blob payloads. This benchmark does
not support claiming that Lance is a faster sequential container.

Once JPEG decoding is included, the gap falls to 158.0 ms over 2,048 images:
Lance is 8.7% slower, while retaining indexed random access, column projection,
predicate filtering, stable row IDs, and dataset versions. Image decode, not
storage materialization, is the dominant cost in this warm-cache workload.

The next meaningful experiment is an asynchronous training pipeline with
parallel decode, augmentation, batch collation, pinned memory, and GPU upload.
It should also compare cold NVMe and object-storage behavior. The current result
supports Lance as a canonical/queryable dataset, but not yet as a proven faster
execution format than sequential TFRecord or WebDataset shards.

Raw timings and environment versions are recorded in
[`benchmarks/imagenette-lance-results.json`](../benchmarks/imagenette-lance-results.json).
