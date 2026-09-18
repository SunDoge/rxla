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
  --window-sizes 128 \
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
| Directory sequential bytes | 13,394 | 200.8 ms | 66,709 | 7,451 |
| Lance sequential bytes | 13,394 | 960.4 ms | 13,946 | 1,558 |
| Directory random bytes | 2,048 | 32.6 ms | 62,839 | 6,839 |
| Lance random bytes | 2,048 | 115.2 ms | 17,776 | 1,935 |
| Directory random read + JPEG decode | 2,048 | 1,816.0 ms | 1,128 | 122.7 |
| Lance random read + JPEG decode | 2,048 | 1,925.1 ms | 1,064 | 115.8 |
| Lance metadata filter | 8,336 matched | 4.80 ms | 1.74M rows/s | n/a |

The metadata query selects `train` rows whose width is at least 320 while
projecting only sample ID, class ID, width, and height.

## Interpretation

Reading already-cached individual files is substantially faster than
materializing the same encoded bytes through Lance Blob v2. The directory has
almost no work beyond Linux page-cache lookup and Python `read_bytes`; Lance
also resolves descriptors and reconstructs blob payloads. This benchmark does
not support claiming that Lance is a faster sequential container.

Once JPEG decoding is included, the gap falls to 109.1 ms over 2,048 images:
Lance is 6.0% slower, while retaining indexed random access, column projection,
predicate filtering, stable row IDs, and dataset versions. Image decode, not
storage materialization, is the dominant cost in this warm-cache workload.

## Locality-aware shuffle

The follow-up holds the 2,048 selected samples fixed and compares three ways of
consuming them. Its physical read window is 128 rows. The pool variant retains
up to four windows (512 encoded images), randomly evicts samples, and can emit
training batches of 32 without coupling batch size to storage request size.

| Plan over the same samples | Bytes only | Read + JPEG decode |
|---|---:|---:|
| Fully scattered row order | 151.9 ms | 2,118.3 ms |
| 128-row range, local shuffle | 103.5 ms | 2,015.1 ms |
| 128-row range, 512-sample shuffle pool | 103.5 ms read plan | 2,044.1 ms |

Contiguous 128-row requests reduce blob materialization time by 31.9% relative
to the same sample IDs in scattered order. JPEG decode dilutes that advantage,
but local-window consumption remains 4.9% faster end to end. The larger pool
costs 1.4% versus window-local shuffle while improving cross-window mixing; it
is still 3.5% faster than the fully scattered plan.

This supports separating the three controls in a future loader:

- `batch_size = 32` controls model input shape;
- `read_window = 128` controls Lance/I/O locality;
- `shuffle_buffer = 512` controls statistical mixing.

They should not be represented by one `batch_size` option.

The next meaningful experiment is an asynchronous training pipeline with
parallel decode, augmentation, batch collation, pinned memory, and GPU upload.
It should also compare cold NVMe and object-storage behavior. The current result
supports Lance as a canonical/queryable dataset, but not yet as a proven faster
execution format than sequential TFRecord or WebDataset shards.

Raw timings and environment versions are recorded in
[`benchmarks/imagenette-lance-results.json`](../benchmarks/imagenette-lance-results.json).
