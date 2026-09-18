# Fory Standard Row in TFRecord

This experiment evaluates TFRecord framing with Apache Fory Standard Row as a
lightweight Python-to-Rust training format. Python exports typed COCO detection
records with `pyfory`; Rust validates TFRecord CRC32C, accesses image and box
fields as borrowed Fory views, and decodes JPEG images concurrently.

```text
TFRecord       framing, corruption detection, sequential shards
Fory Row       typed fields, offsets, validation, borrowed binary views
```

## Reproduce

Export 4,096 deterministic COCO train samples:

```bash
uv run --no-sync python benchmarks/fory_tfrecord.py \
  /data/datasets/coco2017 \
  /data/datasets/coco2017-fory-4096.tfrecord \
  --samples 4096 \
  --force
```

Build and run the standalone Rust reader:

```bash
cd benchmarks/fory_tfrecord_rust
cargo build --release
./target/release/fory-tfrecord-bench \
  /data/datasets/coco2017-fory-4096.tfrecord \
  --repeats 3 \
  --json-output ../fory-tfrecord-results.json
```

## Format and export

The experiment uses this logical row:

```text
boxes: list<float>
height: int32
image: binary
labels: list<int32>
sample_id: utf8
width: int32
```

PyFory canonicalizes fields into lexical order. Rust Standard Row interprets
fields by declaration order, so the Rust struct must use the canonical order.
The exporter records both schema text and its SHA-256 digest; a production
manifest must reject a mismatched schema before reading records.

The Python exporter uses `tfrecord-slow` for standard TFRecord framing and
masked CRC32C. Its recorded output was:

| Metric | Result |
|---|---:|
| Samples | 4,096 |
| Boxes | 29,964 |
| Encoded JPEG bytes | 668,083,232 |
| Fory row bytes | 669,167,152 |
| TFRecord shard bytes | 669,232,688 |
| Export time | 2.369 s |
| Export throughput | 268.9 MiB/s |

Fory adds 1,083,920 bytes over the JPEG payload, or 0.162%. TFRecord adds a
fixed 16 bytes per record, totalling 65,536 bytes.

## Rust results

The machine exposed 24 Rayon threads. Every operation received an untimed
warmup; values are medians of three runs over the same shard.

| Operation | Median | Images/s | Payload MiB/s |
|---|---:|---:|---:|
| `BufReader` + TFRecord CRC32C | 155.0 ms | 26,422 | 4,117 |
| Parallel Fory borrowed field access | 1.14 ms | 3.60M | 560,886 |
| Rayon JPEG decode | 381.6 ms | 10,734 | 1,670 |
| Read fully, then decode | 710.0 ms | 5,769 | 897 |
| Bounded streaming read + parallel decode | 416.7 ms | 9,830 | 1,529 |

The Rust benchmark verifies that every `image()` slice points inside its
original TFRecord-owned `Bytes`; Python-to-Rust image access is genuinely
borrowed and does not allocate or copy the JPEG payload. Fory parsing is too
small to matter beside JPEG decode.

The bounded pipeline overlaps one sequential `BufReader` producer with 24
decode consumers. It reaches 91.6% of decode-only throughput. Its 9,830
images/s is about 11% above the previously recorded native Lance read plus
Rayon decode result of 8,823 images/s, although the deterministic sample sets
are not identical and the difference should not be treated as a format-only
speedup.

## Engineering cost

The standalone Rust program's first release build completed in 12.6 seconds in
the recorded environment. Its stripped executable is 1.4 MiB. By comparison,
the native Lance experiment pulled hundreds of packages, took minutes to build,
and produced an 806 MiB unstripped executable before its profile was changed.

Fory is therefore a much more practical runtime dependency, with caveats:

- Standard Row is intended for trusted analytical data, which fits immutable
  training snapshots but not arbitrary untrusted network input.
- Field order and exact numeric widths are part of the cross-language schema.
- Python currently emits a harmless Cython enum import warning on encoder
  creation.
- Python Row Format brings Arrow-related dependencies into the export
  environment, though not into the Rust training runtime.
- Fory versions should remain pinned until cross-version compatibility is
  covered by golden Python-to-Rust fixtures.

The result supports `TFRecord + Fory Standard Row` as the leading training
execution format. Data management and arbitrary querying can remain upstream;
the Rust runtime only needs sequential shard reading, borrowed row views,
parallel decode/augmentation, and bounded queues.

Raw Rust results are in
[`benchmarks/fory-tfrecord-results.json`](../benchmarks/fory-tfrecord-results.json).
