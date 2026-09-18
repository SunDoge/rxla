use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bytes::Bytes;
use clap::Parser;
use crossbeam_channel::bounded;
use fory::{ForyRow, from_row};
use image::ImageReader;
use rayon::prelude::*;
use serde::Serialize;

#[derive(ForyRow)]
struct DetectionRecord {
    // pyfory canonicalizes struct fields into this lexical order.
    boxes: Vec<f32>,
    height: i32,
    image: Vec<u8>,
    labels: Vec<i32>,
    sample_id: String,
    width: i32,
}

#[derive(Debug, Parser)]
#[command(about = "Benchmark TFRecord-framed Fory Standard Rows")]
struct Args {
    #[arg(default_value = "/data/datasets/coco2017-fory-4096.tfrecord")]
    input: PathBuf,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    json_output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Measurement {
    name: &'static str,
    seconds_median: f64,
    seconds_all: Vec<f64>,
    samples_per_second: f64,
    mib_per_second: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    input: PathBuf,
    samples: usize,
    record_bytes: usize,
    image_bytes: usize,
    boxes: usize,
    rayon_threads: usize,
    borrowed_images: bool,
    measurements: Vec<Measurement>,
}

fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c::crc32c(data);
    crc.rotate_right(15).wrapping_add(0xa282_ead8)
}

fn visit_tfrecord(
    path: &PathBuf,
    mut visit: impl FnMut(Bytes) -> Result<(), String>,
) -> Result<(), String> {
    let mut reader = BufReader::with_capacity(
        8 << 20,
        File::open(path).map_err(|error| error.to_string())?,
    );
    loop {
        let mut length_bytes = [0; 8];
        match reader
            .read(&mut length_bytes[..1])
            .map_err(|error| error.to_string())?
        {
            0 => break,
            1 => reader
                .read_exact(&mut length_bytes[1..])
                .map_err(|error| error.to_string())?,
            _ => unreachable!(),
        }
        let mut length_crc = [0; 4];
        reader
            .read_exact(&mut length_crc)
            .map_err(|error| error.to_string())?;
        if u32::from_le_bytes(length_crc) != masked_crc32c(&length_bytes) {
            return Err("invalid TFRecord length CRC32C".into());
        }
        let length =
            usize::try_from(u64::from_le_bytes(length_bytes)).map_err(|error| error.to_string())?;
        if length > 64 << 20 {
            return Err(format!("TFRecord payload exceeds 64 MiB: {length} bytes"));
        }
        let mut record = vec![0; length];
        reader
            .read_exact(&mut record)
            .map_err(|error| error.to_string())?;
        let mut data_crc = [0; 4];
        reader
            .read_exact(&mut data_crc)
            .map_err(|error| error.to_string())?;
        if u32::from_le_bytes(data_crc) != masked_crc32c(&record) {
            return Err("invalid TFRecord data CRC32C".into());
        }
        visit(Bytes::from(record))?;
    }
    Ok(())
}

fn read_tfrecord(path: &PathBuf) -> Result<Vec<Bytes>, Box<dyn Error>> {
    let mut records = Vec::new();
    visit_tfrecord(path, |record| {
        records.push(record);
        Ok(())
    })
    .map_err(|error| -> Box<dyn Error> { error.into() })?;
    Ok(records)
}

fn inspect(records: &[Bytes]) -> Result<(usize, usize, bool), fory::Error> {
    let mut image_bytes = 0;
    let mut boxes = 0;
    let mut borrowed = true;
    for record in records {
        let row = from_row::<DetectionRecord>(record)?;
        let image = row.image()?;
        let start = record.as_ptr() as usize;
        let end = start + record.len();
        let image_start = image.as_ptr() as usize;
        borrowed &= image_start >= start && image_start + image.len() <= end;
        image_bytes += image.len();
        boxes += row.boxes()?.len() / 4;
        let _ = (
            row.width()?,
            row.height()?,
            row.labels()?.len(),
            row.sample_id()?,
        );
    }
    Ok((image_bytes, boxes, borrowed))
}

fn parallel_inspect(records: &[Bytes]) -> Result<usize, String> {
    records
        .par_iter()
        .map(|record| {
            let row = from_row::<DetectionRecord>(record).map_err(|error| error.to_string())?;
            Ok(row.image().map_err(|error| error.to_string())?.len()
                + row.boxes().map_err(|error| error.to_string())?.len() * 4)
        })
        .sum()
}

fn decode(records: &[Bytes]) -> Result<usize, String> {
    records
        .par_iter()
        .map(|record| {
            let row = from_row::<DetectionRecord>(record).map_err(|error| error.to_string())?;
            let image = ImageReader::new(std::io::Cursor::new(
                row.image().map_err(|error| error.to_string())?,
            ))
            .with_guessed_format()
            .map_err(|error| error.to_string())?
            .decode()
            .map_err(|error| error.to_string())?
            .into_rgb8();
            Ok(image.as_raw().len())
        })
        .sum()
}

fn streaming_read_and_decode(
    path: &PathBuf,
    workers: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    let (sender, receiver) = bounded::<Bytes>(workers * 2);
    let (samples, decoded_bytes) = std::thread::scope(|scope| {
        let producer = scope.spawn(move || -> Result<usize, String> {
            let mut samples = 0;
            visit_tfrecord(path, |record| {
                sender.send(record).map_err(|error| error.to_string())?;
                samples += 1;
                Ok(())
            })?;
            Ok(samples)
        });

        let consumers: Vec<_> = (0..workers)
            .map(|_| {
                let receiver = receiver.clone();
                scope.spawn(move || -> Result<usize, String> {
                    let mut decoded_bytes = 0;
                    for record in receiver {
                        let row = from_row::<DetectionRecord>(&record)
                            .map_err(|error| error.to_string())?;
                        let image = ImageReader::new(std::io::Cursor::new(
                            row.image().map_err(|error| error.to_string())?,
                        ))
                        .with_guessed_format()
                        .map_err(|error| error.to_string())?
                        .decode()
                        .map_err(|error| error.to_string())?
                        .into_rgb8();
                        decoded_bytes += image.as_raw().len();
                    }
                    Ok(decoded_bytes)
                })
            })
            .collect();
        let samples = producer
            .join()
            .map_err(|_| "TFRecord producer panicked")??;
        let decoded_bytes = consumers
            .into_iter()
            .map(|consumer| {
                consumer
                    .join()
                    .map_err(|_| "JPEG consumer panicked")?
                    .map_err(|error| -> Box<dyn Error> { error.into() })
            })
            .sum::<Result<usize, Box<dyn Error>>>()?;
        Ok::<_, Box<dyn Error>>((samples, decoded_bytes))
    })?;
    Ok((samples, decoded_bytes))
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    }
}

fn measurement(
    name: &'static str,
    timings: Vec<Duration>,
    samples: usize,
    bytes: usize,
) -> Measurement {
    let seconds_all: Vec<_> = timings.iter().map(Duration::as_secs_f64).collect();
    let seconds_median = median(&seconds_all);
    Measurement {
        name,
        seconds_median,
        seconds_all,
        samples_per_second: samples as f64 / seconds_median,
        mib_per_second: bytes as f64 / seconds_median / (1024.0 * 1024.0),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.repeats == 0 {
        return Err("repeats must be positive".into());
    }
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }

    let records = read_tfrecord(&args.input)?;
    let record_bytes = records.iter().map(Bytes::len).sum();
    let (image_bytes, boxes, borrowed_images) = inspect(&records)?;
    parallel_inspect(&records)?;
    decode(&records)?;
    streaming_read_and_decode(&args.input, rayon::current_num_threads())?;

    let mut read_timings = Vec::with_capacity(args.repeats);
    let mut inspect_timings = Vec::with_capacity(args.repeats);
    let mut decode_timings = Vec::with_capacity(args.repeats);
    let mut end_to_end_timings = Vec::with_capacity(args.repeats);
    let mut streaming_timings = Vec::with_capacity(args.repeats);
    for _ in 0..args.repeats {
        let started = Instant::now();
        let current = read_tfrecord(&args.input)?;
        read_timings.push(started.elapsed());

        let started = Instant::now();
        parallel_inspect(&current)?;
        inspect_timings.push(started.elapsed());

        let started = Instant::now();
        decode(&current)?;
        decode_timings.push(started.elapsed());

        let started = Instant::now();
        let current = read_tfrecord(&args.input)?;
        decode(&current)?;
        end_to_end_timings.push(started.elapsed());

        let started = Instant::now();
        streaming_read_and_decode(&args.input, rayon::current_num_threads())?;
        streaming_timings.push(started.elapsed());
    }

    let samples = records.len();
    let report = Report {
        input: args.input,
        samples,
        record_bytes,
        image_bytes,
        boxes,
        rayon_threads: rayon::current_num_threads(),
        borrowed_images,
        measurements: vec![
            measurement("tfrecord_read_crc", read_timings, samples, record_bytes),
            measurement(
                "fory_borrowed_access",
                inspect_timings,
                samples,
                record_bytes,
            ),
            measurement("rayon_jpeg_decode", decode_timings, samples, image_bytes),
            measurement(
                "tfrecord_read_and_rayon_decode",
                end_to_end_timings,
                samples,
                image_bytes,
            ),
            measurement(
                "streaming_tfrecord_and_parallel_decode",
                streaming_timings,
                samples,
                image_bytes,
            ),
        ],
    };
    let rendered = serde_json::to_string_pretty(&report)?;
    println!("{rendered}");
    if let Some(output) = args.json_output {
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(output, format!("{rendered}\n"))?;
    }
    Ok(())
}
