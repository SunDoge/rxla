use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use image::ImageReader;
use lance::Dataset;
use lance::dataset::ReadBlob;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Benchmark Lance Blob v2 reads and Rayon JPEG decoding")]
struct Args {
    #[arg(default_value = "/data/datasets/coco2017.lance")]
    dataset: PathBuf,
    #[arg(long, default_value_t = 4096)]
    samples: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
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
    encoded_mib_per_second: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    dataset: PathBuf,
    rows: usize,
    samples: usize,
    encoded_bytes: usize,
    rayon_threads: usize,
    measurements: Vec<Measurement>,
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    }
}

fn measurement(
    name: &'static str,
    timings: Vec<Duration>,
    samples: usize,
    encoded_bytes: usize,
) -> Measurement {
    let seconds_all: Vec<_> = timings.iter().map(Duration::as_secs_f64).collect();
    let seconds_median = median(&seconds_all);
    Measurement {
        name,
        seconds_median,
        seconds_all,
        samples_per_second: samples as f64 / seconds_median,
        encoded_mib_per_second: encoded_bytes as f64 / seconds_median / (1024.0 * 1024.0),
    }
}

fn decode(blob: &ReadBlob) -> Result<usize, image::ImageError> {
    let bytes = blob.data.as_deref().unwrap_or_default();
    let image = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?
        .into_rgb8();
    Ok(image.as_raw().len())
}

fn serial_decode(blobs: &[ReadBlob]) -> Result<usize, image::ImageError> {
    blobs.iter().map(decode).sum()
}

fn parallel_decode(blobs: &[ReadBlob]) -> Result<usize, image::ImageError> {
    blobs.par_iter().map(decode).sum()
}

async fn read(dataset: &Arc<Dataset>, indices: &[u64]) -> lance::Result<Vec<ReadBlob>> {
    dataset
        .read_blobs("image")?
        .with_row_indices(indices.to_vec())
        .preserve_order(false)
        .execute()
        .await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.samples == 0 || args.repeats == 0 {
        return Err("samples and repeats must be positive".into());
    }
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }

    let uri = args
        .dataset
        .to_str()
        .ok_or("dataset path is not valid UTF-8")?;
    let dataset = Arc::new(Dataset::open(uri).await?);
    let rows = dataset.count_rows(None).await?;
    let samples = args.samples.min(rows);
    let mut indices: Vec<_> = (0..rows as u64).collect();
    indices.shuffle(&mut SmallRng::seed_from_u64(args.seed));
    indices.truncate(samples);

    // Warm the Lance planner/page cache and every JPEG decoder path.
    let blobs = read(&dataset, &indices).await?;
    let encoded_bytes = blobs
        .iter()
        .filter_map(|blob| blob.data.as_ref())
        .map(|data| data.len())
        .sum();
    serial_decode(&blobs)?;
    parallel_decode(&blobs)?;

    let mut read_timings = Vec::with_capacity(args.repeats);
    let mut serial_timings = Vec::with_capacity(args.repeats);
    let mut parallel_timings = Vec::with_capacity(args.repeats);
    let mut end_to_end_timings = Vec::with_capacity(args.repeats);

    for _ in 0..args.repeats {
        let started = Instant::now();
        let current = read(&dataset, &indices).await?;
        read_timings.push(started.elapsed());

        let started = Instant::now();
        serial_decode(&current)?;
        serial_timings.push(started.elapsed());

        let started = Instant::now();
        parallel_decode(&current)?;
        parallel_timings.push(started.elapsed());

        let started = Instant::now();
        let current = read(&dataset, &indices).await?;
        parallel_decode(&current)?;
        end_to_end_timings.push(started.elapsed());
    }

    let report = Report {
        dataset: args.dataset,
        rows,
        samples,
        encoded_bytes,
        rayon_threads: rayon::current_num_threads(),
        measurements: vec![
            measurement("lance_read", read_timings, samples, encoded_bytes),
            measurement("serial_jpeg_decode", serial_timings, samples, encoded_bytes),
            measurement(
                "rayon_jpeg_decode",
                parallel_timings,
                samples,
                encoded_bytes,
            ),
            measurement(
                "lance_read_and_rayon_decode",
                end_to_end_timings,
                samples,
                encoded_bytes,
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
