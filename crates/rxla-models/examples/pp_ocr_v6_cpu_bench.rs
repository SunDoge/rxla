use clap::Parser;
use image::{Rgb, RgbImage};
use rxla_models::pp_ocr_v6::{CtcDecoder, DetectorImage, RecognitionPreprocessor};
use std::{hint::black_box, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 100)]
    runs: usize,
    #[arg(long, default_value_t = 32)]
    crops: usize,
}

fn summary(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    (mean, p95)
}

fn main() {
    let args = Args::parse();
    assert!(args.runs > 0 && args.crops > 0);
    let page = RgbImage::from_fn(1920, 1080, |x, y| {
        Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
    });
    let crops: Vec<_> = (0..args.crops)
        .map(|index| {
            let width = 48 + (index as u32 * 37) % 420;
            RgbImage::from_fn(width, 48 + index as u32 % 23, |x, y| {
                Rgb([(x % 255) as u8, (y % 255) as u8, ((x + y) % 255) as u8])
            })
        })
        .collect();
    let preprocessor = RecognitionPreprocessor::default();
    let classes = 18_000;
    let time = 40;
    let probabilities = vec![0.001; args.crops * time * classes];
    let decoder = CtcDecoder::default();

    let mut detector = Vec::with_capacity(args.runs);
    let mut recognition = Vec::with_capacity(args.runs);
    let mut ctc = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        black_box(DetectorImage::prepare(&page, 960).unwrap());
        detector.push(start.elapsed().as_secs_f64() * 1e3);
        let start = Instant::now();
        black_box(preprocessor.prepare(&crops).unwrap());
        recognition.push(start.elapsed().as_secs_f64() * 1e3);
        let start = Instant::now();
        black_box(
            decoder
                .decode(&probabilities, [args.crops, time, classes])
                .unwrap(),
        );
        ctc.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let (det_mean, det_p95) = summary(detector);
    let (rec_mean, rec_p95) = summary(recognition);
    let (ctc_mean, ctc_p95) = summary(ctc);
    println!(
        "PP-OCRv6 CPU host stages ({} runs, {} crops)",
        args.runs, args.crops
    );
    println!("detector resize+normalize: mean {det_mean:.3} ms, p95 {det_p95:.3} ms");
    println!("recognition batch prepare: mean {rec_mean:.3} ms, p95 {rec_p95:.3} ms");
    println!("greedy CTC decode:        mean {ctc_mean:.3} ms, p95 {ctc_p95:.3} ms");
}
