//! A small CIFAR-shaped CNN training smoke test.
//!
//! Data files are decoded and batched with Rayon. Augmentation executes as a
//! Tensor program on CPU PJRT; the resulting batch is DMA-mapped and uploaded
//! asynchronously to CUDA while the previous GPU training step is in flight.
//! Omit `--dataset` for deterministic synthetic images.

use clap::Parser;
use rayon::prelude::*;
use rxla_core::{
    Buffer, CacheLimits, Client, Compiler, Conv2dOptions, DType, PendingHostUpload, Pool2dOptions,
    Runtime, Tensor,
};
use rxla_nn::{Cx, Model, ParamSchema, Result as NnResult};
use rxla_train::{DataRng, prepare_model_sgd};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const BATCH: i64 = 4;
const CLASSES: i64 = 10;

#[derive(Parser)]
struct Args {
    /// Trusted CPU PJRT plugin used by the augmentation program.
    #[arg(long, env = "PJRT_CPU_PLUGIN_PATH")]
    cpu_plugin: String,

    /// Trusted CUDA PJRT plugin used by the model and optimizer program.
    #[arg(long, env = "PJRT_CUDA_PLUGIN_PATH")]
    gpu_plugin: String,

    #[arg(long, default_value_t = 5)]
    steps: usize,

    #[arg(long, default_value_t = 0.03)]
    learning_rate: f32,

    /// Root seed for schedule-independent per-sample augmentation.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Directory containing CIFAR-10 `data_batch_1.bin` through `data_batch_5.bin`.
    /// A deterministic synthetic batch is used when omitted.
    #[arg(long)]
    dataset: Option<PathBuf>,
}

struct Dataset {
    images: Vec<Vec<u8>>,
    labels: Vec<i32>,
}

impl Dataset {
    fn load_cifar10(root: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let paths = (1..=5)
            .map(|index| root.join(format!("data_batch_{index}.bin")))
            .collect::<Vec<_>>();
        let files = paths
            .par_iter()
            .map(std::fs::read)
            .collect::<Result<Vec<_>, _>>()?;
        if files.iter().any(|file| file.len() % 3073 != 0) {
            return Err("invalid CIFAR-10 binary record length".into());
        }
        let records = files
            .par_iter()
            .flat_map_iter(|file| file.as_chunks::<3073>().0.iter())
            .map(|record| {
                let mut nhwc = vec![0_u8; 3072];
                for channel in 0..3 {
                    for pixel in 0..1024 {
                        nhwc[pixel * 3 + channel] = record[1 + channel * 1024 + pixel];
                    }
                }
                (nhwc, i32::from(record[0]))
            })
            .collect::<Vec<_>>();
        let (images, labels) = records.into_iter().unzip();
        Ok(Self { images, labels })
    }

    fn synthetic() -> Self {
        let (image, labels) = synthetic_batch();
        Self {
            images: image
                .as_chunks::<3072>()
                .0
                .iter()
                .map(|sample| sample.to_vec())
                .collect(),
            labels,
        }
    }

    fn batch(&self, step: usize) -> (Vec<u8>, Vec<i32>, Vec<usize>) {
        let samples = (0..BATCH as usize)
            .into_par_iter()
            .map(|offset| {
                let index = (step * BATCH as usize + offset) % self.images.len();
                (self.images[index].clone(), self.labels[index])
            })
            .collect::<Vec<_>>();
        let mut images = Vec::with_capacity(BATCH as usize * 3072);
        let mut labels = Vec::with_capacity(BATCH as usize);
        let mut indices = Vec::with_capacity(BATCH as usize);
        for (image, label) in samples {
            images.extend(image);
            labels.push(label);
            indices.push((step * BATCH as usize + indices.len()) % self.images.len());
        }
        (images, labels, indices)
    }
}

fn classifier(cx: &mut Cx) -> NnResult<Tensor> {
    let images = cx.input(&[BATCH, 32, 32, 3])?;
    let labels = cx.input_dtype(&[BATCH], DType::I32)?;
    let same = Conv2dOptions {
        padding: [[1, 1], [1, 1]],
        ..Default::default()
    };
    let pool = Pool2dOptions::default();

    let hidden = cx
        .named("stem")?
        .conv2d(8, [3, 3])
        .options(same)
        .apply(&images)?
        .relu()?
        .avg_pool2d(pool, false)?;
    let hidden = cx
        .named("block")?
        .conv2d(16, [3, 3])
        .options(same)
        .apply(&hidden)?
        .relu()?
        .avg_pool2d(pool, false)?;
    let features = hidden.mean(&[1, 2], false)?;
    let logits = cx.named("head")?.linear(CLASSES).apply(&features)?;
    logits
        .cross_entropy_with_indices(&labels, 1)?
        .mean(&[0], false)
        .map_err(Into::into)
}

fn initialized_parameters(
    client: &Client,
    schema: &ParamSchema,
) -> Result<BTreeMap<String, Buffer>, Box<dyn std::error::Error>> {
    let mut result = BTreeMap::new();
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    for spec in schema.parameters() {
        let count = spec.shape().iter().product::<i64>() as usize;
        let fan_in = spec.shape().iter().skip(1).product::<i64>().max(1) as f32;
        let scale = if spec.path().ends_with("weight") {
            (2.0 / fan_in).sqrt()
        } else {
            0.0
        };
        let values = (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 40) as f32) / ((1_u32 << 24) as f32);
                (unit * 2.0 - 1.0) * scale
            })
            .collect::<Vec<_>>();
        result.insert(
            spec.path().to_owned(),
            client.buffer(spec.shape(), &values)?,
        );
    }
    Ok(result)
}

fn synthetic_batch() -> (Vec<u8>, Vec<i32>) {
    let mut images = vec![0; (BATCH * 32 * 32 * 3) as usize];
    let labels = (0..BATCH).map(|sample| sample as i32).collect::<Vec<_>>();
    for sample in 0..BATCH as usize {
        for y in 0..32 {
            for x in 0..32 {
                for channel in 0..3 {
                    let index = ((sample * 32 + y) * 32 + x) * 3 + channel;
                    images[index] = (((x * (sample + 1) + y * (channel + 1)) % 31) * 8) as u8;
                }
            }
        }
    }
    (images, labels)
}

fn augment(
    runtime: &mut Runtime,
    images: &[u8],
    flips: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let image = Tensor::from_slice([BATCH, 32, 32, 3], DType::U8, images)?;
    let flip = Tensor::from_slice([BATCH, 1, 1, 1], DType::F32, flips)?;
    let image = image.cast(DType::F32)?.mul_scalar(1.0 / 255.0)?;
    let flipped = image.flip_left_right()?;
    let augmented = flip
        .broadcast_to(image.shape())?
        .select(&flipped, &image)?
        .normalize_nhwc(&[0.4914, 0.4822, 0.4465], &[0.2470, 0.2435, 0.2616])?;
    Ok(runtime.eval(&augmented)?.to_vec::<f32>()?)
}

fn prepare_batch(
    dataset: &Dataset,
    step: usize,
    cpu: &mut Runtime,
    gpu: &Client,
    rng: DataRng,
) -> Result<(PendingHostUpload<f32>, PendingHostUpload<i32>), Box<dyn std::error::Error>> {
    let (images, labels, sample_ids) = dataset.batch(step);
    let epoch = (step * BATCH as usize / dataset.images.len()) as u64;
    let flips = sample_ids
        .into_iter()
        .map(|sample| rng.sample(epoch, sample as u64).bernoulli(0, 0, 0.5) as u8 as f32)
        .collect::<Vec<_>>();
    let images = augment(cpu, &images, &flips)?;
    Ok((
        gpu.upload_pinned(&[BATCH, 32, 32, 3], images)?,
        gpu.upload_pinned(&[BATCH], labels)?,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cpu = unsafe { Client::load(&args.cpu_plugin) }?;
    let gpu = unsafe { Client::load(&args.gpu_plugin) }?;
    let mut augmentation_runtime = Runtime::new(cpu);
    let (schema, model) = Model::new(classifier).trace()?;
    let training = prepare_model_sgd(
        &model,
        &schema.select_all(),
        &model.outputs()[0],
        args.learning_rate,
    )?;
    let mut outputs = vec![model.outputs()[0].clone()];
    outputs.extend(training.outputs());
    let mut compiler = Compiler::new(gpu.clone(), CacheLimits::default());
    let executable = model.compile_tensors(&mut compiler, &outputs)?;

    let dataset = match args.dataset {
        Some(path) => Dataset::load_cifar10(&path)?,
        None => Dataset::synthetic(),
    };
    let data_rng = DataRng::new(args.seed);
    let mut parameters = initialized_parameters(&gpu, &schema)?;
    let mut prepared = (args.steps != 0)
        .then(|| prepare_batch(&dataset, 0, &mut augmentation_runtime, &gpu, data_rng))
        .transpose()?;

    for step_index in 0..args.steps {
        let (images, labels) = prepared.take().expect("prepared above");
        let pending = {
            let arguments = model.bind(
                &[images.buffer(), labels.buffer()],
                parameters
                    .iter()
                    .map(|(name, value)| (name.as_str(), value)),
            )?;
            executable.submit(arguments.as_slice())?
        };
        prepared = (step_index + 1 < args.steps)
            .then(|| {
                prepare_batch(
                    &dataset,
                    step_index + 1,
                    &mut augmentation_runtime,
                    &gpu,
                    data_rng,
                )
            })
            .transpose()?;
        let output = pending.wait()?;
        drop((images, labels));
        let loss = output[0].to_vec::<f32>()?[0];
        for (update, value) in training.updates().iter().zip(output.into_iter().skip(1)) {
            parameters.insert(update.path().to_owned(), value);
        }
        println!("step {step_index:>3}: loss {loss:.6}");
    }
    Ok(())
}
