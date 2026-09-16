//! CIFAR-10 ResNet-20 training with a heterogeneous input pipeline.
//!
//! Data files are decoded and batched with Rayon. Augmentation executes as a
//! Tensor program on CPU PJRT; the resulting batch is DMA-mapped and uploaded
//! asynchronously to CUDA before the GPU consumes it.
//! Omit `--dataset` for deterministic synthetic images.

use clap::Parser;
use rayon::prelude::*;
use rxla_core::{
    Buffer, CacheLimits, Client, Compiler, Conv2dOptions, DType, PendingHostUpload, Runtime, Tensor,
};
use rxla_nn::{Cx, Model, ModelInput, ParamSchema, Result as NnResult};
use rxla_train::{DataRng, apply_model_sgd};
use std::path::{Path, PathBuf};
use std::time::Instant;

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

    #[arg(long, default_value_t = 128)]
    batch_size: i64,

    #[arg(long, default_value_t = 10)]
    log_every: usize,

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

    fn synthetic(batch_size: i64) -> Self {
        let (image, labels) = synthetic_batch(batch_size);
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

    fn batch(
        &self,
        step: usize,
        batch_size: usize,
        rng: DataRng,
    ) -> (Vec<u8>, Vec<i32>, Vec<usize>) {
        let position = step * batch_size;
        let epoch = position / self.images.len();
        let epoch_rng = rng.sample(epoch as u64, 0);
        let mut multiplier = (epoch_rng.word(1, 0) as usize | 1) % self.images.len();
        while gcd(multiplier, self.images.len()) != 1 {
            multiplier = (multiplier + 2) % self.images.len();
        }
        let shuffle_offset = epoch_rng.word(1, 1) as usize % self.images.len();
        let samples = (0..batch_size)
            .into_par_iter()
            .map(|batch_offset| {
                let within_epoch = (position + batch_offset) % self.images.len();
                let index = (within_epoch * multiplier + shuffle_offset) % self.images.len();
                (index, self.images[index].clone(), self.labels[index])
            })
            .collect::<Vec<_>>();
        let mut images = Vec::with_capacity(batch_size * 3072);
        let mut labels = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        for (index, image, label) in samples {
            images.extend(image);
            labels.push(label);
            indices.push(index);
        }
        (images, labels, indices)
    }
}

fn gcd(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        (lhs, rhs) = (rhs, lhs % rhs);
    }
    lhs
}

fn basic_block(
    cx: &mut Cx,
    input: &Tensor,
    stage_index: usize,
    block_index: usize,
    channels: i64,
    stride: i64,
) -> NnResult<Tensor> {
    let mut stage = cx.scope(&format!("stage{stage_index}"))?;
    let mut block = stage.scope(&format!("block{block_index}"))?;
    let convolution = Conv2dOptions {
        strides: [stride, stride],
        padding: [[1, 1], [1, 1]],
        ..Default::default()
    };
    let hidden = block
        .layer("conv1")?
        .conv2d(channels, [3, 3])
        .options(convolution)
        .bias(false)
        .apply(input)?;
    let hidden = block.layer("bn1")?.batch_norm().apply(&hidden)?.relu()?;
    let hidden = block
        .layer("conv2")?
        .conv2d(channels, [3, 3])
        .options(Conv2dOptions {
            padding: [[1, 1], [1, 1]],
            ..Default::default()
        })
        .bias(false)
        .apply(&hidden)?;
    let hidden = block.layer("bn2")?.batch_norm().apply(&hidden)?;
    let residual = if stride == 1 {
        input.clone()
    } else {
        let downsampled = input.slice(&[0, 0, 0, 0], input.shape(), &[1, stride, stride, 1])?;
        let added = channels - input.shape()[3];
        downsampled.pad(
            &[[0, 0], [0, 0], [0, 0], [added / 2, added - added / 2]],
            0.0,
        )?
    };
    Ok(hidden.add(&residual)?.relu()?)
}

fn classifier(cx: &mut Cx, images: Tensor, labels: Tensor) -> NnResult<(Tensor, Tensor)> {
    let mut hidden = cx
        .layer("stem_conv")?
        .conv2d(16, [3, 3])
        .options(Conv2dOptions {
            padding: [[1, 1], [1, 1]],
            ..Default::default()
        })
        .bias(false)
        .apply(&images)?;
    hidden = cx.layer("stem_bn")?.batch_norm().apply(&hidden)?.relu()?;
    for stage in 0..3 {
        let channels = 16 << stage;
        for block in 0..3 {
            let stride = if stage != 0 && block == 0 { 2 } else { 1 };
            hidden = basic_block(cx, &hidden, stage, block, channels, stride)?;
        }
    }
    let features = hidden.mean(&[1, 2], false)?;
    let logits = cx.layer("head")?.linear(CLASSES).apply(&features)?;
    let loss = logits
        .cross_entropy_with_indices(&labels, 1)?
        .mean(&[0], false)?;
    Ok((loss, logits))
}

fn classifier_inputs(batch_size: i64) -> (ModelInput, ModelInput) {
    (
        ModelInput::new([batch_size, 32, 32, 3]),
        ModelInput::new([batch_size]).with_dtype(DType::I32),
    )
}

fn initialized_parameters(
    client: &Client,
    schema: &ParamSchema,
) -> Result<Vec<(String, Buffer)>, Box<dyn std::error::Error>> {
    let mut result = Vec::with_capacity(schema.parameters().len());
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    for spec in schema.parameters() {
        let count = spec.shape().iter().product::<i64>() as usize;
        let fan_in = spec.shape().iter().skip(1).product::<i64>().max(1) as f32;
        let batch_norm_weight = spec.path() == "stem_bn.weight"
            || spec.path().contains(".bn1.weight")
            || spec.path().contains(".bn2.weight");
        let scale = if batch_norm_weight {
            1.0
        } else if spec.path().ends_with("weight") {
            (2.0 / fan_in).sqrt()
        } else {
            0.0
        };
        let values = (0..count)
            .map(|_| {
                if batch_norm_weight {
                    return 1.0;
                }
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 40) as f32) / ((1_u32 << 24) as f32);
                (unit * 2.0 - 1.0) * scale
            })
            .collect::<Vec<_>>();
        result.push((
            spec.path().to_owned(),
            client.buffer(spec.shape(), &values)?,
        ));
    }
    Ok(result)
}

fn synthetic_batch(batch_size: i64) -> (Vec<u8>, Vec<i32>) {
    let mut images = vec![0; (batch_size * 32 * 32 * 3) as usize];
    let labels = (0..batch_size)
        .map(|sample| sample.rem_euclid(CLASSES) as i32)
        .collect::<Vec<_>>();
    for sample in 0..batch_size as usize {
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
    batch_size: i64,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let image = Tensor::from_slice([batch_size, 32, 32, 3], DType::U8, images)?;
    let flip = Tensor::from_slice([batch_size, 1, 1, 1], DType::F32, flips)?;
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
    batch_size: i64,
) -> Result<(PendingHostUpload<f32>, PendingHostUpload<i32>), Box<dyn std::error::Error>> {
    let (images, labels, sample_ids) = dataset.batch(step, batch_size as usize, rng);
    let epoch = (step * batch_size as usize / dataset.images.len()) as u64;
    let flips = sample_ids
        .into_iter()
        .map(|sample| rng.sample(epoch, sample as u64).bernoulli(0, 0, 0.5) as u8 as f32)
        .collect::<Vec<_>>();
    let images = augment(cpu, &images, &flips, batch_size)?;
    Ok((
        gpu.upload_pinned(&[batch_size, 32, 32, 3], images)?,
        gpu.upload_pinned(&[batch_size], labels)?,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.batch_size <= 0 || args.log_every == 0 {
        return Err("batch size and log interval must be positive".into());
    }
    let cpu = unsafe { Client::load(&args.cpu_plugin) }?;
    let gpu = unsafe { Client::load(&args.gpu_plugin) }?;
    let mut augmentation_runtime = Runtime::new(cpu)?;
    let batch_size = args.batch_size;
    let (schema, trainable, mut model) = Model::new(classifier)
        .inputs(classifier_inputs(batch_size))
        .trace_resident(ParamSchema::select_all)?;
    let loss = model.outputs()[0].clone();
    apply_model_sgd(&mut model, &trainable, &loss, args.learning_rate)?;
    let mut compiler = Compiler::new(gpu.clone(), CacheLimits::default());
    let compiled = model.compile_stateful(&mut compiler)?;

    let dataset = match args.dataset {
        Some(path) => Dataset::load_cifar10(&path)?,
        None => Dataset::synthetic(batch_size),
    };
    let data_rng = DataRng::new(args.seed);
    let mut session_builder = compiled
        .session()
        .parameters(initialized_parameters(&gpu, &schema)?)?;
    for state in schema
        .states()
        .iter()
        .filter(|state| state.path().ends_with("running_variance"))
    {
        let count = state.shape().iter().product::<i64>() as usize;
        session_builder = session_builder.state(
            state.path(),
            gpu.buffer(state.shape(), &vec![1.0_f32; count])?,
        )?;
    }
    let mut session = session_builder.build()?;
    let mut prepared = (args.steps != 0)
        .then(|| {
            prepare_batch(
                &dataset,
                0,
                &mut augmentation_runtime,
                &gpu,
                data_rng,
                batch_size,
            )
        })
        .transpose()?;
    let started = Instant::now();
    let mut initial_loss = None;
    let mut final_loss = None;
    let mut total_correct = 0;

    for step_index in 0..args.steps {
        let (images, labels) = prepared.take().expect("prepared above");
        prepared = (step_index + 1 < args.steps)
            .then(|| {
                prepare_batch(
                    &dataset,
                    step_index + 1,
                    &mut augmentation_runtime,
                    &gpu,
                    data_rng,
                    batch_size,
                )
            })
            .transpose()?;
        let (loss, logits): (Buffer, Buffer) = session.run((images.buffer(), labels.buffer()))?;
        let logits = logits.to_vec::<f32>()?;
        let targets = labels.buffer().to_vec::<i32>()?;
        let correct = logits
            .as_chunks::<10>()
            .0
            .iter()
            .zip(targets)
            .filter(|(row, target)| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                    .is_some_and(|(prediction, _)| prediction == *target as usize)
            })
            .count();
        drop((images, labels));
        let loss = loss.to_vec::<f32>()?[0];
        initial_loss.get_or_insert(loss);
        final_loss = Some(loss);
        total_correct += correct;
        if step_index % args.log_every == 0 || step_index + 1 == args.steps {
            println!(
                "step {step_index:>4}: loss {loss:.6}, accuracy {:.1}%",
                correct as f32 * 100.0 / batch_size as f32
            );
        }
    }
    if let (Some(initial), Some(final_)) = (initial_loss, final_loss) {
        let elapsed = started.elapsed().as_secs_f64();
        let samples = args.steps as f64 * batch_size as f64;
        println!(
            "summary: loss {initial:.6} -> {final_:.6}, mean accuracy {:.1}%, {:.1} images/s",
            total_correct as f64 * 100.0 / samples,
            samples / elapsed,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resnet20_has_expected_depth_and_state_schema() {
        let (schema, model) = Model::new(classifier)
            .inputs(classifier_inputs(4))
            .trace()
            .unwrap();
        let convolution_weights = schema
            .parameters()
            .iter()
            .filter(|parameter| {
                parameter.path().ends_with("conv.weight")
                    || parameter.path().contains(".conv1.weight")
                    || parameter.path().contains(".conv2.weight")
            })
            .count();
        assert_eq!(convolution_weights, 19);
        assert_eq!(schema.states().len(), 38);
        assert_eq!(model.outputs()[0].shape(), []);
        assert_eq!(model.outputs()[1].shape(), [4, CLASSES]);
    }

    #[test]
    fn deterministic_shuffle_is_a_permutation() {
        let dataset = Dataset::synthetic(128);
        let (_, _, indices) = dataset.batch(0, 128, DataRng::new(42));
        let mut sorted = indices;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..128).collect::<Vec<_>>());
    }
}
