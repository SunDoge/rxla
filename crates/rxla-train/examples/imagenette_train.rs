//! ResNet-18 training and validation on an ImageFolder-style Imagenette tree.

use clap::Parser;
use image::{GenericImageView, imageops::FilterType};
use rayon::prelude::*;
use rxla_core::{
    Buffer, CacheLimits, Client, Compiler, Conv2dOptions, DType, PendingHostUpload, Pool2dOptions,
    Tensor,
};
use rxla_nn::{AppliedModel, Cx, Model, ParamSchema, Result as NnResult};
use rxla_train::{BoundedPipeline, DataRng, PipelineResult, prepare_model_sgd};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const CLASSES: i64 = 10;
const IMAGE: i64 = 160;
const RESIZE: u32 = 176;

#[derive(Parser)]
struct Args {
    #[arg(long, env = "PJRT_CUDA_PLUGIN_PATH")]
    gpu_plugin: String,
    #[arg(long, default_value = "/data/users/me/datasets/imagenette2")]
    dataset: PathBuf,
    #[arg(long, default_value_t = 16)]
    batch_size: i64,
    #[arg(long, default_value_t = 600)]
    steps: usize,
    #[arg(long, default_value_t = 0.03)]
    learning_rate: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 25)]
    log_every: usize,
    /// Print a host-side breakdown after excluding the first ten steps.
    #[arg(long)]
    profile: bool,
}

#[derive(Default)]
struct PhaseTimes {
    samples: usize,
    input_wait: Duration,
    upload: Duration,
    bind: Duration,
    execute: Duration,
    metrics: Duration,
}

impl PhaseTimes {
    fn print(&self, batch_size: i64) {
        if self.samples == 0 {
            return;
        }
        let milliseconds = |duration: Duration| duration.as_secs_f64() * 1_000.0;
        let total = self.input_wait + self.upload + self.bind + self.execute + self.metrics;
        println!(
            "profile: {samples} samples, mean step {total:.3} ms: input-wait {input_wait:.3} ms ({input_wait_share:.1}%), upload {upload:.3} ms ({upload_share:.1}%), bind {bind:.3} ms ({bind_share:.1}%), execute {execute:.3} ms ({execute_share:.1}%), metrics {metrics:.3} ms ({metrics_share:.1}%); measured throughput {throughput:.1} images/s",
            samples = self.samples,
            total = milliseconds(total) / self.samples as f64,
            input_wait = milliseconds(self.input_wait) / self.samples as f64,
            input_wait_share = self.input_wait.as_secs_f64() * 100.0 / total.as_secs_f64(),
            upload = milliseconds(self.upload) / self.samples as f64,
            upload_share = self.upload.as_secs_f64() * 100.0 / total.as_secs_f64(),
            bind = milliseconds(self.bind) / self.samples as f64,
            bind_share = self.bind.as_secs_f64() * 100.0 / total.as_secs_f64(),
            execute = milliseconds(self.execute) / self.samples as f64,
            execute_share = self.execute.as_secs_f64() * 100.0 / total.as_secs_f64(),
            metrics = milliseconds(self.metrics) / self.samples as f64,
            metrics_share = self.metrics.as_secs_f64() * 100.0 / total.as_secs_f64(),
            throughput = self.samples as f64 * batch_size as f64 / total.as_secs_f64(),
        );
    }
}

#[derive(Clone)]
struct Sample {
    path: PathBuf,
    label: i32,
}

struct ImageFolder {
    samples: Vec<Sample>,
}

type HostBatch = (Vec<u8>, Vec<i32>);
type UploadedBatch = (PendingHostUpload<u8>, PendingHostUpload<i32>);
type PreparedBatch = (
    PendingHostUpload<u8>,
    PendingHostUpload<i32>,
    Duration,
    Duration,
);

impl ImageFolder {
    fn load(root: &Path, classes: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        let labels = classes
            .iter()
            .enumerate()
            .map(|(label, name)| (name.as_str(), label as i32))
            .collect::<BTreeMap<_, _>>();
        let mut samples = Vec::new();
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let label = *labels
                .get(name.as_ref())
                .ok_or_else(|| format!("unknown class directory {name:?}"))?;
            for image in std::fs::read_dir(entry.path())? {
                let image = image?;
                if image.file_type()?.is_file() {
                    samples.push(Sample {
                        path: image.path(),
                        label,
                    });
                }
            }
        }
        samples.sort_by(|lhs, rhs| lhs.path.cmp(&rhs.path));
        if samples.is_empty() {
            return Err(format!("no images below {}", root.display()).into());
        }
        Ok(Self { samples })
    }

    fn batch(
        &self,
        step: usize,
        batch_size: usize,
        rng: DataRng,
        training: bool,
    ) -> Result<HostBatch, Box<dyn std::error::Error>> {
        let position = step * batch_size;
        let samples = (0..batch_size)
            .into_par_iter()
            .map(|offset| {
                let logical = position + offset;
                let epoch = logical / self.samples.len();
                let within_epoch = logical % self.samples.len();
                let index = if training {
                    permute(within_epoch, self.samples.len(), rng, epoch as u64)
                } else {
                    within_epoch
                };
                let sample = &self.samples[index];
                let sample_rng = rng.sample(epoch as u64, index as u64);
                let top = if training {
                    (sample_rng.word(10, 0) % u64::from(RESIZE - IMAGE as u32 + 1)) as u32
                } else {
                    (RESIZE - IMAGE as u32) / 2
                };
                let left = if training {
                    (sample_rng.word(10, 1) % u64::from(RESIZE - IMAGE as u32 + 1)) as u32
                } else {
                    (RESIZE - IMAGE as u32) / 2
                };
                let decoded = image::open(&sample.path)
                    .map_err(|error| format!("{}: {error}", sample.path.display()))?
                    .resize_to_fill(RESIZE, RESIZE, FilterType::Triangle)
                    .to_rgb8();
                let crop = decoded
                    .view(left, top, IMAGE as u32, IMAGE as u32)
                    .to_image();
                Ok::<_, String>((crop.into_raw(), sample.label))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut images = Vec::with_capacity(batch_size * IMAGE as usize * IMAGE as usize * 3);
        let mut labels = Vec::with_capacity(batch_size);
        for (image, label) in samples {
            images.extend(image);
            labels.push(label);
        }
        Ok((images, labels))
    }
}

fn class_names(root: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut names = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_type().ok()?.is_dir().then(|| entry.file_name()))
        .map(|name| name.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    if names.len() != CLASSES as usize {
        return Err(format!("expected {CLASSES} classes, found {}", names.len()).into());
    }
    Ok(names)
}

fn gcd(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        (lhs, rhs) = (rhs, lhs % rhs);
    }
    lhs
}

fn permute(index: usize, count: usize, rng: DataRng, epoch: u64) -> usize {
    let epoch_rng = rng.sample(epoch, 0);
    let mut multiplier = (epoch_rng.word(1, 0) as usize | 1) % count;
    while gcd(multiplier, count) != 1 {
        multiplier = (multiplier + 2) % count;
    }
    let offset = epoch_rng.word(1, 1) as usize % count;
    (index * multiplier + offset) % count
}

fn basic_block(
    cx: &mut Cx,
    input: &Tensor,
    stage_index: usize,
    block_index: usize,
    channels: i64,
    stride: i64,
    training: bool,
) -> NnResult<Tensor> {
    let mut stage = cx.named(&format!("stage{stage_index}"))?;
    let mut block = stage.named(&format!("block{block_index}"))?;
    let hidden = block
        .named("conv1")?
        .conv2d(channels, [3, 3])
        .options(Conv2dOptions {
            strides: [stride, stride],
            padding: [[1, 1], [1, 1]],
            ..Default::default()
        })
        .bias(false)
        .apply(input)?;
    let hidden = block
        .named("bn1")?
        .batch_norm()
        .training(training)
        .apply(&hidden)?
        .relu()?;
    let hidden = block
        .named("conv2")?
        .conv2d(channels, [3, 3])
        .options(Conv2dOptions {
            padding: [[1, 1], [1, 1]],
            ..Default::default()
        })
        .bias(false)
        .apply(&hidden)?;
    let hidden = block
        .named("bn2")?
        .batch_norm()
        .training(training)
        .apply(&hidden)?;
    let residual = if input.shape()[3] == channels && stride == 1 {
        input.clone()
    } else {
        let projected = block
            .named("shortcut_conv")?
            .conv2d(channels, [1, 1])
            .options(Conv2dOptions {
                strides: [stride, stride],
                ..Default::default()
            })
            .bias(false)
            .apply(input)?;
        block
            .named("shortcut_bn")?
            .batch_norm()
            .training(training)
            .apply(&projected)?
    };
    Ok(hidden.add(&residual)?.relu()?)
}

fn resnet18(cx: &mut Cx, batch_size: i64, training: bool) -> NnResult<Vec<Tensor>> {
    let images = cx.input_dtype(&[batch_size, IMAGE, IMAGE, 3], DType::U8)?;
    let labels = cx.input_dtype(&[batch_size], DType::I32)?;
    let mut augmentation_rng = cx.rng("augmentation")?;
    let flips =
        augmentation_rng.bernoulli(&[batch_size, 1, 1, 1], if training { 0.5 } else { 0.0 })?;
    let images = images.cast(DType::F32)?.mul_scalar(1.0 / 255.0)?;
    let images = flips
        .broadcast_to(images.shape())?
        .select(&images.flip_left_right()?, &images)?
        .normalize_nhwc(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225])?;
    let mut hidden = cx
        .named("stem_conv")?
        .conv2d(64, [7, 7])
        .options(Conv2dOptions {
            strides: [2, 2],
            padding: [[3, 3], [3, 3]],
            ..Default::default()
        })
        .bias(false)
        .apply(&images)?;
    hidden = cx
        .named("stem_bn")?
        .batch_norm()
        .training(training)
        .apply(&hidden)?
        .relu()?
        .avg_pool2d(
            Pool2dOptions {
                window: [3, 3],
                strides: [2, 2],
                padding: [[1, 1], [1, 1]],
            },
            false,
        )?;
    for (stage, channels) in [64, 128, 256, 512].into_iter().enumerate() {
        for block in 0..2 {
            let stride = if stage != 0 && block == 0 { 2 } else { 1 };
            hidden = basic_block(cx, &hidden, stage, block, channels, stride, training)?;
        }
    }
    let features = hidden.mean(&[1, 2], false)?;
    let logits = cx.named("head")?.linear(CLASSES).apply(&features)?;
    let loss = logits
        .cross_entropy_with_indices(&labels, 1)?
        .mean(&[0], false)?;
    let predicted = logits.argmax(1, false)?;
    let correct = predicted
        .le_mask(&labels)?
        .mul(&labels.le_mask(&predicted)?)?
        .sum(&[0], false)?;
    Ok(vec![loss, logits, correct])
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
        let parent = spec.path().rsplit('.').nth(1).unwrap_or_default();
        let batch_norm_weight = spec.path().ends_with(".weight")
            && (parent == "stem_bn" || parent.starts_with("bn") || parent == "shortcut_bn");
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
        result.insert(
            spec.path().to_owned(),
            client.buffer(spec.shape(), &values)?,
        );
    }
    Ok(result)
}

fn prepare_batch(
    dataset: &ImageFolder,
    step: usize,
    batch_size: i64,
    rng: DataRng,
    training: bool,
    gpu: &Client,
) -> Result<PreparedBatch, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let host = dataset.batch(step, batch_size as usize, rng, training)?;
    let decoded = started.elapsed();
    let started = Instant::now();
    let (images, labels) = upload_batch(host, batch_size, gpu)?;
    Ok((images, labels, decoded, started.elapsed()))
}

fn upload_batch(
    host: HostBatch,
    batch_size: i64,
    gpu: &Client,
) -> Result<UploadedBatch, Box<dyn std::error::Error>> {
    let (images, labels) = host;
    let images = gpu.upload_pinned(&[batch_size, IMAGE, IMAGE, 3], images)?;
    let labels = gpu.upload_pinned(&[batch_size], labels)?;
    Ok((images, labels))
}

fn prefetch_batches(
    dataset: Arc<ImageFolder>,
    steps: usize,
    batch_size: usize,
    rng: DataRng,
) -> PipelineResult<BoundedPipeline<HostBatch, String>> {
    BoundedPipeline::from_iter(
        4,
        (0..steps).map(move |step| {
            dataset
                .batch(step, batch_size, rng, true)
                .map_err(|error| error.to_string())
        }),
    )
}

fn initialize_session<'a>(
    model: &'a AppliedModel,
    program: &'a rxla_core::StateProgram,
    schema: &ParamSchema,
    gpu: &Client,
    seed: u64,
) -> Result<rxla_core::Session, Box<dyn std::error::Error>> {
    let mut builder = model.session(program);
    for state in schema
        .states()
        .iter()
        .filter(|state| state.path().ends_with("running_variance"))
    {
        let count = state.shape().iter().product::<i64>() as usize;
        builder = builder.state(
            state.path(),
            gpu.buffer(state.shape(), &vec![1.0_f32; count])?,
        )?;
    }
    builder = builder.rng_seed("augmentation", seed)?;
    Ok(builder.build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.batch_size <= 0 || args.steps == 0 || args.log_every == 0 {
        return Err("batch size, steps, and log interval must be positive".into());
    }
    let classes = class_names(&args.dataset.join("train"))?;
    let train = Arc::new(ImageFolder::load(&args.dataset.join("train"), &classes)?);
    let validation = ImageFolder::load(&args.dataset.join("val"), &classes)?;
    println!(
        "dataset: {} train, {} validation, classes={classes:?}",
        train.samples.len(),
        validation.samples.len()
    );

    let gpu = unsafe { Client::load(&args.gpu_plugin) }?;
    let batch_size = args.batch_size;
    let definition = Model::new(move |cx: &mut Cx| resnet18(cx, batch_size, true));
    let (schema, model) = definition.trace()?;
    let training = prepare_model_sgd(
        &model,
        &schema.select_all(),
        &model.outputs()[0],
        args.learning_rate,
    )?;
    let metrics = Tensor::stack(&[model.outputs()[0].clone(), model.outputs()[2].clone()], 0)?;
    let outputs = training.outputs_with(&[metrics]);
    let mut compiler = Compiler::new(gpu.clone(), CacheLimits::default());
    let program = outputs.compile_stateful(&model, &mut compiler)?;
    let mut session = initialize_session(&model, &program, &schema, &gpu, args.seed)?;
    let mut parameters = initialized_parameters(&gpu, &schema)?;
    let rng = DataRng::new(args.seed);
    let started = Instant::now();
    let mut total_correct = 0;
    let mut first_loss = None;
    let mut last_loss = 0.0;
    let mut phases = PhaseTimes::default();
    let profile_after = args.steps.min(10);
    let batches = prefetch_batches(Arc::clone(&train), args.steps, batch_size as usize, rng)?;

    for step in 0..args.steps {
        let phase = Instant::now();
        let host = batches.recv().ok_or("input pipeline stopped early")??;
        let input_wait = phase.elapsed();
        let phase = Instant::now();
        let (images, labels) = upload_batch(host, batch_size, &gpu)?;
        let uploaded = phase.elapsed();
        let phase = Instant::now();
        let arguments = model.bind(
            &[images.buffer(), labels.buffer()],
            parameters
                .iter()
                .map(|(name, value)| (name.as_str(), value)),
        )?;
        let bound = phase.elapsed();
        let phase = Instant::now();
        let output = session.run(arguments.as_slice())?;
        let executed = phase.elapsed();
        let phase = Instant::now();
        let visible = outputs.commit(output, &mut parameters)?;
        let metrics = visible[0].to_vec::<f32>()?;
        last_loss = metrics[0];
        first_loss.get_or_insert(last_loss);
        let batch_correct = metrics[1] as usize;
        total_correct += batch_correct;
        let measured = phase.elapsed();
        if args.profile && step >= profile_after {
            phases.samples += 1;
            phases.input_wait += input_wait;
            phases.upload += uploaded;
            phases.bind += bound;
            phases.execute += executed;
            phases.metrics += measured;
        }
        if step % args.log_every == 0 || step + 1 == args.steps {
            println!(
                "step {step:>4}: loss {last_loss:.6}, accuracy {:.1}%",
                batch_correct as f64 * 100.0 / batch_size as f64
            );
        }
    }
    batches.finish()?;
    let seconds = started.elapsed().as_secs_f64();
    let trained = args.steps as f64 * batch_size as f64;
    println!(
        "train: loss {:.6} -> {last_loss:.6}, mean accuracy {:.1}%, {:.1} images/s",
        first_loss.unwrap(),
        total_correct as f64 * 100.0 / trained,
        trained / seconds,
    );
    if args.profile {
        phases.print(batch_size);
    }

    let inference =
        Model::new(move |cx: &mut Cx| resnet18(cx, batch_size, false)).apply(&schema)?;
    let inference_metrics = Tensor::stack(
        &[
            inference.outputs()[0].clone(),
            inference.outputs()[2].clone(),
        ],
        0,
    )?;
    let inference_program =
        inference.compile_stateful_tensors(&mut compiler, &[inference_metrics])?;
    let resident = session.into_state();
    let mut inference_builder = inference.session(&inference_program);
    for ((name, _), (_, buffer)) in model.states().zip(resident) {
        inference_builder = inference_builder.state(name, buffer)?;
    }
    let mut inference_session = inference_builder.build()?;
    let validation_steps = validation.samples.len() / batch_size as usize;
    let mut validation_correct = 0;
    let mut validation_loss = 0.0;
    for step in 0..validation_steps {
        let (images, labels, _, _) =
            prepare_batch(&validation, step, batch_size, rng, false, &gpu)?;
        let arguments = inference.bind(
            &[images.buffer(), labels.buffer()],
            parameters
                .iter()
                .map(|(name, value)| (name.as_str(), value)),
        )?;
        let output = inference_session.run(arguments.as_slice())?;
        let metrics = output[0].to_vec::<f32>()?;
        validation_loss += metrics[0];
        validation_correct += metrics[1] as usize;
    }
    let evaluated = validation_steps * batch_size as usize;
    println!(
        "validation: loss {:.6}, top-1 {:.2}% ({validation_correct}/{evaluated})",
        validation_loss / validation_steps as f32,
        validation_correct as f64 * 100.0 / evaluated as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resnet18_schema_has_eighteen_convolutions_and_stateful_norms() {
        let (schema, model) = Model::new(|cx: &mut Cx| resnet18(cx, 2, true))
            .trace()
            .unwrap();
        let convolution_count = schema
            .parameters()
            .iter()
            .filter(|parameter| {
                parameter.path().ends_with("conv.weight")
                    || parameter.path().contains("_conv.weight")
                    || parameter.path().contains(".conv1.weight")
                    || parameter.path().contains(".conv2.weight")
            })
            .count();
        assert_eq!(convolution_count, 20); // 17 main-path + 3 projection convolutions.
        assert_eq!(model.outputs()[1].shape(), [2, CLASSES]);
        assert_eq!(schema.states().len(), 44); // 20 BatchNorm pairs + four RNG words.
        assert!(
            schema
                .states()
                .iter()
                .any(|state| state.path() == "augmentation.counter_low")
        );
    }
}
