//! ResNet-18 training and validation on an ImageFolder-style Imagenette tree.

use clap::Parser;
use image::{GenericImageView, imageops::FilterType};
use rayon::prelude::*;
use rxla_core::{
    CacheLimits, Client, Compiler, Conv2dOptions, DType, PendingHostUpload, Pool2dOptions, Tensor,
};
use rxla_nn::{
    AppliedModel, BatchNorm, Conv2d, Cx, ExecutionMode, Linear, Model, ModelInput,
    Result as NnResult, TensorApply, path,
};
use rxla_train::{BoundedPipeline, DataRng, PipelineResult, apply_model_sgd};
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
    execute: Duration,
    metrics: Duration,
}

impl PhaseTimes {
    fn print(&self, batch_size: i64) {
        if self.samples == 0 {
            return;
        }
        let milliseconds = |duration: Duration| duration.as_secs_f64() * 1_000.0;
        let total = self.input_wait + self.upload + self.execute + self.metrics;
        println!(
            "profile: {samples} samples, mean step {total:.3} ms: input-wait {input_wait:.3} ms ({input_wait_share:.1}%), upload {upload:.3} ms ({upload_share:.1}%), execute {execute:.3} ms ({execute_share:.1}%), metrics {metrics:.3} ms ({metrics_share:.1}%); measured throughput {throughput:.1} images/s",
            samples = self.samples,
            total = milliseconds(total) / self.samples as f64,
            input_wait = milliseconds(self.input_wait) / self.samples as f64,
            input_wait_share = self.input_wait.as_secs_f64() * 100.0 / total.as_secs_f64(),
            upload = milliseconds(self.upload) / self.samples as f64,
            upload_share = self.upload.as_secs_f64() * 100.0 / total.as_secs_f64(),
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

fn basic_block(cx: Cx, input: &Tensor, channels: i64, stride: i64) -> NnResult<Tensor> {
    let conv1 = path!(cx / "conv1")?.layer(
        Conv2d::new(channels, [3, 3])
            .options(Conv2dOptions {
                strides: [stride, stride],
                padding: [[1, 1], [1, 1]],
                ..Default::default()
            })
            .bias(false),
    );
    let bn1 = path!(cx / "bn1")?.layer(BatchNorm::new());
    let conv2 = path!(cx / "conv2")?.layer(
        Conv2d::new(channels, [3, 3])
            .options(Conv2dOptions {
                padding: [[1, 1], [1, 1]],
                ..Default::default()
            })
            .bias(false),
    );
    let bn2 = path!(cx / "bn2")?.layer(BatchNorm::new());

    let hidden = input.apply(&conv1)?;
    let hidden = hidden.apply(&bn1)?.relu()?;
    let hidden = hidden.apply(&conv2)?;
    let hidden = hidden.apply(&bn2)?;
    let residual = if input.shape()[3] == channels && stride == 1 {
        input.clone()
    } else {
        let shortcut_conv = path!(cx / "downsample" / 0)?.layer(
            Conv2d::new(channels, [1, 1])
                .options(Conv2dOptions {
                    strides: [stride, stride],
                    ..Default::default()
                })
                .bias(false),
        );
        let shortcut_bn = path!(cx / "downsample" / 1)?.layer(BatchNorm::new());
        input.apply(&shortcut_conv)?.apply(&shortcut_bn)?
    };
    Ok(hidden.add(&residual)?.relu()?)
}

fn resnet18(cx: Cx, images: Tensor, labels: Tensor) -> NnResult<(Tensor, Tensor, Tensor)> {
    let training = cx.mode() == ExecutionMode::Training;
    let batch_size = images.shape()[0];
    let mut augmentation_rng = cx.rng("augmentation")?;
    let flips =
        augmentation_rng.bernoulli(&[batch_size, 1, 1, 1], if training { 0.5 } else { 0.0 })?;
    let images = images.cast(DType::F32)?.mul_scalar(1.0 / 255.0)?;
    let images = flips
        .broadcast_to(images.shape())?
        .select(&images.flip_left_right()?, &images)?
        .normalize_nhwc(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225])?;
    let stem_conv = path!(cx / "conv1")?.layer(
        Conv2d::new(64, [7, 7])
            .options(Conv2dOptions {
                strides: [2, 2],
                padding: [[3, 3], [3, 3]],
                ..Default::default()
            })
            .bias(false),
    );
    let stem_bn = path!(cx / "bn1")?.layer(BatchNorm::new());
    let head = path!(cx / "fc")?.layer(Linear::new(CLASSES));

    let mut hidden = images.apply(&stem_conv)?;
    hidden = hidden.apply(&stem_bn)?.relu()?.avg_pool2d(
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
            let stage_name = format!("layer{}", stage + 1);
            let block_cx = path!(cx / stage_name / block)?;
            hidden = basic_block(block_cx, &hidden, channels, stride)?;
        }
    }
    let features = hidden.mean(&[1, 2], false)?;
    let logits = features.apply(&head)?;
    let loss = logits
        .cross_entropy_with_indices(&labels, 1)?
        .mean(&[0], false)?;
    let predicted = logits.argmax(1, false)?;
    let correct = predicted
        .le_mask(&labels)?
        .mul(&labels.le_mask(&predicted)?)?
        .sum(&[0], false)?;
    Ok((loss, logits, correct))
}

fn resnet18_inputs(batch_size: i64) -> (ModelInput, ModelInput) {
    (
        ModelInput::new([batch_size, IMAGE, IMAGE, 3]).with_dtype(DType::U8),
        ModelInput::new([batch_size]).with_dtype(DType::I32),
    )
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
    seed: u64,
) -> Result<rxla_core::Session, Box<dyn std::error::Error>> {
    let mut builder = model.session(program).initialize_parameters(seed)?;
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
    let definition = Model::new(resnet18)
        .inputs(resnet18_inputs(batch_size))
        .training();
    let (trainable, mut model) = definition.trace_resident_all()?;
    let metrics = Tensor::stack(&[model.outputs()[0].clone(), model.outputs()[2].clone()], 0)?;
    let loss = model.outputs()[0].clone();
    apply_model_sgd(&mut model, &trainable, &loss, args.learning_rate)?;
    let mut compiler = Compiler::new(gpu.clone(), CacheLimits::default());
    let program = model.compile_stateful_tensors(&mut compiler, &[metrics])?;
    let mut session = initialize_session(&model, &program, args.seed)?;
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
        let visible = session.run(&[images.buffer(), labels.buffer()])?;
        let executed = phase.elapsed();
        let phase = Instant::now();
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

    let inference = Model::new(resnet18)
        .inputs(resnet18_inputs(batch_size))
        .inference()
        .trace_resident(&trainable)?;
    let inference_metrics = Tensor::stack(
        &[
            inference.outputs()[0].clone(),
            inference.outputs()[2].clone(),
        ],
        0,
    )?;
    let inference_program =
        inference.compile_stateful_tensors(&mut compiler, &[inference_metrics])?;
    let snapshot = model.take_session(session)?;
    let mut inference_session = snapshot
        .restore_model(inference.session(&inference_program))?
        .build()?;
    let validation_steps = validation.samples.len() / batch_size as usize;
    let mut validation_correct = 0;
    let mut validation_loss = 0.0;
    for step in 0..validation_steps {
        let (images, labels, _, _) =
            prepare_batch(&validation, step, batch_size, rng, false, &gpu)?;
        let output = inference_session.run(&[images.buffer(), labels.buffer()])?;
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
        let model = Model::new(resnet18)
            .inputs(resnet18_inputs(2))
            .training()
            .trace()
            .unwrap();
        let schema = model.schema();
        let convolution_count = schema
            .parameters()
            .iter()
            .filter(|parameter| parameter.shape().len() == 4)
            .count();
        assert_eq!(convolution_count, 20); // 17 main-path + 3 projection convolutions.
        assert!(
            schema
                .parameters()
                .iter()
                .any(|parameter| parameter.path() == "layer4.0.downsample.0.weight")
        );
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
