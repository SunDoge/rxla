//! Train an attention LoRA against a prepared Stable Diffusion noise-prediction batch.

use clap::{Parser, ValueEnum};
use rxla_core::{Buffer, CacheLimits, Client, Compiler, DType, MemoryFittingLevel};
use rxla_models::{UnetConfig, unet};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::{SafeTensors, save_parameter_selection_new};
use rxla_train::{AdamOptions, apply_model_adam, select_lora};
use std::{collections::HashMap, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Train and export a Stable Diffusion attention LoRA")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Diffusers model directory containing the UNet checkpoint.
    model: PathBuf,
    /// F32 SafeTensors batch produced by benchmarks/prepare_sd_lora.py.
    batch: PathBuf,
    /// New adapter SafeTensors path.
    output: PathBuf,
    #[arg(long, default_value_t = 2)]
    rank: i64,
    #[arg(long, default_value_t = 2.0)]
    alpha: f32,
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,
    #[arg(long, default_value_t = 20)]
    steps: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// UNet architecture matching the supplied checkpoint.
    #[arg(long, value_enum, default_value_t = ModelPreset::Tiny)]
    preset: ModelPreset,
    /// Internal CUDA compute precision. BF16 is experimental for fused optimizer graphs;
    /// the public input/output ABI remains F32.
    #[arg(long, value_enum, default_value_t = ComputeDtype::F32)]
    compute_dtype: ComputeDtype,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelPreset {
    Tiny,
    SdV1,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ComputeDtype {
    F32,
    Bf16,
}

impl ModelPreset {
    fn config(self) -> UnetConfig {
        match self {
            Self::Tiny => UnetConfig::tiny(),
            Self::SdV1 => UnetConfig::stable_diffusion_v1(),
        }
    }
}

fn run(args: Args) -> Result<()> {
    if args.rank <= 0
        || !args.alpha.is_finite()
        || !args.learning_rate.is_finite()
        || args.learning_rate < 0.0
        || args.steps == 0
    {
        return Err("rank/steps must be positive and alpha/learning-rate must be finite".into());
    }
    let client = unsafe { Client::load(&args.plugin) }?;
    let mut batch = SafeTensors::open(&args.batch)?;
    let latent = batch.info("sample").ok_or("batch is missing sample")?;
    let timestep = batch
        .info("timestep_embedding")
        .ok_or("batch is missing timestep_embedding")?;
    let context = batch.info("context").ok_or("batch is missing context")?;
    let target = batch.info("target").ok_or("batch is missing target")?;
    let latent_shape = latent
        .shape
        .iter()
        .map(|&dimension| i64::try_from(dimension))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let timestep_shape = timestep
        .shape
        .iter()
        .map(|&dimension| i64::try_from(dimension))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let context_shape = context
        .shape
        .iter()
        .map(|&dimension| i64::try_from(dimension))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let target_shape = target
        .shape
        .iter()
        .map(|&dimension| i64::try_from(dimension))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if latent_shape != target_shape {
        return Err("sample and target shapes differ".into());
    }

    let config = args.preset.config().attention_lora(args.rank, args.alpha);
    let definition = Model::new(move |cx: Cx, sample, timestep, context, target| {
        let prediction = unet(cx, &sample, &timestep, &context, &config)?;
        Ok::<_, rxla_models::Error>(
            prediction
                .sub(&target)?
                .square()?
                .mean(&[0, 1, 2, 3], false)?,
        )
    })
    .inputs((
        ModelInput::new(latent_shape.clone()),
        ModelInput::new(timestep_shape),
        ModelInput::new(context_shape),
        ModelInput::new(target_shape),
    ));

    let schema_trace = definition.trace()?;
    let adapter = select_lora(schema_trace.schema());
    let base = schema_trace
        .schema()
        .select_all()
        .matching(|id, _| !adapter.contains(id));
    eprintln!(
        "base_parameters={} adapter_parameters={}",
        base.len(),
        adapter.len()
    );

    let mut training = definition.trace_resident(&adapter)?;
    let loss = training.outputs()[0].clone();
    apply_model_adam(
        &mut training,
        &adapter,
        &loss,
        AdamOptions {
            learning_rate: args.learning_rate,
            ..AdamOptions::default()
        },
    )?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default())
        .with_memory_fitting_level(MemoryFittingLevel::O2);
    if matches!(args.compute_dtype, ComputeDtype::Bf16) {
        compiler = compiler.with_compute_dtype(DType::BF16);
    }
    let compiled = training.compile_stateful(&mut compiler)?;

    let mut checkpoint = SafeTensors::open(
        args.model
            .join("unet")
            .join("diffusion_pytorch_model.safetensors"),
    )?;
    let base = checkpoint.load_parameter_selection(&client, &base)?;
    let sample = batch.upload_f32(&client, "sample")?;
    let timestep = batch.upload_f32(&client, "timestep_embedding")?;
    let context = batch.upload_f32(&client, "context")?;
    let target = batch.upload_f32(&client, "target")?;
    let mut session = compiled.initialize_session(args.seed)?;

    let bindings = base.bindings();
    let mut initial_loss = None;
    let mut final_loss = f32::NAN;
    for step in 0..args.steps {
        let loss: Buffer = session.run_with_parameters(
            (&sample, &timestep, &context, &target),
            bindings.iter().copied(),
        )?;
        final_loss = loss.to_vec::<f32>()?[0];
        initial_loss.get_or_insert(final_loss);
        println!("step={} loss={final_loss:.8}", step + 1);
    }

    let metadata = HashMap::from([
        ("format".to_owned(), "rxla-sd-lora-v1".to_owned()),
        ("base_model".to_owned(), args.model.display().to_string()),
        ("rank".to_owned(), args.rank.to_string()),
        ("alpha".to_owned(), args.alpha.to_string()),
        ("steps".to_owned(), args.steps.to_string()),
        ("preset".to_owned(), format!("{:?}", args.preset)),
    ]);
    save_parameter_selection_new(
        &args.output,
        &training,
        session.raw(),
        &adapter,
        Some(&metadata),
    )?;
    println!(
        "saved={} initial_loss={:.8} final_loss={final_loss:.8}",
        args.output.display(),
        initial_loss.unwrap_or(f32::NAN)
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
