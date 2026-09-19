//! Export raw RXLA YOLOX heads for comparison with the official PyTorch model.

use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::{YoloConfig, yolo_forward};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Export deterministic raw YOLOX heads for a numerical oracle")]
struct Args {
    plugin: PathBuf,
    weights: PathBuf,
    output: PathBuf,
    #[arg(long, default_value_t = 64)]
    size: i64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.size <= 0 || args.size % 32 != 0 {
        return Err("size must be a positive multiple of 32".into());
    }
    let client = unsafe { Client::load(&args.plugin) }?;
    let definition = Model::new(|cx: Cx, image| {
        let forward = yolo_forward(cx, &image, &YoloConfig::yolox_tiny(80))?;
        Ok::<_, rxla_models::Error>((
            forward.backbone_features[0].clone(),
            forward.backbone_features[1].clone(),
            forward.backbone_features[2].clone(),
            forward.backbone_features[3].clone(),
            forward.backbone_features[4].clone(),
            forward.features[0].clone(),
            forward.features[1].clone(),
            forward.features[2].clone(),
            forward.heads[0].clone(),
            forward.heads[1].clone(),
            forward.heads[2].clone(),
        ))
    })
    .inputs(ModelInput::new([1, args.size, args.size, 3]))
    .inference();
    let (_, model) = definition.trace_resident_all()?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compiled = model.compile_stateful(&mut compiler)?;
    let mut checkpoint = SafeTensors::open(&args.weights)?;
    checkpoint.require_metadata(&[("format", "rxla-yolox-v1")])?;
    let weights = checkpoint.load_model_schema(&client, model.schema())?;
    let mut session = weights.initialize(compiled.session())?.build()?;

    let elements = usize::try_from(args.size * args.size * 3)?;
    let input = (0..elements)
        .map(|index| ((index * 17 + 13) % 256) as f32)
        .collect::<Vec<_>>();
    let input = client.buffer(&[1, args.size, args.size, 3], &input)?;
    let outputs: [Buffer; 11] = session.run(&input)?;

    std::fs::create_dir_all(&args.output)?;
    for (name, output) in [
        "stem", "dark2", "dark3", "dark4", "dark5", "feature0", "feature1", "feature2", "head0",
        "head1", "head2",
    ]
    .into_iter()
    .zip(&outputs)
    {
        write_f32(
            &args.output.join(format!("{name}.f32")),
            &output.to_vec::<f32>()?,
        )?;
    }
    println!("output={} size={}", args.output.display(), args.size);
    Ok(())
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
