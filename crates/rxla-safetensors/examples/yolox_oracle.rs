//! Export raw RXLA YOLOX heads for comparison with the official PyTorch model.

use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::{
    YoloConfig, YoloLossConfig, YoloSimOtaConfig, YoloTargets, yolo_forward,
    yolox_loss_with_l1_weight, yolox_simota,
};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use rxla_train::{DecodedYoloBatch, DetectionTarget, SimOtaConfig, YoloHead};
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
    let heads = outputs[8..]
        .iter()
        .map(Buffer::to_vec::<f32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let size = usize::try_from(args.size)?;
    let shapes = [8, 16, 32].map(|stride| [1, size / stride, size / stride, 85]);
    let model_shapes = [8, 16, 32].map(|stride| [1, args.size / stride, args.size / stride, 85]);
    let decoded = DecodedYoloBatch::from_heads([
        YoloHead::new(&heads[0], shapes[0]).stride(8),
        YoloHead::new(&heads[1], shapes[1]).stride(16),
        YoloHead::new(&heads[2], shapes[2]).stride(32),
    ])?;
    let assignments = SimOtaConfig::default().assign(
        &decoded.input(),
        &[
            DetectionTarget {
                batch: 0,
                class: 0,
                cxcywh: [20.0, 20.0, 12.0, 16.0],
            },
            DetectionTarget {
                batch: 0,
                class: 7,
                cxcywh: [44.0, 42.0, 18.0, 14.0],
            },
        ],
    )?;
    let loss_definition = Model::new(
        |_cx: Cx,
         head0: rxla_core::Tensor,
         head1: rxla_core::Tensor,
         head2: rxla_core::Tensor,
         boxes: rxla_core::Tensor,
         classes: rxla_core::Tensor,
         valid: rxla_core::Tensor,
         l1_weight: rxla_core::Tensor| {
            let assignments = yolox_simota(
                &[head0.clone(), head1.clone(), head2.clone()],
                &YoloTargets {
                    boxes,
                    classes,
                    valid,
                },
                YoloSimOtaConfig::default(),
            )?;
            let loss = yolox_loss_with_l1_weight(
                &[head0, head1, head2],
                &assignments,
                YoloLossConfig::default(),
                &l1_weight,
            )?;
            let metrics = rxla_core::Tensor::stack(
                &[
                    loss.total,
                    loss.boxes,
                    loss.objectness,
                    loss.classes,
                    loss.l1,
                ],
                0,
            )?;
            Ok::<_, rxla_models::Error>((
                assignments.boxes,
                assignments.objectness,
                assignments.classes,
                metrics,
            ))
        },
    )
    .inputs((
        ModelInput::new(model_shapes[0]),
        ModelInput::new(model_shapes[1]),
        ModelInput::new(model_shapes[2]),
        ModelInput::new([1, 2, 4]),
        ModelInput::new([1, 2, 80]),
        ModelInput::new([1, 2, 1]),
        ModelInput::new([]),
    ))
    .inference();
    let (_, loss_model) = loss_definition.trace_resident_all()?;
    let loss_program = loss_model.compile_stateful(&mut compiler)?;
    let mut loss_session = loss_program.session().initialize_parameters(0)?.build()?;
    let target_boxes = client.buffer(
        &[1, 2, 4],
        &[20.0_f32, 20.0, 12.0, 16.0, 44.0, 42.0, 18.0, 14.0],
    )?;
    let mut target_classes = vec![0.0_f32; 160];
    target_classes[0] = 1.0;
    target_classes[80 + 7] = 1.0;
    let target_classes = client.buffer(&[1, 2, 80], &target_classes)?;
    let valid = client.buffer(&[1, 2, 1], &[1.0_f32, 1.0])?;
    let l1_weight = client.buffer(&[], &[1.0_f32])?;
    let loss_outputs: [Buffer; 4] = loss_session.run([
        &outputs[8],
        &outputs[9],
        &outputs[10],
        &target_boxes,
        &target_classes,
        &valid,
        &l1_weight,
    ])?;
    let device_boxes = loss_outputs[0].to_vec::<f32>()?;
    let device_objectness = loss_outputs[1].to_vec::<f32>()?;
    let device_classes = loss_outputs[2].to_vec::<f32>()?;
    if device_boxes != assignments.boxes
        || device_objectness != assignments.objectness
        || device_classes
            .iter()
            .zip(&assignments.classes)
            .any(|(left, right)| (left - right).abs() > 1e-6)
    {
        return Err("device SimOTA differs from the host reference".into());
    }
    write_f32(&args.output.join("assignment_boxes.f32"), &device_boxes)?;
    write_f32(
        &args.output.join("assignment_objectness.f32"),
        &device_objectness,
    )?;
    write_f32(&args.output.join("assignment_classes.f32"), &device_classes)?;
    write_f32(
        &args.output.join("loss.f32"),
        &loss_outputs[3].to_vec::<f32>()?,
    )?;
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
