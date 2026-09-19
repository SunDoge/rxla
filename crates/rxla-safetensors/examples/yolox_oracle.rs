//! Export raw RXLA YOLOX heads for comparison with the official PyTorch model.

use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::{
    YoloAssignments, YoloConfig, YoloLossConfig, yolo_forward, yolox_loss_with_l1_weight,
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
    write_f32(
        &args.output.join("assignment_boxes.f32"),
        &assignments.boxes,
    )?;
    write_f32(
        &args.output.join("assignment_objectness.f32"),
        &assignments.objectness,
    )?;
    write_f32(
        &args.output.join("assignment_classes.f32"),
        &assignments.classes,
    )?;

    let anchors = i64::try_from(assignments.anchors)?;
    let loss_definition = Model::new(
        |_cx: Cx, head0, head1, head2, boxes, objectness, classes, l1_weight| {
            let loss = yolox_loss_with_l1_weight(
                &[head0, head1, head2],
                &YoloAssignments {
                    boxes,
                    objectness,
                    classes,
                },
                YoloLossConfig::default(),
                &l1_weight,
            )?;
            Ok::<_, rxla_models::Error>((
                loss.total,
                loss.boxes,
                loss.objectness,
                loss.classes,
                loss.l1,
            ))
        },
    )
    .inputs((
        ModelInput::new(model_shapes[0]),
        ModelInput::new(model_shapes[1]),
        ModelInput::new(model_shapes[2]),
        ModelInput::new([1, anchors, 4]),
        ModelInput::new([1, anchors, 1]),
        ModelInput::new([1, anchors, 80]),
        ModelInput::new([]),
    ))
    .inference();
    let (_, loss_model) = loss_definition.trace_resident_all()?;
    let metrics = rxla_core::Tensor::stack(loss_model.outputs(), 0)?;
    let loss_program = loss_model.compile_stateful_tensors(&mut compiler, &[metrics])?;
    let mut loss_session = loss_model
        .session(&loss_program)
        .initialize_parameters(0)?
        .build()?;
    let boxes = client.buffer(&[1, anchors, 4], &assignments.boxes)?;
    let objectness = client.buffer(&[1, anchors, 1], &assignments.objectness)?;
    let classes = client.buffer(&[1, anchors, 80], &assignments.classes)?;
    let l1_weight = client.buffer(&[], &[1.0_f32])?;
    let loss_outputs = loss_session.run(&[
        &outputs[8],
        &outputs[9],
        &outputs[10],
        &boxes,
        &objectness,
        &classes,
        &l1_weight,
    ])?;
    write_f32(
        &args.output.join("loss.f32"),
        &loss_outputs[0].to_vec::<f32>()?,
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
