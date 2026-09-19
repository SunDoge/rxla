//! Export raw RXLA YOLOX heads for comparison with the official PyTorch model.

use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::{
    YoloConfig, YoloLossConfig, YoloSimOtaConfig, YoloTargets, yolo, yolo_forward,
    yolox_loss_with_l1_weight, yolox_simota,
};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use rxla_train::{
    DecodedYoloBatch, DetectionTarget, SgdOptions, SimOtaConfig, YoloHead,
    apply_model_sgd_with_learning_rate,
};
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
    /// Number of resident optimizer steps used by the trajectory oracle.
    #[arg(long, default_value_t = 1)]
    training_steps: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.size <= 0 || args.size % 32 != 0 || args.training_steps == 0 {
        return Err(
            "size must be a positive multiple of 32 and training steps must be nonzero".into(),
        );
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
         class_indices: rxla_core::Tensor,
         valid: rxla_core::Tensor,
         l1_weight: rxla_core::Tensor| {
            let heads = [head0, head1, head2];
            let assignments = yolox_simota(
                &heads,
                &YoloTargets {
                    boxes,
                    class_indices,
                    valid,
                },
                YoloSimOtaConfig::default(),
            )?;
            let loss = yolox_loss_with_l1_weight(
                &heads,
                &assignments,
                YoloLossConfig::default(),
                &l1_weight,
            )?;
            let gradients = loss.total.grad(&heads)?;
            let metrics = rxla_core::Tensor::stack(
                &[
                    loss.total.clone(),
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
                gradients[0].clone(),
                gradients[1].clone(),
                gradients[2].clone(),
            ))
        },
    )
    .inputs((
        ModelInput::new(model_shapes[0]),
        ModelInput::new(model_shapes[1]),
        ModelInput::new(model_shapes[2]),
        ModelInput::new([1, 2, 4]),
        ModelInput::new([1, 2, 1]).with_dtype(rxla_core::DType::I32),
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
    let target_classes = client.buffer(&[1, 2, 1], &[0_i32, 7])?;
    let valid = client.buffer(&[1, 2, 1], &[1.0_f32, 1.0])?;
    let l1_weight = client.buffer(&[], &[1.0_f32])?;
    let loss_outputs: [Buffer; 7] = loss_session.run([
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
    for (index, gradient) in loss_outputs[4..].iter().enumerate() {
        write_f32(
            &args.output.join(format!("head{index}_gradient.f32")),
            &gradient.to_vec::<f32>()?,
        )?;
    }

    // The head-gradient oracle above isolates loss and SimOTA semantics. This
    // second, training-mode graph differentiates through the actual model,
    // including BatchNorm and all three SPP max-pool branches, and catches
    // backbone/autodiff mismatches that raw head cotangents cannot observe.
    const MODEL_GRADIENT_PATHS: [&str; 4] = [
        "backbone.backbone.stem.conv.conv.weight",
        "backbone.backbone.dark5.1.conv1.conv.weight",
        "backbone.backbone.dark5.1.conv2.conv.weight",
        "head.cls_preds.0.weight",
    ];
    let full_definition = Model::new(
        |cx: Cx,
         image: rxla_core::Tensor,
         boxes: rxla_core::Tensor,
         class_indices: rxla_core::Tensor,
         valid: rxla_core::Tensor,
         l1_weight: rxla_core::Tensor,
         learning_rate: rxla_core::Tensor| {
            let heads = yolo(cx, &image, &YoloConfig::yolox_tiny(80))?;
            let assignments = yolox_simota(
                &heads,
                &YoloTargets {
                    boxes,
                    class_indices,
                    valid,
                },
                YoloSimOtaConfig::default(),
            )?;
            let loss = yolox_loss_with_l1_weight(
                &heads,
                &assignments,
                YoloLossConfig::default(),
                &l1_weight,
            )?;
            Ok::<_, rxla_models::Error>((
                loss.total,
                assignments.boxes,
                assignments.objectness,
                assignments.classes,
                learning_rate,
            ))
        },
    )
    .inputs((
        ModelInput::new([1, args.size, args.size, 3]),
        ModelInput::new([1, 2, 4]),
        ModelInput::new([1, 2, 1]).with_dtype(rxla_core::DType::I32),
        ModelInput::new([1, 2, 1]),
        ModelInput::new([]),
        ModelInput::new([]),
    ))
    .training();
    let (all_parameters, mut full_model) = full_definition.trace_resident_all()?;
    let selected = all_parameters
        .clone()
        .matching(|_, parameter| MODEL_GRADIENT_PATHS.contains(&parameter.path()));
    if selected.len() != MODEL_GRADIENT_PATHS.len() {
        return Err("full-model gradient oracle paths do not match the model schema".into());
    }
    let selected_paths = selected
        .parameters()
        .map(|(_, parameter)| parameter.path().to_owned())
        .collect::<Vec<_>>();
    let parameter_tensors = full_model.parameter_tensors(&selected)?;
    let gradients = full_model.outputs()[0].grad(&parameter_tensors)?;
    let learning_rate_tensor = full_model.outputs()[4].clone();
    let full_loss = full_model.outputs()[0].clone();
    apply_model_sgd_with_learning_rate(
        &mut full_model,
        &all_parameters,
        &full_loss,
        &learning_rate_tensor,
        |parameter| {
            let decay =
                parameter.path().ends_with(".weight") && !parameter.path().contains(".bn.weight");
            SgdOptions::default()
                .momentum(0.9)
                .nesterov(true)
                .weight_decay(if decay { 5e-4 } else { 0.0 })
        },
    )?;
    let full_outputs = full_model.outputs()[..4]
        .iter()
        .cloned()
        .chain(gradients)
        .collect::<Vec<_>>();
    let full_program = full_model.compile_stateful_tensors(&mut compiler, &full_outputs)?;
    let full_weights = checkpoint.load_model_schema(&client, full_model.schema())?;
    let mut full_session = full_weights
        .initialize_raw(full_model.session(&full_program))?
        .build()?;
    let learning_rate = client.buffer(&[], &[0.00015625_f32])?;
    let full_outputs: Vec<Buffer> = full_session.run(&[
        &input,
        &target_boxes,
        &target_classes,
        &valid,
        &l1_weight,
        &learning_rate,
    ])?;
    write_f32(
        &args.output.join("model_training_loss.f32"),
        &full_outputs[0].to_vec::<f32>()?,
    )?;
    for (name, value) in ["boxes", "objectness", "classes"]
        .into_iter()
        .zip(&full_outputs[1..4])
    {
        write_f32(
            &args.output.join(format!("model_assignment_{name}.f32")),
            &value.to_vec::<f32>()?,
        )?;
    }
    for ((path, expected_path), gradient) in selected_paths
        .iter()
        .zip(MODEL_GRADIENT_PATHS)
        .zip(&full_outputs[4..])
    {
        if path != expected_path {
            return Err(format!(
                "gradient schema order mismatch: expected {expected_path}, got {path}"
            )
            .into());
        }
        write_f32(
            &args
                .output
                .join(format!("model_gradient_{}.f32", path.replace('.', "_"))),
            &gradient.to_vec::<f32>()?,
        )?;
    }
    for path in [
        "backbone.backbone.stem.conv.bn.running_mean",
        "backbone.backbone.stem.conv.bn.running_variance",
    ] {
        let buffer = full_model
            .state_buffers(&full_session)?
            .into_iter()
            .find(|(candidate, _)| *candidate == path)
            .ok_or_else(|| format!("missing training state {path}"))?
            .1;
        write_f32(
            &args
                .output
                .join(format!("model_state_{}.f32", path.replace('.', "_"))),
            &buffer.to_vec::<f32>()?,
        )?;
    }
    let updated = full_model.selected_resident_parameter_buffers(&full_session, &selected)?;
    for (path, buffer) in updated {
        write_f32(
            &args
                .output
                .join(format!("model_update_{}.f32", path.replace('.', "_"))),
            &buffer.to_vec::<f32>()?,
        )?;
    }

    // Replay more updates in the same resident session. Each step exports its
    // device-selected assignment so the Python oracle can use exactly the same
    // detached targets and isolate forward/BatchNorm/SGD trajectory semantics
    // from SimOTA's intentionally discontinuous matching decisions.
    for step in 2..=args.training_steps {
        let outputs: Vec<Buffer> = full_session.run(&[
            &input,
            &target_boxes,
            &target_classes,
            &valid,
            &l1_weight,
            &learning_rate,
        ])?;
        write_f32(
            &args.output.join(format!("model_step_{step}_loss.f32")),
            &outputs[0].to_vec::<f32>()?,
        )?;
        for (name, value) in ["boxes", "objectness", "classes"]
            .into_iter()
            .zip(&outputs[1..4])
        {
            write_f32(
                &args
                    .output
                    .join(format!("model_step_{step}_assignment_{name}.f32")),
                &value.to_vec::<f32>()?,
            )?;
        }
    }
    for (path, buffer) in
        full_model.selected_resident_parameter_buffers(&full_session, &selected)?
    {
        write_f32(
            &args.output.join(format!(
                "model_final_parameter_{}.f32",
                path.replace('.', "_")
            )),
            &buffer.to_vec::<f32>()?,
        )?;
    }
    let state = full_model.state_buffers(&full_session)?;
    for path in [
        "backbone.backbone.stem.conv.bn.running_mean",
        "backbone.backbone.stem.conv.bn.running_variance",
    ] {
        let buffer = state
            .iter()
            .find(|(candidate, _)| *candidate == path)
            .ok_or_else(|| format!("missing final training state {path}"))?
            .1;
        write_f32(
            &args
                .output
                .join(format!("model_final_state_{}.f32", path.replace('.', "_"))),
            &buffer.to_vec::<f32>()?,
        )?;
    }
    for path in &selected_paths {
        let state_path = format!("__optimizer.sgd.{path}.momentum");
        let buffer = state
            .iter()
            .find(|(candidate, _)| *candidate == state_path)
            .ok_or_else(|| format!("missing final optimizer state {state_path}"))?
            .1;
        write_f32(
            &args.output.join(format!(
                "model_final_momentum_{}.f32",
                path.replace('.', "_")
            )),
            &buffer.to_vec::<f32>()?,
        )?;
    }
    println!(
        "output={} size={} training_steps={}",
        args.output.display(),
        args.size,
        args.training_steps
    );
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
