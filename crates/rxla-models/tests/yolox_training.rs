use rxla_core::{CacheLimits, Client, Compiler, Tensor};
use rxla_models::{YoloAssignments, YoloConfig, YoloLossConfig, yolo, yolox_loss};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_train::{AdamOptions, apply_model_adam};

const IMAGE: i64 = 64;
const CLASSES: i64 = 2;
const ANCHORS: i64 = 84;

fn training_model(
    cx: Cx,
    images: Tensor,
    boxes: Tensor,
    objectness: Tensor,
    classes: Tensor,
) -> rxla_models::Result<(Tensor, Tensor, Tensor, Tensor)> {
    let outputs = yolo(cx, &images, &YoloConfig::yolox_tiny(CLASSES))?;
    let loss = yolox_loss(
        &outputs,
        &YoloAssignments {
            boxes,
            objectness,
            classes,
        },
        YoloLossConfig::default(),
    )?;
    Ok((loss.total, loss.boxes, loss.objectness, loss.classes))
}

#[test]
#[ignore = "requires a trusted PJRT plugin and compiles a complete YOLOX training step"]
fn pjrt_executes_complete_yolox_adam_step() -> Result<(), Box<dyn std::error::Error>> {
    let plugin = std::env::var("PJRT_PLUGIN_PATH")?;
    let client = unsafe { Client::load(plugin) }?;
    let definition = Model::new(training_model)
        .inputs((
            ModelInput::new([1, IMAGE, IMAGE, 3]),
            ModelInput::new([1, ANCHORS, 4]),
            ModelInput::new([1, ANCHORS, 1]),
            ModelInput::new([1, ANCHORS, CLASSES]),
        ))
        .training();
    let (trainable, mut model) = definition.trace_resident_all()?;
    let metrics = Tensor::stack(model.outputs(), 0)?;
    let loss = model.outputs()[0].clone();
    apply_model_adam(
        &mut model,
        &trainable,
        &loss,
        AdamOptions::default().learning_rate(1e-4),
    )?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let program = model.compile_stateful_tensors(&mut compiler, &[metrics])?;
    let mut session = model.session(&program).initialize_parameters(7)?.build()?;

    let images = client.upload_pinned(
        &[1, IMAGE, IMAGE, 3],
        vec![0.25_f32; (IMAGE * IMAGE * 3) as usize],
    )?;
    let mut boxes = vec![0.0_f32; (ANCHORS * 4) as usize];
    boxes[..4].copy_from_slice(&[4.0, 4.0, 8.0, 8.0]);
    let boxes = client.upload_pinned(&[1, ANCHORS, 4], boxes)?;
    let mut objectness = vec![0.0_f32; ANCHORS as usize];
    objectness[0] = 1.0;
    let objectness = client.upload_pinned(&[1, ANCHORS, 1], objectness)?;
    let mut classes = vec![0.0_f32; (ANCHORS * CLASSES) as usize];
    classes[0] = 1.0;
    let classes = client.upload_pinned(&[1, ANCHORS, CLASSES], classes)?;

    let outputs = session.run(&[
        images.buffer(),
        boxes.buffer(),
        objectness.buffer(),
        classes.buffer(),
    ])?;
    let losses = outputs[0].to_vec::<f32>()?;
    assert_eq!(losses.len(), 4);
    assert!(losses.iter().all(|value| value.is_finite()));
    Ok(())
}
