#![cfg(all(feature = "model", feature = "export"))]

use rxla_core::{Buffer, CacheLimits, Client, Compiler, Tensor};
use rxla_nn::{Cx, LoraLinear, Model, Result as NnResult, TensorApply};
use rxla_safetensors::{SafeTensors, save_parameter_selection_new};
use rxla_train::{AdamOptions, apply_model_adam, select_lora};
use std::collections::HashMap;

fn regression(cx: Cx) -> NnResult<Tensor> {
    let input = cx.input(&[4, 2])?;
    let target = cx.input(&[4, 1])?;
    let prediction =
        input.apply(&cx.named_layer("proj", LoraLinear::new(1, 1).alpha(1.0).bias(false))?)?;
    Ok(prediction.sub(&target)?.square()?.mean(&[0, 1], false)?)
}

#[test]
#[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
fn trains_exports_and_reloads_only_lora_parameters() {
    let client =
        unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path")) }
            .unwrap();
    let definition = Model::new(regression);
    let schema_trace = definition.trace().unwrap();
    let selection = select_lora(schema_trace.schema());
    assert_eq!(selection.len(), 2);

    let mut training = definition.trace_resident(&selection).unwrap();
    let loss = training.outputs()[0].clone();
    apply_model_adam(
        &mut training,
        &selection,
        &loss,
        AdamOptions {
            learning_rate: 0.05,
            ..AdamOptions::default()
        },
    )
    .unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compiled = training.compile_stateful(&mut compiler).unwrap();
    let mut session = compiled.initialize_session(42).unwrap();
    let input = client
        .buffer(&[4, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0])
        .unwrap();
    let target = client.buffer(&[4, 1], &[2.0, -3.0, -1.0, 7.0]).unwrap();
    let base = client.buffer(&[1, 2], &[0.0, 0.0]).unwrap();
    let mut initial_loss = None;
    let mut final_loss = f32::INFINITY;
    for _ in 0..250 {
        let value: Buffer = session
            .run_with_parameters((&input, &target), [("proj.weight", &base)])
            .unwrap();
        final_loss = value.to_vec::<f32>().unwrap()[0];
        initial_loss.get_or_insert(final_loss);
    }
    assert_eq!(initial_loss, Some(15.75));
    assert!(final_loss < 1e-3, "LoRA loss = {final_loss}");
    assert_eq!(base.to_vec::<f32>().unwrap(), [0.0, 0.0]);

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("adapter.safetensors");
    let metadata = HashMap::from([
        ("format".to_owned(), "rxla-lora-v1".to_owned()),
        ("rank".to_owned(), "1".to_owned()),
        ("alpha".to_owned(), "1".to_owned()),
    ]);
    save_parameter_selection_new(&path, &training, session.raw(), &selection, Some(&metadata))
        .unwrap();

    let mut checkpoint = SafeTensors::open(&path).unwrap();
    checkpoint
        .require_metadata(&[("format", "rxla-lora-v1"), ("rank", "1"), ("alpha", "1")])
        .unwrap();
    assert_eq!(
        checkpoint.names(),
        ["proj.lora_A.weight", "proj.lora_B.weight"]
    );

    let inference = definition.trace_resident(&selection).unwrap();
    let adapter = checkpoint
        .load_parameter_selection(&client, &selection)
        .unwrap();
    let compiled = inference.compile_stateful(&mut compiler).unwrap();
    let mut restored = adapter
        .initialize(compiled.session())
        .unwrap()
        .build()
        .unwrap();
    let restored_loss: Buffer = restored
        .run_with_parameters((&input, &target), [("proj.weight", &base)])
        .unwrap();
    let restored_loss = restored_loss.to_vec::<f32>().unwrap()[0];
    assert!(
        (restored_loss - final_loss).abs() < 1e-6,
        "trained={final_loss}, restored={restored_loss}"
    );
}
