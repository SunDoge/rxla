#![cfg(feature = "model")]

use rxla_core::{CacheLimits, Client, Compiler, Tensor};
use rxla_nn::{Cx, Linear, Model, ModelInput, Result as NnResult, TensorApply};
use rxla_safetensors::{Dtype, SafeTensors};
use safetensors::tensor::{TensorView, serialize};
use std::io::Cursor;

fn linear(cx: Cx, input: Tensor) -> NnResult<Tensor> {
    input.apply(&cx.layer("head", Linear::new(2).bias(false))?)
}

#[test]
#[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
fn safetensors_initializes_and_exports_a_resident_model_session() {
    let values = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let checkpoint = serialize(
        [(
            "head.weight",
            TensorView::new(Dtype::F32, vec![2, 3], &bytes).unwrap(),
        )],
        None,
    )
    .unwrap();
    let mut checkpoint = SafeTensors::new(Cursor::new(checkpoint)).unwrap();
    let client =
        unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path")) }
            .unwrap();
    let definition = Model::new(linear).inputs(ModelInput::new([1, 3]));
    let (_, model) = definition.trace_resident_all().unwrap();
    let schema = model.schema().clone();
    let weights = checkpoint.load_parameter_schema(&client, &schema).unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compiled = model.compile_stateful(&mut compiler).unwrap();
    let mut session = weights
        .initialize(compiled.session())
        .unwrap()
        .build()
        .unwrap();

    let input = client.buffer(&[1, 3], &[2.0, 3.0, 4.0]).unwrap();
    let output = session.run::<_, rxla_core::Buffer>(&input).unwrap();
    assert_eq!(output.to_vec::<f32>().unwrap(), [2.0, 3.0]);

    let current = model.resident_parameter_buffers(session.raw()).unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].0, "head.weight");
    assert_eq!(current[0].1.to_vec::<f32>().unwrap(), values);
}
