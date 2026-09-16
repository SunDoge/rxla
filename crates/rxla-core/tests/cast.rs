use rxla_core::{Client, DType, Tracer};

#[test]
fn floating_cast_is_explicit_and_shape_preserving() {
    let graph = Tracer::default();
    let input = graph.input(&[4]).unwrap();
    let half = input.cast(DType::F16).unwrap();
    assert_eq!(half.shape(), [4]);
    assert_eq!(half.dtype(), DType::F16);
    assert!(input.cast(DType::I32).is_err());
    assert_eq!(input.cast(DType::F32).unwrap().dtype(), DType::F32);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_f16_and_bf16_round_trips_execute() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let input = graph.input(&[5]).unwrap();
    let outputs =
        [DType::F16, DType::BF16].map(|dtype| input.cast(dtype).unwrap().cast(DType::F32).unwrap());
    let executable = graph.compile_many(&client, &outputs).unwrap();
    let actual = executable
        .run_many(&[&[0.0, 1.0, -2.5, 0.3333, 1000.25]])
        .unwrap();
    assert_eq!(actual[0], [0.0, 1.0, -2.5, 0.333_251_95, 1000.0]);
    assert_eq!(actual[1], [0.0, 1.0, -2.5, 0.333_984_38, 1000.0]);
}
