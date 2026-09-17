use rxla_core::{CacheLimits, Client, Compiler, DType, Dim, Tracer};

#[test]
fn bounded_shape_survives_tensor_ir_and_elementwise_lowering() {
    let tracer = Tracer::new();
    let input = tracer
        .input_shape(&[Dim::Bounded { upper: 8 }, Dim::Static(4)], DType::F32)
        .unwrap();
    assert_eq!(input.static_dim(0), None);
    assert_eq!(input.dim_bound(0), Some(8));
    assert_eq!(input.static_dim(1), Some(4));
    assert_eq!(input.dim_bound(1), None);

    let output = input.add_scalar(1.0).unwrap();
    let traced = tracer.program(vec![output]).unwrap();
    let code = std::str::from_utf8(traced.lowered_program().code()).unwrap();
    assert!(code.contains("tensor<?x4xf32, #stablehlo.bounds<8, ?>>"));
    let spec = traced.input_spec(0).unwrap();
    assert_eq!(spec.shape, [-1, 4]);
    assert_eq!(spec.bound(0), Some(8));
    assert_eq!(spec.bound(1), None);

    let invalid = Tracer::new()
        .input_shape(&[Dim::Bounded { upper: 0 }], DType::F32)
        .err()
        .unwrap();
    assert!(matches!(
        invalid,
        rxla_core::Error::InvalidBoundedShape { .. }
    ));
}

#[test]
#[ignore = "requires a trusted PJRT_CPU_PLUGIN_PATH"]
fn xla_cpu_accepts_bounded_dynamic_stablehlo() {
    let tracer = Tracer::new();
    let input = tracer
        .input_shape(&[Dim::Bounded { upper: 8 }], DType::F32)
        .unwrap();
    assert_eq!(input.static_dim(0), None);
    assert_eq!(input.dim_bound(0), Some(8));
    let output = input.clone();
    let traced = tracer.program(vec![output]).unwrap();
    let program = traced.lowered_program();
    let code = std::str::from_utf8(program.code()).unwrap();
    assert!(code.contains("tensor<?xf32, #stablehlo.bounds<8>>"));
    let client = unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client, CacheLimits::default());
    compiler.compile_lowered(program).unwrap();
}
