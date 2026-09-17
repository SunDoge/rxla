use rxla_core::{
    CacheLimits, Client, Compiler, Conv2dOptions, ConvTranspose2dOptions, DType, Dim,
    Pool2dOptions, Tensor, Tracer,
};

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
    assert_eq!(input.static_numel(), None);

    let output = input.add_scalar(1.0).unwrap();
    let count = input.numel().unwrap();
    let traced = tracer.program(vec![output, count]).unwrap();
    let code = std::str::from_utf8(traced.lowered_program().code()).unwrap();
    assert!(code.contains("tensor<?x4xf32, #stablehlo.bounds<8, ?>>"));
    assert_eq!(code.matches("stablehlo.get_dimension_size").count(), 2);
    assert!(code.contains("stablehlo.multiply"));
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
fn axis_transforms_keep_dynamic_bounds_attached_to_their_axes() {
    let tracer = Tracer::new();
    let input = tracer
        .input_shape(
            &[Dim::Bounded { upper: 8 }, Dim::Static(1), Dim::Static(4)],
            DType::F32,
        )
        .unwrap();

    let transposed = input.transpose(&[2, 0, 1]).unwrap();
    assert_eq!(transposed.shape(), [4, -1, 1]);
    assert_eq!(transposed.dim_bound(1), Some(8));

    let squeezed = transposed.squeeze(2).unwrap();
    assert_eq!(squeezed.shape(), [4, -1]);
    assert_eq!(squeezed.dim_bound(1), Some(8));
    let restored = squeezed.unsqueeze(0).unwrap();
    assert_eq!(restored.shape(), [1, 4, -1]);
    assert_eq!(restored.dim_bound(2), Some(8));
    let broadcast = squeezed.broadcast_in_dim(&[2, 4, -1], &[1, 2]).unwrap();
    assert_eq!(broadcast.shape(), [2, 4, -1]);
    assert_eq!(broadcast.dim_bound(2), Some(8));
    assert!(squeezed.broadcast_in_dim(&[-1, 4, -1], &[1, 2]).is_err());

    let reduced = restored.sum(&[1], false).unwrap();
    assert_eq!(reduced.shape(), [1, -1]);
    assert_eq!(reduced.dim_bound(1), Some(8));
    let keepdims = restored.sum(&[1], true).unwrap();
    assert_eq!(keepdims.shape(), [1, 1, -1]);
    assert_eq!(keepdims.dim_bound(2), Some(8));

    let indices = restored.argmax(1, true).unwrap();
    assert_eq!(indices.shape(), [1, 1, -1]);
    assert_eq!(indices.dim_bound(2), Some(8));

    let program = tracer
        .program(vec![
            transposed, squeezed, restored, broadcast, reduced, keepdims, indices,
        ])
        .unwrap();
    let code = std::str::from_utf8(program.lowered_program().code()).unwrap();
    assert!(code.contains("tensor<4x?x1xf32, #stablehlo.bounds<?, 8, ?>>"));
    assert!(code.contains("tensor<1x1x?xi32, #stablehlo.bounds<?, ?, 8>>"));
}

#[test]
fn concatenate_and_stack_infer_bounds_from_all_operands() {
    let tracer = Tracer::new();
    let dynamic = tracer
        .input_shape(&[Dim::Bounded { upper: 8 }, Dim::Static(4)], DType::F32)
        .unwrap();
    let fixed = tracer.input(&[2, 4]).unwrap();
    let concatenated = Tensor::concatenate(&[dynamic.clone(), fixed], 0).unwrap();
    assert_eq!(concatenated.shape(), [-1, 4]);
    assert_eq!(concatenated.dim_bound(0), Some(10));

    let stacked = Tensor::stack(&[dynamic.clone(), dynamic], 0).unwrap();
    assert_eq!(stacked.shape(), [2, -1, 4]);
    assert_eq!(stacked.dim_bound(1), Some(8));

    let incompatible = tracer
        .input_shape(&[Dim::Static(2), Dim::Bounded { upper: 5 }], DType::F32)
        .unwrap();
    assert!(Tensor::concatenate(&[concatenated.clone(), incompatible], 0).is_err());

    let program = tracer.program(vec![concatenated, stacked]).unwrap();
    let code = std::str::from_utf8(program.lowered_program().code()).unwrap();
    assert!(code.contains("tensor<?x4xf32, #stablehlo.bounds<10, ?>>"));
    assert!(code.contains("tensor<2x?x4xf32, #stablehlo.bounds<?, 8, ?>>"));
}

#[test]
fn vision_ops_preserve_a_bounded_batch_axis() {
    let tracer = Tracer::new();
    let input = tracer
        .input_shape(
            &[
                Dim::Bounded { upper: 16 },
                Dim::Static(8),
                Dim::Static(8),
                Dim::Static(3),
            ],
            DType::F32,
        )
        .unwrap();
    let kernel = tracer.input(&[3, 3, 3, 4]).unwrap();
    let convolution = input.conv2d(&kernel, Conv2dOptions::default()).unwrap();
    assert_eq!(convolution.shape(), [-1, 6, 6, 4]);
    assert_eq!(convolution.dim_bound(0), Some(16));

    let max_pool = input.max_pool2d(Pool2dOptions::default()).unwrap();
    let average_pool = input.avg_pool2d(Pool2dOptions::default(), false).unwrap();
    for output in [&max_pool, &average_pool] {
        assert_eq!(output.shape(), [-1, 4, 4, 3]);
        assert_eq!(output.dim_bound(0), Some(16));
    }

    let transpose_kernel = tracer.input(&[3, 3, 4, 3]).unwrap();
    let transposed = input
        .conv_transpose2d(&transpose_kernel, ConvTranspose2dOptions::default())
        .unwrap();
    assert_eq!(transposed.shape(), [-1, 10, 10, 4]);
    assert_eq!(transposed.dim_bound(0), Some(16));

    let dynamic_spatial = tracer
        .input_shape(
            &[
                Dim::Static(1),
                Dim::Bounded { upper: 8 },
                Dim::Static(8),
                Dim::Static(3),
            ],
            DType::F32,
        )
        .unwrap();
    assert!(
        dynamic_spatial
            .conv2d(&kernel, Conv2dOptions::default())
            .is_err()
    );
    assert!(
        dynamic_spatial
            .max_pool2d(Pool2dOptions::default())
            .is_err()
    );

    let program = tracer
        .program(vec![convolution, max_pool, average_pool, transposed])
        .unwrap();
    let code = std::str::from_utf8(program.lowered_program().code()).unwrap();
    assert!(code.contains("tensor<?x6x6x4xf32, #stablehlo.bounds<16, ?, ?, ?>>"));
    assert!(code.contains("tensor<?x4x4x3xf32, #stablehlo.bounds<16, ?, ?, ?>>"));
    assert!(code.contains("tensor<?x10x10x4xf32, #stablehlo.bounds<16, ?, ?, ?>>"));
}

#[test]
fn matmul_broadcasts_bounded_batch_axes_into_unbatched_weights() {
    let tracer = Tracer::new();
    let activations = tracer
        .input_shape(
            &[
                Dim::Bounded { upper: 16 },
                Dim::Bounded { upper: 8 },
                Dim::Static(32),
            ],
            DType::F32,
        )
        .unwrap();
    let weights = tracer.input(&[32, 64]).unwrap();
    let output = activations.matmul(&weights).unwrap();
    assert_eq!(output.shape(), [-1, -1, 64]);
    assert_eq!(output.dim_bound(0), Some(16));
    assert_eq!(output.dim_bound(1), Some(8));

    let batched_weights = tracer.input(&[1, 32, 64]).unwrap();
    let broadcast_output = activations.matmul(&batched_weights).unwrap();
    assert_eq!(broadcast_output.shape(), [-1, -1, 64]);
    assert_eq!(broadcast_output.dim_bound(0), Some(16));
    assert_eq!(broadcast_output.dim_bound(1), Some(8));

    let incompatible = tracer
        .input_shape(
            &[Dim::Bounded { upper: 12 }, Dim::Static(32), Dim::Static(64)],
            DType::F32,
        )
        .unwrap();
    assert!(activations.matmul(&incompatible).is_err());

    let program = tracer.program(vec![output, broadcast_output]).unwrap();
    let code = std::str::from_utf8(program.lowered_program().code()).unwrap();
    assert!(code.contains("tensor<?x?x64xf32, #stablehlo.bounds<16, 8, ?>>"));
    assert_eq!(code.matches("stablehlo.dot_general").count(), 2);
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
