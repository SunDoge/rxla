use rxla_core::{
    Client, DType, Mesh, PartitionSpec, Runtime, Sharding, Storage, Tensor, TensorFunction,
    TensorLayout, Tracer,
};
use rxla_pjrt::{Shape, StridedLayout};

fn client() -> Client {
    // Loading a native plugin is unsafe because the selected shared library is
    // trusted code supplied by the test operator.
    unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn explicit_program_compiles_on_first_execution_and_reuses_cache() {
    let mut executor = Runtime::new(client()).unwrap();
    let program = executor
        .trace(|trace| {
            let x = trace.input(&[2])?;
            let one = trace.constant(&[2], &[1.0, 1.0])?;
            Ok(vec![x.add(&one)?])
        })
        .unwrap();
    assert_eq!(executor.stats().misses, 0);
    assert_eq!(
        program.run(&mut executor, &[&[2.0, 4.0]]).unwrap(),
        [[3.0, 5.0]]
    );
    assert_eq!(executor.stats().misses, 1);
    assert_eq!(
        program.run(&mut executor, &[&[3.0, 5.0]]).unwrap(),
        [[4.0, 6.0]]
    );
    assert_eq!(executor.stats().hits, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn runtime_dispatches_typed_buffers() {
    let program = Tracer::trace(|trace| {
        let x = trace.input_dtype(&[3], rxla_core::DType::I32)?;
        Ok(vec![x.wrapping_add_scalar(7)?])
    })
    .unwrap();
    let mut executor = Runtime::new(client()).unwrap();
    let input = executor.client().buffer(&[3], &[1, -2, i32::MAX]).unwrap();
    let outputs = program.run_buffers(&mut executor, &[&input]).unwrap();
    assert_eq!(outputs[0].to_vec::<i32>().unwrap(), [8, 5, i32::MIN + 6]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn runtime_results_are_materialized_lazy_tensor_leaves() {
    let tracer = Tracer::new();
    let input = tracer.input(&[2]).unwrap();
    let output = input.exp().unwrap();
    let program = tracer.program(vec![output]).unwrap();
    let bytes = [0.0f32, 1.0]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect::<Vec<_>>();
    let layout = StridedLayout::row_major(Shape::new(&[2]).unwrap(), 4).unwrap();
    let input = input
        .with_host_storage(Storage::host(DType::F32, bytes), layout)
        .unwrap();

    let mut executor = Runtime::new(client()).unwrap();
    let result = program
        .run_tensors(&mut executor, &[&input])
        .unwrap()
        .remove(0);
    assert!(result.is_materialized());
    assert!(matches!(result.layout(), TensorLayout::Pjrt(_)));
    assert_eq!(
        result.storage().unwrap().kind(),
        rxla_core::StorageKind::Pjrt
    );
    assert_eq!(
        result
            .to_buffer(executor.client())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [1.0, std::f32::consts::E]
    );
    assert!(result.add(&result).is_ok());

    // A materialized result can feed another independently traced program.
    let twice = Tracer::trace(|trace| {
        let x = trace.input(&[2])?;
        let two = trace.constant(&[2], &[2.0, 2.0])?;
        Ok(vec![x.mul(&two)?])
    })
    .unwrap();
    let result = twice
        .run_tensors(&mut executor, &[&result])
        .unwrap()
        .remove(0);
    assert_eq!(
        result
            .to_buffer(executor.client())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [2.0, 2.0 * std::f32::consts::E]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn eval_many_preserves_materialized_values_and_only_executes_lazy_roots() {
    let mut runtime = Runtime::new(client()).unwrap();
    assert!(runtime.eval_many(&[]).unwrap().is_empty());
    let empty: [Tensor; 0] = runtime.eval([] as [&Tensor; 0]).unwrap();
    assert!(empty.is_empty());

    let ready = Tensor::from_slice([2], DType::F32, [3.0, 4.0])
        .unwrap()
        .to_device(runtime.client())
        .unwrap();
    let input = Tensor::from_slice([2], DType::F32, [1.0, 2.0]).unwrap();
    let lazy = input.add_scalar(5.0).unwrap();

    let outputs = runtime.eval_many(&[ready.clone(), lazy]).unwrap();
    assert!(outputs[0].same_expression(&ready));
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [3.0, 4.0]);
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [6.0, 7.0]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn async_eval_publishes_only_after_wait_and_releases_dropped_claims() {
    let mut runtime = Runtime::new(client()).unwrap();
    assert!(
        runtime
            .eval_many_async(&[])
            .unwrap()
            .wait()
            .unwrap()
            .is_empty()
    );

    let ready = Tensor::from_slice([1], DType::F32, [9.0])
        .unwrap()
        .to_device(runtime.client())
        .unwrap();
    let input = Tensor::from_slice([2], DType::F32, [1.0, 2.0]).unwrap();
    let lazy = input.add_scalar(4.0).unwrap();
    let pending = runtime
        .eval_many_async(&[ready.clone(), lazy.clone()])
        .unwrap();
    assert!(!lazy.is_materialized());
    assert!(matches!(
        runtime.eval_many_async(std::slice::from_ref(&lazy)),
        Err(rxla_core::Error::EvaluationInFlight { index: 0 })
    ));
    assert!(matches!(
        runtime.eval_many(std::slice::from_ref(&lazy)),
        Err(rxla_core::Error::EvaluationInFlight { index: 0 })
    ));
    let outputs = pending.wait().unwrap();
    assert!(outputs[0].same_expression(&ready));
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [5.0, 6.0]);
    assert!(lazy.is_materialized());

    let retry = input.add_scalar(7.0).unwrap();
    drop(
        runtime
            .eval_many_async(std::slice::from_ref(&retry))
            .unwrap(),
    );
    assert!(!retry.is_materialized());
    let pending_retry = retry.eval_async(&mut runtime).unwrap();
    let retry_result = pending_retry.wait().unwrap();
    assert!(retry_result.same_expression(&retry));
    assert_eq!(retry.to_vec::<f32>().unwrap(), [8.0, 9.0]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn mlx_style_eval_materializes_lazy_tensor_operations_and_reuses_cache() {
    let x = rxla_core::Tensor::from_slice([2], DType::F32, [1.0, 2.0]).unwrap();
    let y = rxla_core::Tensor::from_slice([2], DType::F32, [3.0, 4.0]).unwrap();
    assert!(x.is_materialized());
    assert!(matches!(x.layout(), TensorLayout::Host(_)));

    let expression = x.add(&y).unwrap().exp().unwrap();
    let alias = expression.clone();
    assert!(!expression.is_materialized());
    let mut executor = Runtime::new(client()).unwrap();
    expression.eval(&mut executor).unwrap();
    assert!(expression.is_materialized());
    assert!(alias.is_materialized());
    let result = expression.clone();
    assert!(result.is_materialized());
    let values = result
        .to_buffer(executor.client())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert!((values[0] - 4.0f32.exp()).abs() < 1e-4);
    assert!((values[1] - 6.0f32.exp()).abs() < 1e-3);
    assert_eq!(executor.stats().misses, 1);

    let before = executor.stats();
    let same = result.eval(&mut executor).unwrap();
    assert!(same.is_materialized());
    assert_eq!(executor.stats(), before);

    let continued = expression.add(&x).unwrap().eval(&mut executor).unwrap();
    let values = continued
        .to_buffer(executor.client())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert!((values[0] - (4.0f32.exp() + 1.0)).abs() < 1e-4);
    assert!((values[1] - (6.0f32.exp() + 2.0)).abs() < 1e-3);

    x.add(&y)
        .unwrap()
        .exp()
        .unwrap()
        .eval(&mut executor)
        .unwrap();
    assert_eq!(executor.stats().hits, 1);

    let sum = x.add(&y).unwrap();
    let product = x.mul(&y).unwrap();
    let pair = executor.eval((&sum, &product)).unwrap();
    assert!(pair.0.is_materialized());
    assert!(pair.1.is_materialized());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn tensor_function_reuses_a_tensor_only_computation() -> Result<(), Box<dyn std::error::Error>> {
    let function = TensorFunction::new([2], DType::F32, |x| x.square()?.add_scalar(1.0))?;
    let first = Tensor::from_slice([2], DType::F32, [2.0, 3.0])?;
    let second = Tensor::from_slice([2], DType::F32, [4.0, 5.0])?;
    let mut runtime = Runtime::new(client())?;

    let first = function.call(&mut runtime, &first)?;
    assert_eq!(first.to_vec::<f32>()?, [5.0, 10.0]);
    assert_eq!(runtime.stats().misses, 1);

    let second = function.call(&mut runtime, &second)?;
    assert_eq!(second.to_vec::<f32>()?, [17.0, 26.0]);
    assert_eq!(runtime.stats().hits, 1);
    Ok(())
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn explicit_partition_constraint_executes_with_replicated_boundaries() {
    let tracer = Tracer::new();
    let mesh = Mesh::new([("data", 2)]).unwrap();
    let x = tracer
        .input(&[4])
        .unwrap()
        .with_sharding(Sharding::partitioned(
            mesh.clone(),
            PartitionSpec::new([Some("data")]).unwrap(),
        ))
        .unwrap();
    let program = tracer.program(vec![x]).unwrap();
    let mut runtime = Runtime::builder()
        .client(client())
        .auto_sharding(mesh, 64)
        .unwrap()
        .build()
        .unwrap();

    let plan = runtime.plan(&program).unwrap();
    assert_eq!(plan.required_devices(), 2);
    assert!(plan.requires_spmd_partitioning());

    let executable = program.compile(&mut runtime).unwrap();
    assert_eq!(executable.device_count(), 2);
    assert_eq!(runtime.stats().misses, 1);
    assert_eq!(
        program.run(&mut runtime, &[&[1.0, 2.0, 3.0, 4.0]]).unwrap(),
        [vec![1.0, 2.0, 3.0, 4.0]]
    );
}

#[test]
#[ignore = "requires two trusted PJRT devices"]
fn auto_spmd_executes_programs_and_lazy_partition_constraints() {
    let mesh = Mesh::new([("data", 2)]).unwrap();
    let mut runtime = Runtime::builder()
        .client(client())
        .auto_sharding(mesh.clone(), 64)
        .unwrap()
        .build()
        .unwrap();
    let program = Tracer::trace(|tracer| {
        let x = tracer.input(&[8])?;
        Ok(vec![x.exp()?])
    })
    .unwrap();

    let executable = program.compile(&mut runtime).unwrap();
    assert_eq!(executable.device_count(), 2);
    assert_eq!(runtime.stats().misses, 1);
    assert!(program.compile(&mut runtime).is_ok());
    assert_eq!(runtime.stats().hits, 1);

    let input =
        rxla_core::Tensor::from_slice([8], DType::F32, [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, -0.5])
            .unwrap();
    let output = program.run_tensors(&mut runtime, &[&input]).unwrap();
    let actual = output[0]
        .to_buffer(runtime.client())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let expected = [0.0f32, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, -0.5].map(f32::exp);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5);
    }
    let host = program.run(&mut runtime, &[&[0.0; 8]]).unwrap();
    assert_eq!(host, [vec![1.0; 8]]);

    let lazy = rxla_core::Tensor::from_slice([8], DType::F32, [0.25; 8])
        .unwrap()
        .exp()
        .unwrap()
        .with_sharding(Sharding::partitioned(
            mesh,
            PartitionSpec::new([Some("data")]).unwrap(),
        ))
        .unwrap();
    runtime.eval(&lazy).unwrap();
    assert!(lazy.is_materialized());
    let actual = lazy
        .to_buffer(runtime.client())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        actual
            .iter()
            .all(|value| (*value - 0.25f32.exp()).abs() < 1e-5)
    );
}

#[test]
#[ignore = "requires four trusted PJRT devices"]
fn multi_axis_partition_constraint_executes_on_a_device_mesh() {
    let mesh = Mesh::new([("data", 2), ("model", 2)]).unwrap();
    let mut runtime = Runtime::builder()
        .client(client())
        .auto_sharding(mesh.clone(), 128)
        .unwrap()
        .build()
        .unwrap();
    let tracer = Tracer::new();
    let x = tracer.input(&[4, 4]).unwrap();
    let y = x
        .exp()
        .unwrap()
        .with_sharding(Sharding::partitioned(
            mesh,
            PartitionSpec::new([Some("model"), Some("data")]).unwrap(),
        ))
        .unwrap();
    let program = tracer.program(vec![y]).unwrap();
    let values = (0..16).map(|value| value as f32 / 8.0).collect::<Vec<_>>();
    let outputs = program.run(&mut runtime, &[&values]).unwrap();
    assert_eq!(runtime.plan(&program).unwrap().required_devices(), 4);
    for (actual, input) in outputs[0].iter().zip(values) {
        assert!((actual - input.exp()).abs() < 1e-5);
    }
}
