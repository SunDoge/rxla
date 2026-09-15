use rxla_core::{Client, ClientOptions, DType, Runtime, Tensor, Tracer};

#[test]
#[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH and PJRT_CUDA_PLUGIN_PATH"]
fn one_runtime_routes_independent_cpu_and_cuda_clients() {
    let cpu = unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").unwrap()) }.unwrap();
    let cuda_options = ClientOptions::new().set("preallocate", false);
    let cuda = unsafe {
        Client::load_with_options(
            std::env::var("PJRT_CUDA_PLUGIN_PATH").unwrap(),
            &cuda_options,
        )
    }
    .unwrap();
    let mut runtime = Runtime::builder()
        .backend("cpu", cpu.clone())
        .backend("cuda", cuda.clone())
        .default_backend("cuda")
        .build()
        .unwrap();

    assert_eq!(runtime.backend_names().count(), 2);
    let cpu_device = runtime.devices("cpu").unwrap()[0].clone();
    let cuda_device = runtime.devices("cuda").unwrap()[0].clone();
    assert_eq!(runtime.default_device(), &cuda_device);
    let program = Tracer::trace(|trace| {
        let input = trace.input(&[2])?;
        Ok(vec![input.mul_scalar(2.0)?])
    })
    .unwrap();

    assert_eq!(
        runtime
            .on(&cpu_device)
            .unwrap()
            .run(&program, &[&[1.0, 2.0]])
            .unwrap(),
        [vec![2.0, 4.0]]
    );
    assert_eq!(
        runtime
            .on(&cuda_device)
            .unwrap()
            .run(&program, &[&[3.0, 4.0]])
            .unwrap(),
        [vec![6.0, 8.0]]
    );
    assert_eq!(runtime.on(&cpu_device).unwrap().stats().misses, 1);
    assert_eq!(runtime.on(&cuda_device).unwrap().stats().misses, 1);

    // The ordinary path is Tensor-first: no public graph or tracing object is
    // required, even when two independent PJRT clients share one runtime.
    let cpu_value = Tensor::from_slice([2], DType::F32, [2.0, 3.0])
        .unwrap()
        .mul_scalar(4.0)
        .unwrap();
    let cpu_value = runtime.on(&cpu_device).unwrap().eval(&cpu_value).unwrap();
    assert_eq!(
        cpu_value.to_buffer(&cpu).unwrap().to_vec::<f32>().unwrap(),
        [8.0, 12.0]
    );

    let cuda_value = Tensor::from_slice([2], DType::F32, [5.0, 7.0])
        .unwrap()
        .add_scalar(1.0)
        .unwrap();
    let cuda_value = runtime.on(&cuda_device).unwrap().eval(&cuda_value).unwrap();
    assert_eq!(
        cuda_value
            .to_buffer(&cuda)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [6.0, 8.0]
    );

    let cpu_buffer = cpu.buffer(&[2], &[5.0, 6.0]).unwrap();
    assert!(
        runtime
            .on(&cuda_device)
            .unwrap()
            .run_buffers(&program, &[&cpu_buffer])
            .is_err()
    );
    let cuda_buffer = runtime
        .on(&cuda_device)
        .unwrap()
        .transfer(&cpu_buffer)
        .unwrap();
    let output = runtime
        .on(&cuda_device)
        .unwrap()
        .run_buffers(&program, &[&cuda_buffer])
        .unwrap();
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [10.0, 12.0]);
}
