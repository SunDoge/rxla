use rxla_core::{Client, Runtime, Tracer};

#[test]
#[ignore = "requires PJRT_CPU_PLUGIN_PATH and two addressable CPU devices"]
fn one_cpu_client_places_arrays_and_executables_on_each_device() {
    let client = unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut runtime = Runtime::builder().backend("cpu", client).build().unwrap();
    let devices = runtime.devices("cpu").unwrap();
    assert!(devices.len() >= 2);
    let first = devices[0].clone();
    let second = devices[1].clone();
    assert_ne!(first.id(), second.id());

    let program = Tracer::trace(|trace| {
        let input = trace.input(&[2])?;
        Ok(vec![input.add_scalar(1.0)?])
    })
    .unwrap();
    assert_eq!(
        runtime
            .on(&first)
            .unwrap()
            .run(&program, &[&[1.0, 2.0]])
            .unwrap(),
        [vec![2.0, 3.0]]
    );
    assert_eq!(
        runtime
            .on(&second)
            .unwrap()
            .run(&program, &[&[4.0, 5.0]])
            .unwrap(),
        [vec![5.0, 6.0]]
    );
    assert_eq!(runtime.on(&first).unwrap().stats().misses, 2);
}
