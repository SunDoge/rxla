use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_index_conversion_rounding_and_nondifferentiable_origin() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let ids = graph.input_i32(&[7]).unwrap();
    let exe = graph.compile(&client, &ids.to_f32().unwrap()).unwrap();
    let values = [
        i32::MIN,
        -16_777_217,
        -1,
        0,
        16_777_216,
        16_777_217,
        i32::MAX,
    ];
    let input = client.buffer(&[7], &values).unwrap();
    assert_eq!(
        exe.execute(&[&input]).unwrap()[0].to_vec::<f32>().unwrap(),
        values.map(|v| v as f32)
    );
    let graph = Tracer::default();
    let x = graph.input(&[3]).unwrap();
    let converted = x.argmax(0, false).unwrap().to_f32().unwrap();
    let gradient = converted.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let exe = graph.compile_many(&client, &[converted, gradient]).unwrap();
    assert_eq!(
        exe.run_many(&[&[1., 3., 2.]]).unwrap(),
        [vec![1.], vec![0.; 3]]
    );
    let graph = Tracer::default();
    let empty = graph.input_i32(&[0, 2]).unwrap().to_f32().unwrap();
    let exe = graph.compile(&client, &empty).unwrap();
    assert!(
        exe.execute(&[&client.buffer::<i32>(&[0, 2], &[]).unwrap()])
            .unwrap()[0]
            .to_vec::<f32>()
            .unwrap()
            .is_empty()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_resident_position_drives_sine_and_frequency_gradient() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let position = graph.state_i32(&[]).unwrap();
    let frequency = graph.input(&[]).unwrap();
    let p = graph.read(&position).unwrap();
    let y = p.to_f32().unwrap().mul(&frequency).unwrap().sin().unwrap();
    let gradient = y.grad(&[frequency]).unwrap().remove(0);
    graph
        .write(&position, &p.wrapping_add_scalar(1).unwrap())
        .unwrap();
    let program = graph.compile(&mut compiler, &[y, gradient]).unwrap();
    let mut session = program
        .session(vec![(position.clone(), client.buffer(&[], &[0]).unwrap())])
        .unwrap();
    let input = client.buffer(&[], &[0.25]).unwrap();
    for p in 0..6 {
        let output = session.run(&[&input]).unwrap();
        let angle = p as f64 * 0.25;
        assert!((output[0].to_vec::<f32>().unwrap()[0] as f64 - angle.sin()).abs() < 1e-6);
        assert!(
            (output[1].to_vec::<f32>().unwrap()[0] as f64 - p as f64 * angle.cos()).abs() < 1e-6
        );
        assert_eq!(
            session.state(&position).unwrap().to_vec::<i32>().unwrap(),
            [p + 1]
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}
