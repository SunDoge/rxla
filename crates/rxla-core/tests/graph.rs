use rxla_core::{Client, Tracer};

#[test]
fn stablehlo_and_shape_validation() {
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    let w = g.input(&[3, 4]).unwrap();
    let y = x.matmul(&w).unwrap();
    assert_eq!(y.shape(), [2, 4]);
    assert!(x.add(&w).is_err());
    assert!(x.matmul(&x).is_err());
    assert!(x.reshape(&[5]).is_err());
    assert!(g.input(&[-1]).is_err());
    assert!(g.constant(&[2], &[1.]).is_err());
    assert!(x.add(&Tracer::default().input(&[2, 3]).unwrap()).is_err());
    let stablehlo = g.stablehlo(&y).unwrap();
    assert!(stablehlo.contains("stablehlo.dot_general"));
}

#[test]
#[ignore = "requires an explicitly trusted PJRT_PLUGIN_PATH; never silently skip runtime checks"]
fn real_cpu_execution_and_lifetimes() {
    let path = std::env::var("PJRT_PLUGIN_PATH").expect("set PJRT_PLUGIN_PATH");
    let client = unsafe { Client::load(path) }.unwrap();
    let graph = Tracer::default();
    let x = graph.input(&[2, 3]).unwrap();
    let w = graph.input(&[3, 2]).unwrap();
    let y = x.matmul(&w).unwrap();
    let executable = graph.compile(&client, &y).unwrap();
    let a = client.buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let b = client.buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]).unwrap();
    assert!(client.buffer(&[3], &[1.]).is_err());
    let wrong = client.buffer(&[6], &[1.; 6]).unwrap();
    assert!(executable.execute(&[&wrong, &b]).is_err());
    assert!(executable.execute(&[&a]).is_err());
    drop(client); // Buffers and executable keep the client/plugin alive.
    for _ in 0..10 {
        let output = executable.execute(&[&a, &b]).unwrap();
        assert_eq!(output[0].to_vec::<f32>().unwrap(), [22., 28., 49., 64.]);
    }
    assert_eq!(a.to_vec::<f32>().unwrap(), [1., 2., 3., 4., 5., 6.]); // No donation.
    let outputs = executable.execute(&[&a, &b]).unwrap();
    drop(executable);
    drop(a);
    drop(b);
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [22., 28., 49., 64.]);
}
