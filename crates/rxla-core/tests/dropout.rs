use rxla_core::{CacheLimits, Client, Compiler, Graph};

#[test]
fn dropout_validates_probability_shape_and_owner() {
    let graph = Graph::default();
    let x = graph.input(&[2, 3]).unwrap();
    let mask = graph.input(&[2, 3]).unwrap();
    for probability in [0., -0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(x.dropout_with_mask(&mask, probability).is_err());
    }
    assert!(
        x.dropout_with_mask(&graph.input(&[3]).unwrap(), 0.5)
            .is_err()
    );
    assert!(
        x.dropout_with_mask(&Graph::default().input(&[2, 3]).unwrap(), 0.5)
            .is_err()
    );
    assert!(x.dropout_with_mask(&mask, 1.).is_ok());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_dropout_scalar_empty_and_select_mask_semantics() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let graph = Graph::default();
    let x = graph.input(&[]).unwrap();
    let mask = graph.input(&[]).unwrap();
    let empty = graph.input(&[0, 3]).unwrap();
    let outputs = [
        x.dropout_with_mask(&mask, 1.).unwrap(),
        x.dropout_with_mask(&mask, 0.25).unwrap(),
        empty.dropout_with_mask(&empty, 0.5).unwrap(),
    ];
    let executable = compiler.compile_many(&graph, &outputs).unwrap();
    let input = client.buffer(&[], &[3.]).unwrap();
    let empty = client.buffer::<f32>(&[0, 3], &[]).unwrap();
    for (value, keep) in [(0., false), (-0., false), (-2., true), (f32::NAN, true)] {
        let mask = client.buffer(&[], &[value]).unwrap();
        let outputs = executable.execute(&[&input, &mask, &empty]).unwrap();
        assert_eq!(
            outputs[0].to_vec::<f32>().unwrap(),
            [if keep { 3. } else { 0. }]
        );
        assert_eq!(
            outputs[1].to_vec::<f32>().unwrap(),
            [if keep { 12. } else { 0. }]
        );
        assert_eq!(outputs[2].dimensions().unwrap(), [0, 3]);
        assert!(outputs[2].to_vec::<f32>().unwrap().is_empty());
    }
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_dropout_runtime_masks_and_higher_order_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let graph = Graph::default();
    let x = graph.input(&[2, 3]).unwrap();
    let mask = graph.input(&[2, 3]).unwrap();
    let y = x.dropout_with_mask(&mask, 0.5).unwrap();
    let loss = y.mul(&y).unwrap().sum(&[0, 1], false).unwrap();
    let grads = loss.grad(&[x.clone(), mask.clone()]).unwrap();
    let second = grads[0]
        .sum(&[0, 1], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let forward_grad = y
        .sum(&[0, 1], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let executable = compiler
        .compile_many(
            &graph,
            &[y, grads[0].clone(), grads[1].clone(), second, forward_grad],
        )
        .unwrap();
    let values = [-3., -1., 0., 1., 2., 7.];
    let input = client.buffer(&[2, 3], &values).unwrap();
    for masks in [
        [0., 1., 0., 1., 1., 0.],
        [1.; 6],
        [0.; 6],
        [1., 0., 1., 0., 0., 1.],
    ] {
        let mask = client.buffer(&[2, 3], &masks).unwrap();
        let outputs = executable.execute(&[&input, &mask]).unwrap();
        let outputs: Vec<_> = outputs.iter().map(|b| b.to_vec::<f32>().unwrap()).collect();
        for i in 0..6 {
            assert_eq!(outputs[0][i], values[i] * masks[i] * 2.);
            assert_eq!(outputs[1][i], values[i] * masks[i] * 8.);
            assert_eq!(outputs[2][i], 0.);
            assert_eq!(outputs[3][i], masks[i] * 8.);
            assert_eq!(outputs[4][i], masks[i] * 2.);
        }
    }
    let nonfinite = client
        .buffer(
            &[2, 3],
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1., 2., 3.],
        )
        .unwrap();
    let mask = client.buffer(&[2, 3], &[0., 0., 0., 1., 1., 1.]).unwrap();
    let outputs = executable.execute(&[&nonfinite, &mask]).unwrap();
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        [0., 0., 0., 2., 4., 6.]
    );
    assert_eq!(
        outputs[1].to_vec::<f32>().unwrap(),
        [0., 0., 0., 8., 16., 24.]
    );
    assert_eq!(compiler.stats().misses, 1);
}
