use rxla_core::{Client, Tracer};

#[test]
fn binary_cross_entropy_requires_exact_shape_and_graph() {
    let g = Tracer::default();
    let logits = g.input(&[2, 3]).unwrap();
    assert!(
        logits
            .binary_cross_entropy_with_logits(&g.input(&[3]).unwrap())
            .is_err()
    );
    assert!(
        logits
            .binary_cross_entropy_with_logits(&Tracer::default().input(&[2, 3]).unwrap())
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_binary_cross_entropy_loss_both_gradients_and_curvature() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[2, 6]).unwrap();
    let y = g.input(&[2, 6]).unwrap();
    let seed = g.input(&[2, 6]).unwrap();
    let loss = x.binary_cross_entropy_with_logits(&y).unwrap();
    let gradients = loss.vjp(&[x.clone(), y.clone()], &seed).unwrap();
    let second = gradients[0]
        .sum(&[0, 1], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let exe = g
        .compile_many(
            &client,
            &[loss, gradients[0].clone(), gradients[1].clone(), second],
        )
        .unwrap();
    let xv = [
        -1000., -80., -20., -3., -0., 0., 1., 3., 20., 80., 100., 1000.,
    ];
    let seeds = [1., 2., 3., -1., 0., 2., -2., 1., 3., -1., 2., 1.];
    for targets in [
        [0.; 12],
        [1.; 12],
        [0., 0.2, 0.5, 1., 0.3, 0.8, 0., 1., 0.2, 0.5, 0.7, 1.],
    ] {
        let actual = exe.run_many(&[&xv, &targets, &seeds]).unwrap();
        for i in 0..12 {
            let x = f64::from(xv[i]);
            let y = f64::from(targets[i]);
            let seed = f64::from(seeds[i]);
            let z = (-x.abs()).exp();
            let pos = if x >= 0. { 1. / (1. + z) } else { z / (1. + z) };
            let neg = if x <= 0. { 1. / (1. + z) } else { z / (1. + z) };
            let expected = [
                y * ((-x).max(0.) + z.ln_1p()) + (1. - y) * (x.max(0.) + z.ln_1p()),
                seed * ((1. - y) * pos - y * neg),
                -seed * x,
                seed * z / (1. + z).powi(2),
            ];
            for (output, expected) in expected.into_iter().enumerate() {
                let actual = f64::from(actual[output][i]);
                assert!(actual.is_finite());
                assert!(
                    (actual - expected).abs() <= 3e-5 * expected.abs() + 1e-37,
                    "output={output} x={x} y={y} actual={actual} expected={expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_binary_cross_entropy_scalar_empty_and_detached_targets() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![], vec![0, 3]] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let y = g.input(&shape).unwrap();
        let loss = x
            .binary_cross_entropy_with_logits(&y.detach().unwrap())
            .unwrap();
        let axes: Vec<_> = (0..shape.len()).collect();
        let gradients = loss.sum(&axes, false).unwrap().grad(&[x, y]).unwrap();
        let exe = g.compile_many(&client, &gradients).unwrap();
        let count = usize::from(shape.is_empty());
        assert_eq!(
            exe.run_many(&[&vec![0.; count], &vec![1.; count]]).unwrap(),
            [vec![-0.5; count], vec![0.; count]]
        );
    }
}
