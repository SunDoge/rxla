use rxla_core::{Client, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_sine_cosine_values_and_four_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for cosine in [false, true] {
        let graph = Tracer::default();
        let x = graph.input(&[9]).unwrap();
        let mut y = if cosine { x.cos() } else { x.sin() }.unwrap();
        let mut roots = vec![y.clone()];
        for _ in 0..4 {
            y = y
                .sum(&[0], false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            roots.push(y.clone());
        }
        let exe = graph.compile_many(&client, &roots).unwrap();
        for scale in [1., -0.5] {
            let values = [
                -1000_f32,
                -std::f32::consts::PI,
                -1.,
                -0.,
                0.,
                1e-6,
                1.,
                std::f32::consts::FRAC_PI_2,
                1000.,
            ]
            .map(|x| x * scale);
            let output = exe.run_many(&[&values]).unwrap();
            for (i, &x) in values.iter().enumerate() {
                let (s, c) = (x as f64).sin_cos();
                let derivatives = if cosine {
                    [c, -s, -c, s, c]
                } else {
                    [s, c, -s, -c, s]
                };
                for (order, expected) in derivatives.into_iter().enumerate() {
                    assert!(
                        (output[order][i] as f64 - expected).abs() < 2e-6,
                        "cosine={cosine} x={x} order={order}"
                    );
                }
            }
        }
    }
    for shape in [vec![], vec![0, 3]] {
        let graph = Tracer::default();
        let x = graph.input(&shape).unwrap();
        let exe = graph
            .compile_many(&client, &[x.sin().unwrap(), x.cos().unwrap()])
            .unwrap();
        let input = if shape.is_empty() { vec![0.] } else { vec![] };
        assert_eq!(
            exe.run_many(&[&input]).unwrap(),
            if shape.is_empty() {
                vec![vec![0.], vec![1.]]
            } else {
                vec![vec![], vec![]]
            }
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_runtime_positions_and_learnable_frequencies() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let positions = graph.input(&[3, 1]).unwrap();
    let frequencies = graph.input(&[1, 2]).unwrap();
    let angles = positions
        .broadcast_to(&[3, 2])
        .unwrap()
        .mul(&frequencies.broadcast_to(&[3, 2]).unwrap())
        .unwrap();
    let features = angles
        .sin()
        .unwrap()
        .add(&angles.cos().unwrap().mul_scalar(0.5).unwrap())
        .unwrap();
    let gradients = features
        .sum(&[0, 1], false)
        .unwrap()
        .grad(&[positions, frequencies])
        .unwrap();
    let exe = graph
        .compile_many(
            &client,
            &[features, gradients[0].clone(), gradients[1].clone()],
        )
        .unwrap();
    for p in [[0_f32, 1., 2.], [3., 4., 5.]] {
        let f = [0.125_f32, 0.5];
        let out = exe.run_many(&[&p, &f]).unwrap();
        let mut dp = [0_f64; 3];
        let mut df = [0_f64; 2];
        for i in 0..3 {
            for j in 0..2 {
                let a = p[i] as f64 * f[j] as f64;
                let expected = a.sin() + 0.5 * a.cos();
                assert!((out[0][i * 2 + j] as f64 - expected).abs() < 2e-6);
                let d = a.cos() - 0.5 * a.sin();
                dp[i] += d * f[j] as f64;
                df[j] += d * p[i] as f64;
            }
        }
        for (a, b) in out[1].iter().chain(&out[2]).zip(dp.into_iter().chain(df)) {
            assert!((*a as f64 - b).abs() < 2e-6);
        }
    }
}
