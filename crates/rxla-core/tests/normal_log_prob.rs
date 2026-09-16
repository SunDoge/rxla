use rxla_core::{Client, Tracer};

#[test]
fn normal_density_requires_explicit_broadcast_and_same_graph() {
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    for bad in [
        g.input(&[3]).unwrap(),
        Tracer::default().input(&[2, 3]).unwrap(),
    ] {
        assert!(x.normal_log_prob(&bad, &x).is_err());
        assert!(x.normal_log_prob(&x, &bad).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_detaching_actions_selects_score_instead_of_pathwise_derivative() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let mean = g.input(&[3]).unwrap();
    let log_std = g.input(&[3]).unwrap();
    let eps = g.constant(&[3], &[-1.5, 0., 2.]).unwrap();
    let action = mean
        .add(&log_std.exp().unwrap().mul(&eps).unwrap())
        .unwrap();
    let pathwise = action
        .normal_log_prob(&mean, &log_std)
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(&[mean.clone(), log_std.clone()])
        .unwrap();
    let score = action
        .detach()
        .unwrap()
        .normal_log_prob(&mean, &log_std)
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(&[mean, log_std])
        .unwrap();
    let executable = g
        .compile_many(
            &client,
            &[
                pathwise[0].clone(),
                pathwise[1].clone(),
                score[0].clone(),
                score[1].clone(),
            ],
        )
        .unwrap();
    let inputs = [
        client.buffer(&[3], &[0.5, -1., 2.]).unwrap(),
        client.buffer(&[3], &[-0.7, 0., 0.4]).unwrap(),
    ];
    let outputs = executable.execute(&[&inputs[0], &inputs[1]]).unwrap();
    let expected = [
        vec![0.; 3],
        vec![-1.; 3],
        vec![-1.5 * 0.7f64.exp(), 0., 2. * (-0.4f64).exp()],
        vec![1.25, -1., 3.],
    ];
    for (actual, reference) in outputs.iter().zip(expected) {
        for (a, e) in actual.to_vec::<f32>().unwrap().iter().zip(reference) {
            assert!((*a as f64 - e).abs() < 2e-5, "{a} != {e}");
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_normal_density_and_two_derivatives_match_f64() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![], vec![2, 3], vec![0, 3]] {
        let n = shape.iter().product::<i64>() as usize;
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let mean = g.input(&shape).unwrap();
        let log_std = g.input(&shape).unwrap();
        let axes: Vec<_> = (0..shape.len()).collect();
        let log_prob = x.normal_log_prob(&mean, &log_std).unwrap();
        let loss = log_prob.sum(&axes, false).unwrap();
        let parameters = [x, mean, log_std];
        let gradient = loss.grad(&parameters).unwrap();
        let mut outputs = vec![log_prob];
        outputs.extend(gradient.iter().cloned());
        for i in 0..3 {
            outputs.push(
                gradient[i]
                    .sum(&axes, false)
                    .unwrap()
                    .grad(std::slice::from_ref(&parameters[i]))
                    .unwrap()
                    .remove(0),
            );
        }
        let executable = g.compile_many(&client, &outputs).unwrap();
        let values = [0.25f32, -1., 2., 0., 1e10, -3.];
        let means = [0.25f32, 0.5, -0.5, 0., -1e10, 2.];
        let scales = [0.4f32, -0.7, 0., 50., 20., 2.];
        let input = [
            client.buffer(&shape, &values[..n]).unwrap(),
            client.buffer(&shape, &means[..n]).unwrap(),
            client.buffer(&shape, &scales[..n]).unwrap(),
        ];
        let actual = executable
            .execute(&[&input[0], &input[1], &input[2]])
            .unwrap();
        for (output_index, buffer) in actual.iter().enumerate() {
            assert_eq!(buffer.dimensions().unwrap(), shape);
            for (i, a) in buffer.to_vec::<f32>().unwrap().into_iter().enumerate() {
                let r = values[i] as f64 - means[i] as f64;
                let s = scales[i] as f64;
                let inv2 = (-2. * s).exp();
                let q = r * r * inv2;
                let expected = [
                    -0.5 * q - s - 0.5 * std::f64::consts::TAU.ln(),
                    -r * inv2,
                    r * inv2,
                    q - 1.,
                    -inv2,
                    -inv2,
                    -2. * q,
                ][output_index];
                assert!(
                    a.is_finite() && (a as f64 - expected).abs() < 2e-5 + 2e-5 * expected.abs(),
                    "shape={shape:?} output={output_index} index={i}: {a} != {expected}"
                );
            }
        }
    }
}
