use rxla_core::{Client, Tracer};

#[test]
fn training_batch_norm_validates_parameters_and_observation_count() {
    let g = Tracer::default();
    let x = g.input(&[2, 3, 2]).unwrap();
    let affine = g.input(&[2]).unwrap();
    for epsilon in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(x.batch_norm_training(2, &affine, &affine, epsilon).is_err());
    }
    assert!(x.batch_norm_training(3, &affine, &affine, 0.1).is_err());
    assert!(x.batch_norm_training(1, &affine, &affine, 0.1).is_err());
    let foreign = Tracer::default().input(&[2]).unwrap();
    assert!(x.batch_norm_training(2, &foreign, &affine, 0.1).is_err());
    assert!(
        g.input(&[0, 2])
            .unwrap()
            .batch_norm_training(1, &affine, &affine, 0.1)
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_training_batch_norm_statistics_and_all_gradients_match_f64() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axis in 0..3 {
        let g = Tracer::default();
        let shape = [2, 3, 2];
        let channels = shape[axis] as usize;
        let x = g.input(&shape).unwrap();
        let gamma = g.input(&[channels as i64]).unwrap();
        let beta = g.input(&[channels as i64]).unwrap();
        let seed = g.input(&shape).unwrap();
        let result = x.batch_norm_training(axis, &gamma, &beta, 0.125).unwrap();
        assert_eq!(result.mean.shape(), [channels as i64]);
        assert_eq!(result.variance.shape(), [channels as i64]);
        let gradients = result.output.vjp(&[x, gamma, beta], &seed).unwrap();
        let exe = g
            .compile_many(
                &client,
                &[
                    result.output,
                    result.mean,
                    result.variance,
                    gradients[0].clone(),
                    gradients[1].clone(),
                    gradients[2].clone(),
                ],
            )
            .unwrap();
        let gammas: Vec<_> = (0..channels).map(|i| i as f32 * 0.75 - 0.5).collect();
        let betas: Vec<_> = (0..channels).map(|i| i as f32 * 0.125).collect();
        let seeds: Vec<_> = (0..12).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
        let channel = |i: usize| [i / 6, i / 2 % 3, i % 2][axis];
        let count = (12 / channels) as f64;
        for values in [
            vec![2.; 12],
            (0..12).map(|i| (i % 7) as f32 * 0.5 - 1.).collect(),
        ] {
            let mut means = vec![0.0_f64; channels];
            for (i, value) in values.iter().enumerate() {
                means[channel(i)] += f64::from(*value) / count;
            }
            let mut variances = vec![0.0_f64; channels];
            let mut mean_seed = vec![0.0_f64; channels];
            let mut seed_center = vec![0.0_f64; channels];
            for i in 0..12 {
                let c = channel(i);
                variances[c] += (f64::from(values[i]) - means[c]).powi(2) / count;
                mean_seed[c] += f64::from(seeds[i]) / count;
                seed_center[c] += f64::from(seeds[i]) * (f64::from(values[i]) - means[c]) / count;
            }
            let mut output = vec![0.; 12];
            let mut dx = vec![0.; 12];
            let mut dg = vec![0.; channels];
            let mut db = vec![0.; channels];
            for i in 0..12 {
                let c = channel(i);
                let inv = 1. / (variances[c] + 0.125).sqrt();
                let centered = f64::from(values[i]) - means[c];
                output[i] = centered * inv * f64::from(gammas[c]) + f64::from(betas[c]);
                dx[i] = f64::from(gammas[c])
                    * inv
                    * (f64::from(seeds[i]) - mean_seed[c] - centered * inv * inv * seed_center[c]);
                dg[c] += f64::from(seeds[i]) * centered * inv;
                db[c] += f64::from(seeds[i]);
            }
            let actual = exe.run_many(&[&values, &gammas, &betas, &seeds]).unwrap();
            for (actual, expected) in actual.iter().zip([output, means, variances, dx, dg, db]) {
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!(
                        (f64::from(*actual) - expected).abs() < 1e-5,
                        "axis={axis} actual={actual} expected={expected}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_training_batch_norm_single_observation() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[2]).unwrap();
    let gamma = g.input(&[2]).unwrap();
    let beta = g.input(&[2]).unwrap();
    let result = x.batch_norm_training(0, &gamma, &beta, 0.125).unwrap();
    let gradients = result
        .output
        .sum(&[0], false)
        .unwrap()
        .grad(&[x, gamma, beta])
        .unwrap();
    let exe = g
        .compile_many(
            &client,
            &[
                result.output,
                result.mean,
                result.variance,
                gradients[0].clone(),
                gradients[1].clone(),
                gradients[2].clone(),
            ],
        )
        .unwrap();
    assert_eq!(
        exe.run_many(&[&[2., -3.], &[0.5, 2.], &[1., -1.]]).unwrap(),
        [
            vec![1., -1.],
            vec![2., -3.],
            vec![0., 0.],
            vec![0., 0.],
            vec![0., 0.],
            vec![1., 1.]
        ]
    );
}
