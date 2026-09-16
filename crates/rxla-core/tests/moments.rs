use rxla_core::{Client, Tracer};

#[test]
fn moments_validates_axes_and_degrees_of_freedom() {
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    for axes in [vec![2], vec![1, 1]] {
        assert!(x.moments(&axes, false, 0).is_err());
    }
    for correction in [3, usize::MAX] {
        assert!(x.variance(&[1], false, correction).is_err());
    }
    assert!(x.variance(&[], false, 1).is_err());
    assert!(g.input(&[2, 0]).unwrap().variance(&[1], false, 0).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_moments_values_and_joint_gradient_match_f64() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axes in [vec![0], vec![1], vec![1, 0]] {
        for keepdims in [false, true] {
            for correction in [0, 1] {
                let g = Tracer::default();
                let x = g.input(&[2, 3]).unwrap();
                let (mean, variance) = x.moments(&axes, keepdims, correction).unwrap();
                assert_eq!(mean.shape(), variance.shape());
                let weight = g.input(mean.shape()).unwrap();
                let loss = mean
                    .add(&variance)
                    .unwrap()
                    .mul(&weight)
                    .unwrap()
                    .sum(&(0..mean.shape().len()).collect::<Vec<_>>(), false)
                    .unwrap();
                let gradient = loss.grad(&[x]).unwrap().remove(0);
                let count: usize = mean.shape().iter().map(|&v| v as usize).product();
                let exe = g
                    .compile_many(&client, &[mean, variance, gradient])
                    .unwrap();
                let weights: Vec<_> = (0..count).map(|i| i as f32 - 0.5).collect();
                let group = |i: usize| {
                    let mut group = 0;
                    if !axes.contains(&0) {
                        group = i / 3;
                    }
                    if !axes.contains(&1) {
                        group = group * 3 + i % 3;
                    }
                    group
                };
                for values in [
                    [-2., 0., 2., 1., 3., 5.],
                    [10000., 10003., 10006., 10006., 10009., 10012.],
                    [2.; 6],
                ] {
                    let mut means = vec![0.0_f64; count];
                    let mut counts = vec![0usize; count];
                    for (i, value) in values.iter().enumerate() {
                        means[group(i)] += f64::from(*value);
                        counts[group(i)] += 1;
                    }
                    for i in 0..count {
                        means[i] /= counts[i] as f64;
                    }
                    let mut variances = vec![0.0_f64; count];
                    for (i, value) in values.iter().enumerate() {
                        variances[group(i)] += (f64::from(*value) - means[group(i)]).powi(2);
                    }
                    for i in 0..count {
                        variances[i] /= (counts[i] - correction) as f64;
                    }
                    let actual = exe.run_many(&[&values, &weights]).unwrap();
                    for i in 0..count {
                        assert!((f64::from(actual[0][i]) - means[i]).abs() < 1e-5);
                        assert!((f64::from(actual[1][i]) - variances[i]).abs() < 1e-5);
                    }
                    for i in 0..6 {
                        let group = group(i);
                        let expected = f64::from(weights[group])
                            * (1. / counts[group] as f64
                                + 2. * (f64::from(values[i]) - means[group])
                                    / (counts[group] - correction) as f64);
                        assert!((f64::from(actual[2][i]) - expected).abs() < 1e-5);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_moments_scalar_identity_and_empty_output() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[]).unwrap();
    let (mean, variance) = x.moments(&[], true, 0).unwrap();
    let gradient = variance.grad(&[x]).unwrap().remove(0);
    let exe = g
        .compile_many(&client, &[mean, variance, gradient])
        .unwrap();
    assert_eq!(
        exe.run_many(&[&[3.]]).unwrap(),
        [vec![3.], vec![0.], vec![0.]]
    );
    let g = Tracer::default();
    let x = g.input(&[0, 3]).unwrap();
    let (mean, variance) = x.moments(&[1], false, 1).unwrap();
    let gradient = variance
        .sum(&[0], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g
        .compile_many(&client, &[mean, variance, gradient])
        .unwrap();
    assert_eq!(
        exe.run_many(&[&[]]).unwrap(),
        [Vec::<f32>::new(), vec![], vec![]]
    );
}
