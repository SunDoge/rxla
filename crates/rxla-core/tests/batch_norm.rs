use rxla_core::{CacheLimits, Client, Compiler, Graph};

#[test]
fn inference_batch_norm_validation() {
    let g = Graph::default();
    let x = g.input(&[2, 3, 4]).unwrap();
    let p = g.input(&[3]).unwrap();
    assert_eq!(
        x.batch_norm_inference(1, &p, &p, &p, &p, 1e-5)
            .unwrap()
            .shape(),
        x.shape()
    );
    assert!(x.batch_norm_inference(3, &p, &p, &p, &p, 1e-5).is_err());
    for epsilon in [0., -1., f32::INFINITY, f32::NAN] {
        assert!(x.batch_norm_inference(1, &p, &p, &p, &p, epsilon).is_err());
    }
    let wrong = g.input(&[1, 3]).unwrap();
    let foreign = Graph::default().input(&[3]).unwrap();
    // Every parameter position must enforce both exact shape and graph ownership.
    for invalid in [&wrong, &foreign] {
        for position in 0..4 {
            let mut parameters = [&p; 4];
            parameters[position] = invalid;
            assert!(
                x.batch_norm_inference(
                    1,
                    parameters[0],
                    parameters[1],
                    parameters[2],
                    parameters[3],
                    1e-5
                )
                .is_err()
            );
        }
    }
    assert!(
        g.input(&[2, 0])
            .unwrap()
            .batch_norm_inference(1, &p, &p, &p, &p, 1e-5)
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_inference_batch_norm_uses_runtime_statistics_without_recompile() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client, CacheLimits::default());
    let shape = [2, 3, 2, 2];
    let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 2.).collect();
    for axis in [0, 1, 3] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let channels = shape[axis];
        let mean = g.input(&[channels]).unwrap();
        let variance = g.input(&[channels]).unwrap();
        let scale = g.input(&[channels]).unwrap();
        let bias = g.input(&[channels]).unwrap();
        let y = x
            .batch_norm_inference(axis, &mean, &variance, &scale, &bias, 1e-3)
            .unwrap();
        let executable = compiler.compile(&g, &y).unwrap();
        let variance: Vec<f32> = (0..channels).map(|c| c as f32).collect(); // includes zero variance
        let scale: Vec<f32> = (0..channels).map(|c| c as f32 - 1.).collect(); // negative and zero scales
        let bias: Vec<f32> = (0..channels).map(|c| c as f32 * 0.5).collect();
        let stride = shape[axis + 1..].iter().product::<i64>() as usize;
        for shift in [0., 5.] {
            let mean: Vec<f32> = (0..channels).map(|c| c as f32 + shift).collect();
            let actual = executable
                .run(&[&values, &mean, &variance, &scale, &bias])
                .unwrap();
            for (i, &a) in actual.iter().enumerate() {
                let c = i / stride % channels as usize;
                let expected = (values[i] as f64 - mean[c] as f64)
                    / (variance[c] as f64 + 1e-3f32 as f64).sqrt()
                    * scale[c] as f64
                    + bias[c] as f64;
                assert!(
                    a.is_finite() && (a as f64 - expected).abs() < 2e-5 * expected.abs().max(1.),
                    "axis {axis}, index {i}: {a} vs {expected}"
                );
            }
        }
    }
    // One compile per distinct channel-axis specialization, not per statistic value.
    assert_eq!(compiler.stats().misses, 3);
}
