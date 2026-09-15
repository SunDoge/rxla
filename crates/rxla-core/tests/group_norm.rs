use rxla_core::{Client, Graph};

#[test]
fn validates_group_norm_contract() {
    let graph = Graph::default();
    let x = graph.input(&[2, 4, 3]).unwrap();
    for groups in [-1, 0, 3, 5] {
        assert!(x.group_norm(groups, None, None, 1e-3).is_err());
    }
    for epsilon in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(x.group_norm(2, None, None, epsilon).is_err());
    }
    for shape in [vec![], vec![4], vec![2, 0], vec![2, 4, 0]] {
        assert!(
            graph
                .input(&shape)
                .unwrap()
                .group_norm(2, None, None, 1e-3)
                .is_err()
        );
    }
    for affine in [
        graph.input(&[1, 4]).unwrap(),
        Graph::default().input(&[4]).unwrap(),
    ] {
        assert!(x.group_norm(2, Some(&affine), None, 1e-3).is_err());
        assert!(x.group_norm(2, None, Some(&affine), 1e-3).is_err());
    }
    assert_eq!(
        graph
            .input(&[0, 4, 3])
            .unwrap()
            .group_norm(2, None, None, 1e-3)
            .unwrap()
            .shape(),
        [0, 4, 3]
    );
}

// Independent f64 coordinate reference and analytic weighted-loss derivative.
fn reference(
    x: &[f64],
    channels: usize,
    spatial: usize,
    groups: usize,
    gamma: &[f64],
    beta: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let width = channels / groups * spatial;
    let mut y = vec![0.; x.len()];
    let mut dx = y.clone();
    let mut dg = vec![0.; channels];
    let mut db = dg.clone();
    for (block, values) in x.chunks(width).enumerate() {
        let mean = values.iter().sum::<f64>() / width as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / width as f64;
        let inv = (var + 0.001f32 as f64).sqrt().recip();
        let mut sum_d = 0.;
        let mut sum_dz = 0.;
        for (j, &v) in values.iter().enumerate() {
            let i = block * width + j;
            let c = (i / spatial) % channels;
            let w = (i % 5 + 1) as f64;
            let z = (v - mean) * inv;
            y[i] = z * gamma[c] + beta[c];
            dg[c] += w * z;
            db[c] += w;
            sum_d += w * gamma[c];
            sum_dz += w * gamma[c] * z;
        }
        for (j, &v) in values.iter().enumerate() {
            let i = block * width + j;
            let c = (i / spatial) % channels;
            dx[i] = inv
                * ((i % 5 + 1) as f64 * gamma[c]
                    - sum_d / width as f64
                    - (v - mean) * inv * sum_dz / width as f64);
        }
    }
    (y, dx, dg, db)
}

fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!((a as f64 - e).abs() <= 3e-4 + 3e-4 * e.abs(), "{a} != {e}");
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_group_norm_values_affine_gradients_and_hessian_vector() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![2, 4], vec![2, 4, 3], vec![1, 4, 2, 3], vec![0, 4, 3]] {
        let n = shape.iter().product::<i64>() as usize;
        let spatial = shape[2..].iter().product::<i64>() as usize;
        for groups in [1, 2, 4] {
            let graph = Graph::default();
            let x = graph.input(&shape).unwrap();
            let gamma = graph.input(&[4]).unwrap();
            let beta = graph.input(&[4]).unwrap();
            let y = x
                .group_norm(groups, Some(&gamma), Some(&beta), 0.001)
                .unwrap();
            let weights: Vec<_> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
            let direction: Vec<_> = (0..n).map(|i| (i % 3) as f32 - 1.).collect();
            let axes: Vec<_> = (0..shape.len()).collect();
            let loss = y
                .mul(&graph.constant(&shape, &weights).unwrap())
                .unwrap()
                .sum(&axes, false)
                .unwrap();
            let grads = loss.grad(&[x.clone(), gamma, beta]).unwrap();
            let hvp = grads[0]
                .mul(&graph.constant(&shape, &direction).unwrap())
                .unwrap()
                .sum(&axes, false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            let values: Vec<_> = (0..n).map(|i| ((i * 7 % 17) as f32 - 8.) / 4.).collect();
            let g = [0.5, -1., 2., 0.25];
            let b = [0.1, 0.2, -0.3, 0.4];
            let xd: Vec<_> = values.iter().map(|&v| v as f64).collect();
            let gd = g.map(|v| v as f64);
            let bd = b.map(|v| v as f64);
            let (expected, dx, dg, db) = reference(&xd, 4, spatial, groups as usize, &gd, &bd);
            let delta = 1e-4;
            let shifted = |sign: f64| {
                xd.iter()
                    .zip(&direction)
                    .map(|(&v, &d)| v + sign * delta * d as f64)
                    .collect::<Vec<_>>()
            };
            let plus = reference(&shifted(1.), 4, spatial, groups as usize, &gd, &bd).1;
            let minus = reference(&shifted(-1.), 4, spatial, groups as usize, &gd, &bd).1;
            let second: Vec<_> = plus
                .iter()
                .zip(minus)
                .map(|(p, m)| (p - m) / (2. * delta))
                .collect();
            let mut outputs = vec![y];
            outputs.extend(grads);
            outputs.push(hvp);
            let result = graph
                .compile_many(&client, &outputs)
                .unwrap()
                .run_many(&[&values, &g, &b])
                .unwrap();
            for (actual, expected) in result.iter().zip([expected, dx, dg, db, second]) {
                close(actual, &expected);
            }
        }
    }
}
