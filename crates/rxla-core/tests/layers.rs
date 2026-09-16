use rxla_core::{Client, Tracer};

#[test]
fn layer_shape_and_ownership_validation() {
    let g = Tracer::default();
    let x = g.input(&[2, 3, 4]).unwrap();
    let w = g.input(&[5, 4]).unwrap();
    let b = g.input(&[5]).unwrap();
    assert_eq!(x.linear(&w, Some(&b)).unwrap().shape(), [2, 3, 5]);
    assert_eq!(
        g.input(&[4]).unwrap().linear(&w, None).unwrap().shape(),
        [5]
    );
    assert!(g.input(&[]).unwrap().linear(&w, None).is_err());
    assert!(x.linear(&g.input(&[4, 5]).unwrap(), None).is_err());
    assert!(x.linear(&g.input(&[1, 5, 4]).unwrap(), None).is_err());
    assert!(x.linear(&w, Some(&g.input(&[1, 5]).unwrap())).is_err());
    let foreign = Tracer::default();
    assert!(x.linear(&foreign.input(&[5, 4]).unwrap(), None).is_err());
    assert!(x.linear(&w, Some(&foreign.input(&[5]).unwrap())).is_err());
    for shape in [vec![], vec![3], vec![2, 4], vec![1, 2, 3, 4], vec![0]] {
        assert!(x.layer_norm(&shape, None, None, 1e-5).is_err());
    }
    for epsilon in [0., -1., f32::INFINITY, f32::NAN] {
        assert!(x.layer_norm(&[4], None, None, epsilon).is_err());
    }
    assert!(
        x.layer_norm(&[3, 4], Some(&g.input(&[4]).unwrap()), None, 1e-5)
            .is_err()
    );
    assert!(
        x.layer_norm(&[4], None, Some(&foreign.input(&[4]).unwrap()), 1e-5)
            .is_err()
    );
    assert!(
        g.input(&[2, 0])
            .unwrap()
            .layer_norm(&[0], None, None, 1e-5)
            .is_err()
    );
    assert_eq!(
        x.layer_norm(&[3, 4], None, None, 1e-5).unwrap().shape(),
        x.shape()
    );
}

fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a as f64 - e).abs() <= 3e-5 * e.abs().max(1.),
            "{a} != {e}"
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_linear_rank_and_bias() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![3], vec![2, 3], vec![2, 2, 3]] {
        let g = Tracer::default();
        let rows = shape.iter().product::<i64>() as usize / 3;
        let values: Vec<f32> = (0..rows * 3).map(|i| i as f32 - 2.).collect();
        let x = g.input(&shape).unwrap();
        let weight = g.constant(&[2, 3], &[1., -2., 3., 4., 0., -1.]).unwrap();
        let bias = g.constant(&[2], &[0.5, -1.]).unwrap();
        let bare = x.linear(&weight, None).unwrap();
        let affine = x.linear(&weight, Some(&bias)).unwrap();
        let results = g
            .compile_many(&client, &[bare, affine])
            .unwrap()
            .run_many(&[&values])
            .unwrap();
        let expected: Vec<f64> = values
            .chunks(3)
            .flat_map(|v| {
                [
                    v[0] as f64 - 2. * v[1] as f64 + 3. * v[2] as f64,
                    4. * v[0] as f64 - v[2] as f64,
                ]
            })
            .collect();
        close(&results[0], &expected);
        close(
            &results[1],
            &expected
                .iter()
                .enumerate()
                .map(|(i, v)| v + if i % 2 == 0 { 0.5 } else { -1. })
                .collect::<Vec<_>>(),
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_layer_norm_matches_centered_f64_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, normalized, values) in [
        (
            vec![2, 3],
            vec![3],
            vec![10000., 10001., 10002., -4., -4., -4.],
        ),
        (
            vec![2, 2, 2],
            vec![2, 2],
            vec![1., 2., 3., 4., -2., 0., 2., 4.],
        ),
        (vec![3], vec![3], vec![2., 2., 2.]),
        (vec![2, 1], vec![1], vec![5., -7.]),
    ] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let count = normalized.iter().product::<i64>() as usize;
        let weights: Vec<f32> = (0..count).map(|i| 0.5 + i as f32).collect();
        let biases: Vec<f32> = (0..count).map(|i| i as f32 * 0.25 - 1.).collect();
        let w = g.constant(&normalized, &weights).unwrap();
        let b = g.constant(&normalized, &biases).unwrap();
        let plain = x.layer_norm(&normalized, None, None, 1e-5).unwrap();
        let weighted = x.layer_norm(&normalized, Some(&w), None, 1e-5).unwrap();
        let biased = x.layer_norm(&normalized, None, Some(&b), 1e-5).unwrap();
        let affine = x.layer_norm(&normalized, Some(&w), Some(&b), 1e-5).unwrap();
        let results = g
            .compile_many(&client, &[plain, weighted, biased, affine])
            .unwrap()
            .run_many(&[&values])
            .unwrap();
        let expected: Vec<f64> = values
            .chunks(count)
            .flat_map(|row| {
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / count as f64;
                let variance =
                    row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / count as f64;
                row.iter()
                    .map(move |&v| (v as f64 - mean) / (variance + 1e-5f32 as f64).sqrt())
            })
            .collect();
        for (variant, actual) in results.iter().enumerate() {
            let expected: Vec<_> = expected
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let v = if variant == 1 || variant == 3 {
                        v * weights[i % count] as f64
                    } else {
                        v
                    };
                    if variant >= 2 {
                        v + biases[i % count] as f64
                    } else {
                        v
                    }
                })
                .collect();
            close(actual, &expected);
        }
    }
}
