use rxla_core::{Client, Tracer};
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a - b).abs() < 2e-5,
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn shape_errors() {
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    assert!(x.transpose(&[0, 0]).is_err());
    assert!(x.sum(&[2], false).is_err());
    assert!(x.mean(&[0, 0], true).is_err());
    assert!(x.broadcast_to(&[3]).is_err());
    assert!(x.broadcast_to(&[2, 4]).is_err());
    assert!(x.softmax(2).is_err());
    assert!(g.input(&[2, 0]).unwrap().mean(&[1], true).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_nn_ops() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    let values: &[f32] = &[1., 2., 3., 4., 5., 6.];
    let sum = x.sum(&[1], false).unwrap();
    close(
        &g.compile(&client, &sum).unwrap().run(&[values]).unwrap(),
        &[6., 15.],
    );
    let mean = x.mean(&[0], true).unwrap().broadcast_to(&[2, 3]).unwrap();
    close(
        &g.compile(&client, &mean).unwrap().run(&[values]).unwrap(),
        &[2.5, 3.5, 4.5, 2.5, 3.5, 4.5],
    );
    let transposed = x.transpose(&[1, 0]).unwrap();
    close(
        &g.compile(&client, &transposed)
            .unwrap()
            .run(&[values])
            .unwrap(),
        &[1., 4., 2., 5., 3., 6.],
    );
    let softmax = x.softmax(1).unwrap();
    close(
        &g.compile(&client, &softmax)
            .unwrap()
            .run(&[&[1000., 1001., 1002., -1000., -999., -998.]])
            .unwrap(),
        &[
            0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094,
        ],
    );
    let weight = g.constant(&[3], &[1., 1., 1.]).unwrap();
    let rms = x.rms_norm(&weight, 1e-5).unwrap();
    let expected: Vec<_> = values
        .chunks(3)
        .flat_map(|row| {
            let norm = (row.iter().map(|v| v * v).sum::<f32>() / 3. + 1e-5).sqrt();
            row.iter().map(move |v| v / norm)
        })
        .collect();
    close(
        &g.compile(&client, &rms).unwrap().run(&[values]).unwrap(),
        &expected,
    );
    let w = g.constant(&[3, 2], &[1., 0., 0., 1., 1., 1.]).unwrap();
    let b = g.constant(&[2], &[-5., 0.]).unwrap();
    let mlp = x
        .matmul(&w)
        .unwrap()
        .add(&b.broadcast_to(&[2, 2]).unwrap())
        .unwrap()
        .relu()
        .unwrap()
        .softmax(1)
        .unwrap();
    let output = g.compile(&client, &mlp).unwrap().run(&[values]).unwrap();
    let a = 1. / (1. + 5f32.exp());
    let b = 1. / (1. + 6f32.exp());
    close(&output, &[a, 1. - a, b, 1. - b]);
}
