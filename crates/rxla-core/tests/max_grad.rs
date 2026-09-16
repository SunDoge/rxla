use rxla_core::{Client, Tracer};

fn client() -> Client {
    unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_max_grad_axes_keepdims_and_equal_winner_sharing() {
    let client = client();
    for axes in [vec![], vec![0], vec![1], vec![2], vec![2, 0], vec![0, 1, 2]] {
        for keepdims in [false, true] {
            let g = Tracer::default();
            let x = g.input(&[2, 3, 2]).unwrap();
            let y = x.max(&axes, keepdims).unwrap();
            let seed = g.input(y.shape()).unwrap();
            let dx = y.vjp(std::slice::from_ref(&x), &seed).unwrap().remove(0);
            let exe = g.compile_many(&client, &[y.clone(), dx]).unwrap();
            let count: usize = y.shape().iter().map(|&d| d as usize).product();
            let seeds: Vec<_> = (0..count).map(|i| i as f32 - 2.).collect();
            // Both executions use the same graph, with different winner/tie patterns.
            for values in [
                vec![1.; 12],
                vec![0., 2., 2., -1., 0., 2., 3., 2., 2., -1., 3., 2.],
            ] {
                let group = |i: usize| {
                    let coords = [i / 6, (i / 2) % 3, i % 2];
                    let mut group = 0;
                    for (axis, size) in [2, 3, 2].into_iter().enumerate() {
                        if !axes.contains(&axis) {
                            group = group * size + coords[axis];
                        }
                    }
                    group
                };
                let mut maxima = vec![f32::NEG_INFINITY; count];
                let mut ties = vec![0; count];
                for (i, &value) in values.iter().enumerate() {
                    maxima[group(i)] = maxima[group(i)].max(value);
                }
                for (i, &value) in values.iter().enumerate() {
                    if value == maxima[group(i)] {
                        ties[group(i)] += 1;
                    }
                }
                let reference: Vec<_> = values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let group = group(i);
                        if v == maxima[group] {
                            seeds[group] / ties[group] as f32
                        } else {
                            0.
                        }
                    })
                    .collect();
                let actual = exe.run_many(&[&values, &seeds]).unwrap();
                assert_eq!(actual[0], maxima);
                for (actual, expected) in actual[1].iter().zip(reference) {
                    assert!(
                        (actual - expected).abs() < 1e-6,
                        "axes={axes:?} keepdims={keepdims}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_max_grad_nonfinite_empty_and_higher_order() {
    let client = client();
    let g = Tracer::default();
    let x = g.input(&[4, 3]).unwrap();
    let y = x.max(&[1], false).unwrap();
    let seed = g.constant(&[4], &[6., 6., 6., 6.]).unwrap();
    let dx = y.vjp(&[x], &seed).unwrap().remove(0);
    let exe = g.compile_many(&client, &[y, dx]).unwrap();
    let actual = exe
        .run_many(&[&[
            1.,
            f32::NAN,
            2.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            1.,
            f32::INFINITY,
            -0.,
            0.,
            -1.,
        ]])
        .unwrap();
    // Forward reduction NaN behavior is native/backend-dependent; backward
    // must identify NaN inputs even when the native reduction ignores them.
    assert!(actual[1][..3].iter().all(|v| v.is_nan()));
    assert_eq!(actual[1][3..], [2., 2., 2., 3., 0., 3., 3., 3., 0.]);

    for (shape, axes) in [
        (vec![], vec![]),
        (vec![2, 0, 3], vec![1]),
        (vec![0, 3], vec![1]),
    ] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let y = x.max(&axes, true).unwrap();
        let count: usize = y.shape().iter().map(|&d| d as usize).product();
        let seed = g.input(y.shape()).unwrap();
        let dx = y.vjp(&[x], &seed).unwrap().remove(0);
        let exe = g.compile_many(&client, &[y, dx]).unwrap();
        let values = vec![2.; usize::from(shape.is_empty())];
        let actual = exe.run_many(&[&values, &vec![3.; count]]).unwrap();
        assert_eq!(actual[1], vec![3.; values.len()]);
        if !shape.is_empty() {
            assert!(actual[0].iter().all(|v| *v == f32::NEG_INFINITY));
        }
    }

    let g = Tracer::default();
    let x = g.input(&[3]).unwrap();
    let loss = x.max(&[0], false).unwrap().square().unwrap();
    let first = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let second = first
        .sum(&[0], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g.compile_many(&client, &[first, second]).unwrap();
    assert_eq!(
        exe.run_many(&[&[3., 3., 1.]]).unwrap(),
        [vec![3., 3., 0.], vec![1., 1., 0.]]
    );
    assert_eq!(
        exe.run_many(&[&[1., 3., 2.]]).unwrap(),
        [vec![0., 6., 0.], vec![0., 2., 0.]]
    );
}
