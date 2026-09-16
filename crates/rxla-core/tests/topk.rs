use rxla_core::{Client, Tracer};

#[test]
fn argsort_validates_axis_and_empty_shapes() {
    let g = Tracer::default();
    assert!(g.input(&[]).unwrap().argsort(0, false).is_err());
    assert!(g.input(&[2]).unwrap().argsort(1, true).is_err());
    for descending in [false, true] {
        let indices = g.input(&[2, 0]).unwrap().argsort(1, descending).unwrap();
        assert_eq!(indices.shape(), &[2, 0]);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_argsort_both_directions_reorder_payloads_and_empty_axes() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let rows = [
        [f32::NAN, -0., 3., 0., f32::NEG_INFINITY],
        [2., f32::NAN, 2., f32::INFINITY, -3.],
    ];
    for (shape, axis) in [([2, 5], 1), ([5, 2], 0), ([2, 0], 1)] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let asc = x.argsort(axis, false).unwrap();
        let desc = x.argsort(axis, true).unwrap();
        let mut outputs = vec![asc.clone(), desc];
        if shape[axis] != 0 {
            outputs.push(x.take_along_axis(&asc, axis).unwrap());
            outputs.push(x.topk(1, axis).unwrap().1);
            outputs.push(x.argmax(axis, false).unwrap());
        }
        let exe = g.compile_many(&client, &outputs).unwrap();
        let offset = |row, col| {
            if axis == 1 {
                row * 5 + col
            } else {
                col * 2 + row
            }
        };
        let count = shape.iter().product::<i64>() as usize;
        let mut values = vec![0.; count];
        if count != 0 {
            for (row, data) in rows.iter().enumerate() {
                for (col, &value) in data.iter().enumerate() {
                    values[offset(row, col)] = value;
                }
            }
        }
        let input = client.buffer(&shape, &values).unwrap();
        let out = exe.execute(&[&input]).unwrap();
        let reordered = if count != 0 {
            out[2].to_vec::<f32>().unwrap()
        } else {
            vec![]
        };
        for (direction, reference) in [
            [[4, 1, 3, 2, 0], [4, 0, 2, 3, 1]],
            [[2, 1, 3, 4, 0], [3, 0, 2, 4, 1]],
        ]
        .iter()
        .enumerate()
        {
            let actual = out[direction].to_vec::<i32>().unwrap();
            assert_eq!(out[direction].dimensions().unwrap(), shape);
            if count == 0 {
                assert!(actual.is_empty());
                continue;
            }
            for (row, ids) in reference.iter().enumerate() {
                for (col, &id) in ids.iter().enumerate() {
                    assert_eq!(actual[offset(row, col)], id);
                    if direction == 0 {
                        let selected = reordered[offset(row, col)];
                        let expected = rows[row][id as usize];
                        assert!(
                            selected.is_nan() && expected.is_nan()
                                || selected.to_bits() == expected.to_bits()
                        );
                    }
                }
            }
        }
        if count != 0 {
            assert_eq!(out[3].to_vec::<i32>().unwrap(), [2, 3]);
            assert_eq!(out[4].to_vec::<i32>().unwrap(), [0, 1]);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_argsort_gather_gradient_tracks_original_entries() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[5]).unwrap();
    let indices = x.argsort(0, false).unwrap();
    let values = x.take_along_axis(&indices, 0).unwrap();
    let loss = values
        .mul(&g.constant(&[5], &[1., 2., 3., 4., 5.]).unwrap())
        .unwrap()
        .sum(&[0], false)
        .unwrap();
    let gradient = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let exe = g.compile_many(&client, &[values, gradient]).unwrap();
    let input = client.buffer(&[5], &[3., 1., 2., 2., -1.]).unwrap();
    let out = exe.execute(&[&input]).unwrap();
    assert_eq!(out[0].to_vec::<f32>().unwrap(), [-1., 1., 2., 2., 3.]);
    assert_eq!(out[1].to_vec::<f32>().unwrap(), [5., 2., 3., 4., 1.]);
}

#[test]
fn validates_topk_and_keeps_ir_size_independent_of_k() {
    let g = Tracer::default();
    let x = g.input(&[2, 64]).unwrap();
    for (k, axis) in [(0, 2), (65, 1), (1, 2), (usize::MAX, 0)] {
        assert!(x.topk(k, axis).is_err());
    }
    assert!(g.input(&[]).unwrap().topk(1, 0).is_err());
    assert!(g.input(&[0]).unwrap().topk(1, 0).is_err());
    for shape in [[2, 64], [2, 0]] {
        let (values, indices) = g.input(&shape).unwrap().topk(0, 1).unwrap();
        assert_eq!(values.shape(), &[2, 0]);
        assert_eq!(indices.shape(), &[2, 0]);
    }
    let counts: Vec<_> = [1, 32, 64]
        .into_iter()
        .map(|k| {
            let g = Tracer::default();
            let x = g.input(&[2, 64]).unwrap();
            let (values, indices) = x.topk(k, 1).unwrap();
            let stablehlo = g.stablehlo_many(&[values, indices]).unwrap();
            assert_eq!(stablehlo.matches("stablehlo.sort").count(), 1);
            stablehlo.lines().count()
        })
        .collect();
    assert!(counts.iter().all(|&n| n == counts[0]));
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_zero_topk_has_empty_outputs_and_zero_derivatives_without_sorting() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [[2, 3], [0, 3], [2, 0]] {
        for axis in 0..2 {
            let g = Tracer::default();
            let x = g.input(&shape).unwrap();
            let (values, indices) = x.topk(0, axis).unwrap();
            let loss = values.mul(&values).unwrap().sum(&[0, 1], false).unwrap();
            let dx = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
            let ddx = dx
                .sum(&[0, 1], false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            let outputs = [values, indices, loss, dx, ddx];
            assert!(
                !g.stablehlo_many(&outputs)
                    .unwrap()
                    .contains("stablehlo.sort")
            );
            let exe = g.compile_many(&client, &outputs).unwrap();
            let count = shape.iter().product::<i64>() as usize;
            let data: Vec<_> = (0..count)
                .map(|i| [f32::NAN, f32::INFINITY, -1.][i % 3])
                .collect();
            let input = client.buffer(&shape, &data).unwrap();
            let out = exe.execute(&[&input]).unwrap();
            let mut empty_shape = shape;
            empty_shape[axis] = 0;
            assert_eq!(out[0].dimensions().unwrap(), empty_shape);
            assert_eq!(out[1].dimensions().unwrap(), empty_shape);
            assert!(out[0].to_vec::<f32>().unwrap().is_empty());
            assert!(out[1].to_vec::<i32>().unwrap().is_empty());
            assert_eq!(out[2].to_vec::<f32>().unwrap(), [0.]);
            for gradient in &out[3..] {
                assert_eq!(gradient.to_vec::<f32>().unwrap(), vec![0.; count]);
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_topk_stable_nonfinite_axes_and_empty_batches() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, axis) in [(vec![2, 7], 1), (vec![7, 2], 0), (vec![0, 7], 1)] {
        let rows = [
            [f32::NAN, 3., -0., 3., 0., f32::NEG_INFINITY, f32::INFINITY],
            [
                f32::NEG_INFINITY,
                f32::NAN,
                5.,
                5.,
                f32::NAN,
                -2.,
                f32::NEG_INFINITY,
            ],
        ];
        let count = shape.iter().product::<i64>() as usize;
        let mut data = vec![0.; count];
        for (row, values) in rows.iter().enumerate().take(count / 7) {
            for (col, &value) in values.iter().enumerate() {
                data[if axis == 1 {
                    row * 7 + col
                } else {
                    col * 2 + row
                }] = value;
            }
        }
        for k in [1, 4, 7] {
            let g = Tracer::default();
            let x = g.input(&shape).unwrap();
            let (values, indices) = x.topk(k, axis).unwrap();
            let exe = g.compile_many(&client, &[values, indices]).unwrap();
            let input = client.buffer(&shape, &data).unwrap();
            let out = exe.execute(&[&input]).unwrap();
            let actual = out[0].to_vec::<f32>().unwrap();
            let indices = out[1].to_vec::<i32>().unwrap();
            for (row, values) in rows.iter().enumerate().take(count / 7) {
                let mut order: Vec<usize> = (0..7).collect();
                order.sort_by(|&a, &b| match (values[a].is_nan(), values[b].is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => values[b].partial_cmp(&values[a]).unwrap(),
                });
                for (col, &original) in order.iter().take(k).enumerate() {
                    let offset = if axis == 1 {
                        row * k + col
                    } else {
                        col * 2 + row
                    };
                    assert_eq!(indices[offset], original as i32);
                    let expected = values[original];
                    assert!(
                        actual[offset].is_nan() && expected.is_nan()
                            || actual[offset].to_bits() == expected.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_topk_selected_gradients_and_second_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[5]).unwrap();
    let (values, _) = x.topk(2, 0).unwrap();
    let loss = values.mul(&values).unwrap().sum(&[0], false).unwrap();
    let dx = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let ddx = dx
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let exe = g.compile_many(&client, &[values, dx, ddx]).unwrap();
    let input = client.buffer(&[5], &[1., 3., 3., 2., -5.]).unwrap();
    let out = exe.execute(&[&input]).unwrap();
    assert_eq!(out[0].to_vec::<f32>().unwrap(), [3., 3.]);
    assert_eq!(out[1].to_vec::<f32>().unwrap(), [0., 6., 6., 0., 0.]);
    assert_eq!(out[2].to_vec::<f32>().unwrap(), [0., 2., 2., 0., 0.]);
}
