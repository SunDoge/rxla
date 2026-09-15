use rxla_core::{Client, Graph};

#[test]
fn diagonal_validates_axes_and_extreme_offsets() {
    let g = Graph::default();
    let x = g.input(&[3, 2, 4]).unwrap();
    assert_eq!(x.diagonal(0, 0, 2).unwrap().shape(), [2, 3]);
    assert_eq!(x.diagonal(-1, 0, 2).unwrap().shape(), [2, 2]);
    for offset in [i64::MIN, i64::MAX, -3, 4] {
        assert_eq!(x.diagonal(offset, 0, 2).unwrap().shape(), [2, 0]);
    }
    assert!(x.diagonal(0, 0, 0).is_err());
    assert!(x.trace(0, 0, 3).is_err());
    assert!(g.input(&[]).unwrap().diagonal(0, 0, 1).is_err());
    assert!(g.input(&[]).unwrap().diag_embed(0).is_err());
    assert!(x.diag_embed(i64::MIN).is_err());
    assert!(x.diag_embed(i64::MAX).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_diag_embed_offsets_nonfinite_and_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2, 3]).unwrap();
    let positive = x.diag_embed(1).unwrap();
    let negative = x.diag_embed(-2).unwrap();
    let main = x.diag_embed(0).unwrap();
    let loss = positive
        .mul(&positive)
        .unwrap()
        .sum(&[0, 1, 2], false)
        .unwrap();
    let dx = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let ddx = dx
        .sum(&[0, 1], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let exe = g
        .compile_many(&client, &[positive, negative, main, dx, ddx])
        .unwrap();
    for values in [
        [1., -2., 3., 4., 5., 6.],
        [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 4., 5., 6.],
    ] {
        let input = client.buffer(&[2, 3], &values).unwrap();
        let out = exe.execute(&[&input]).unwrap();
        for (output, offset) in out[..3].iter().zip([1i64, -2, 0]) {
            let size = 3 + offset.unsigned_abs() as usize;
            let actual = output.to_vec::<f32>().unwrap();
            for batch in 0..2 {
                for row in 0..size {
                    for col in 0..size {
                        let index = if offset >= 0 { row } else { col };
                        let expected = if col as i64 - row as i64 == offset && index < 3 {
                            values[batch * 3 + index]
                        } else {
                            0.
                        };
                        let value = actual[(batch * size + row) * size + col];
                        assert!(value == expected || value.is_nan() && expected.is_nan());
                    }
                }
            }
        }
        if values.iter().all(|v| v.is_finite()) {
            assert_eq!(out[3].to_vec::<f32>().unwrap(), values.map(|v| 2. * v));
            assert_eq!(out[4].to_vec::<f32>().unwrap(), [2.; 6]);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_empty_diag_embed() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2, 0]).unwrap();
    let y = x.diag_embed(-2).unwrap();
    assert_eq!(y.shape(), [2, 2, 2]);
    let empty = x.diag_embed(0).unwrap();
    assert_eq!(empty.shape(), [2, 0, 0]);
    let dx = y
        .sum(&[0, 1, 2], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g.compile_many(&client, &[y, empty, dx]).unwrap();
    let input = client.buffer::<f32>(&[2, 0], &[]).unwrap();
    let out = exe.execute(&[&input]).unwrap();
    assert_eq!(out[0].to_vec::<f32>().unwrap(), [0.; 8]);
    assert!(out[1].to_vec::<f32>().unwrap().is_empty());
    assert!(out[2].to_vec::<f32>().unwrap().is_empty());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_batched_offset_diagonals_trace_and_second_derivative() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[3, 2, 4]).unwrap();
    let positive = x.diagonal(1, 0, 2).unwrap();
    let loss = positive
        .mul(&positive)
        .unwrap()
        .sum(&[0, 1], false)
        .unwrap();
    let gradient = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let second = gradient
        .sum(&[0, 1, 2], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let outputs = [
        x.diagonal(0, 0, 2).unwrap(),
        positive,
        x.diagonal(-1, 0, 2).unwrap(),
        x.diagonal(1, 2, 0).unwrap(),
        x.trace(0, 0, 2).unwrap(),
        x.trace(i64::MIN, 0, 2).unwrap(),
        gradient,
        second,
    ];
    let exe = g.compile_many(&client, &outputs).unwrap();
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input = client.buffer(&[3, 2, 4], &values).unwrap();
    let out = exe.execute(&[&input]).unwrap();
    for (i, expected) in [
        vec![0., 9., 18., 4., 13., 22.],
        vec![1., 10., 19., 5., 14., 23.],
        vec![8., 17., 12., 21.],
        vec![8., 17., 12., 21.],
        vec![27., 39.],
        vec![0., 0.],
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(out[i].to_vec::<f32>().unwrap(), *expected);
    }
    let mut dx = vec![0.; 24];
    let mut ddx = vec![0.; 24];
    for i in [1, 10, 19, 5, 14, 23] {
        dx[i] = 2. * values[i];
        ddx[i] = 2.;
    }
    assert_eq!(out[6].to_vec::<f32>().unwrap(), dx);
    assert_eq!(out[7].to_vec::<f32>().unwrap(), ddx);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_empty_matrix_diagonal_and_trace() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2, 0, 3]).unwrap();
    let diagonal = x.diagonal(0, 1, 2).unwrap();
    assert_eq!(diagonal.shape(), [2, 0]);
    let trace = x.trace(0, 1, 2).unwrap();
    let gradient = trace
        .sum(&[0], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g
        .compile_many(&client, &[diagonal, trace, gradient])
        .unwrap();
    let input = client.buffer::<f32>(&[2, 0, 3], &[]).unwrap();
    let out = exe.execute(&[&input]).unwrap();
    assert!(out[0].to_vec::<f32>().unwrap().is_empty());
    assert_eq!(out[1].to_vec::<f32>().unwrap(), [0., 0.]);
    assert!(out[2].to_vec::<f32>().unwrap().is_empty());
}
