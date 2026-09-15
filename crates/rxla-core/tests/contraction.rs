use rxla_core::{Client, Graph};

#[test]
fn validates_axes() {
    let g = Graph::default();
    let x = g.input(&[2, 3, 4]).unwrap();
    let w = g.input(&[4, 2, 5]).unwrap();
    assert_eq!(
        x.dot_general(&w, &[2], &[0], &[0], &[1]).unwrap().shape(),
        [2, 3, 5]
    );
    for (lc, rc, lb, rb) in [
        (vec![2], vec![], vec![], vec![]),
        (vec![2], vec![0], vec![2], vec![1]),
        (vec![3], vec![0], vec![], vec![]),
        (vec![2, 2], vec![0, 1], vec![], vec![]),
        (vec![2], vec![0], vec![1], vec![1]),
    ] {
        assert!(x.dot_general(&w, &lc, &rc, &lb, &rb).is_err());
    }
    assert!(
        x.tensordot(&Graph::default().input(&[4]).unwrap(), &[2], &[0])
            .is_err()
    );
}

fn close(a: &[f32], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            (f64::from(a) - b).abs() < 2e-4 * (1. + b.abs()),
            "{a} != {b}"
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_batched_contraction_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2, 3, 4]).unwrap();
    let w = g.input(&[4, 2, 5]).unwrap();
    let y = x.dot_general(&w, &[2], &[0], &[0], &[1]).unwrap();
    let loss = y.mul(&y).unwrap().sum(&[0, 1, 2], false).unwrap();
    let grad = loss.grad(&[x.clone(), w]).unwrap();
    let second = grad[0]
        .sum(&[0, 1, 2], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g
        .compile_many(&client, &[y, grad[0].clone(), grad[1].clone(), second])
        .unwrap();
    let xv: Vec<_> = (0..24).map(|i| (i as f32 - 8.) * 0.125).collect();
    let wv: Vec<_> = (0..40).map(|i| (i as f32 - 17.) * 0.0625).collect();
    let (mut y, mut dx, mut dw, mut ddx) = (vec![0.; 30], vec![0.; 24], vec![0.; 40], vec![0.; 24]);
    for b in 0..2 {
        for i in 0..3 {
            for j in 0..5 {
                let yi = (b * 3 + i) * 5 + j;
                for k in 0..4 {
                    y[yi] +=
                        f64::from(xv[(b * 3 + i) * 4 + k]) * f64::from(wv[(k * 2 + b) * 5 + j]);
                }
                let ws: f64 = (0..4).map(|k| f64::from(wv[(k * 2 + b) * 5 + j])).sum();
                for k in 0..4 {
                    let xi = (b * 3 + i) * 4 + k;
                    let wi = (k * 2 + b) * 5 + j;
                    dx[xi] += 2. * y[yi] * f64::from(wv[wi]);
                    dw[wi] += 2. * y[yi] * f64::from(xv[xi]);
                    ddx[xi] += 2. * f64::from(wv[wi]) * ws;
                }
            }
        }
    }
    let xb = client.buffer(&[2, 3, 4], &xv).unwrap();
    let wb = client.buffer(&[4, 2, 5], &wv).unwrap();
    let out = exe.execute(&[&xb, &wb]).unwrap();
    for (a, b) in out.iter().zip([y, dx, dw, ddx]) {
        close(&a.to_vec::<f32>().unwrap(), &b);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_outer_scalar_multiple_axes_and_empty() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let s = g.constant(&[], &[2.]).unwrap();
    let v = g.constant(&[2], &[3., 4.]).unwrap();
    let outer = v.tensordot(&v, &[], &[]).unwrap();
    let scaled = s.tensordot(&v, &[], &[]).unwrap();
    let x = g.constant(&[2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let w = g.constant(&[3, 2], &[1., 4., 2., 5., 3., 6.]).unwrap();
    let full = x.tensordot(&w, &[1, 0], &[0, 1]).unwrap();
    let a = g.input(&[2, 0]).unwrap();
    let b = g.input(&[0, 3]).unwrap();
    let zero = a.tensordot(&b, &[1], &[0]).unwrap();
    let exe = g
        .compile_many(&client, &[outer, scaled, full, zero])
        .unwrap();
    let a = client.buffer::<f32>(&[2, 0], &[]).unwrap();
    let b = client.buffer::<f32>(&[0, 3], &[]).unwrap();
    let out = exe.execute(&[&a, &b]).unwrap();
    for (a, b) in out.iter().zip([
        vec![9., 12., 12., 16.],
        vec![6., 8.],
        vec![91.],
        vec![0.; 6],
    ]) {
        close(&a.to_vec::<f32>().unwrap(), &b);
    }
}
