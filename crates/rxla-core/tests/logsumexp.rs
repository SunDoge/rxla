use rxla_core::{Client, Graph};

fn client() -> Client {
    unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap()
}

#[test]
fn reduction_axes_are_validated() {
    let g = Graph::default();
    let x = g.input(&[2, 3]).unwrap();
    for axes in [vec![2], vec![0, 0]] {
        assert!(x.logsumexp(&axes, false).is_err());
        assert!(x.min(&axes, false).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_logsumexp_values_and_vjp_against_f64() {
    let client = client();
    for axes in [vec![], vec![0], vec![1], vec![1, 0]] {
        for keepdims in [false, true] {
            let g = Graph::default();
            let x = g.input(&[2, 3]).unwrap();
            let y = x.logsumexp(&axes, keepdims).unwrap();
            let seed = g.input(y.shape()).unwrap();
            let dx = y.vjp(&[x], &seed).unwrap().remove(0);
            let count: usize = y.shape().iter().map(|&d| d as usize).product();
            let exe = g.compile_many(&client, &[y, dx]).unwrap();
            let seed: Vec<_> = (0..count).map(|i| i as f32 - 0.5).collect();
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
                [0.; 6],
                [1000., 999., 998., -1000., -999., -998.],
                [1., -2., 0.5, 3., 2., -1.],
            ] {
                let mut maximum = vec![f64::NEG_INFINITY; count];
                for (i, value) in values.iter().enumerate() {
                    maximum[group(i)] = maximum[group(i)].max(f64::from(*value));
                }
                let mut sums = vec![0.; count];
                for (i, value) in values.iter().enumerate() {
                    sums[group(i)] += (f64::from(*value) - maximum[group(i)]).exp();
                }
                let actual = exe.run_many(&[&values, &seed]).unwrap();
                for i in 0..count {
                    let expected = maximum[i] + sums[i].ln();
                    assert!((f64::from(actual[0][i]) - expected).abs() < 5e-5);
                }
                for i in 0..6 {
                    let group = group(i);
                    let expected = f64::from(seed[group])
                        * (f64::from(values[i]) - maximum[group]).exp()
                        / sums[group];
                    assert!((f64::from(actual[1][i]) - expected).abs() < 1e-6);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_logsumexp_nonfinite_empty_and_second_derivative() {
    let client = client();
    let g = Graph::default();
    let x = g.input(&[4, 3]).unwrap();
    let y = x.logsumexp(&[1], false).unwrap();
    let gradient = y.sum(&[0], false).unwrap().grad(&[x]).unwrap().remove(0);
    let exe = g.compile_many(&client, &[y, gradient]).unwrap();
    let actual = exe
        .run_many(&[&[
            0.,
            f32::NEG_INFINITY,
            0.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            1.,
            f32::INFINITY,
            2.,
            1.,
            f32::NAN,
            2.,
        ]])
        .unwrap();
    assert!((actual[0][0] - 2.0_f32.ln()).abs() < 1e-6);
    assert_eq!(actual[0][1], f32::NEG_INFINITY);
    assert_eq!(actual[0][2], f32::INFINITY);
    assert!(actual[0][3].is_nan());
    assert_eq!(actual[1][..3], [0.5, 0., 0.5]);

    for shape in [vec![2, 0], vec![0, 3]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let y = x.logsumexp(&[1], false).unwrap();
        let gradient = y.sum(&[0], false).unwrap().grad(&[x]).unwrap().remove(0);
        let exe = g.compile_many(&client, &[y, gradient]).unwrap();
        let actual = exe.run_many(&[&[]]).unwrap();
        assert_eq!(actual[0], vec![f32::NEG_INFINITY; shape[0] as usize]);
        assert!(actual[1].is_empty());
    }
    let g = Graph::default();
    let x = g.input(&[3]).unwrap();
    let seed = g.constant(&[3], &[1., -2., 3.]).unwrap();
    let first = x
        .logsumexp(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let second = first
        .mul(&seed)
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g.compile_many(&client, &[second]).unwrap();
    // At equal logits, H*v = (v - mean(v))/3.
    let actual = exe.run_many(&[&[2., 2., 2.]]).unwrap();
    for (actual, expected) in actual[0].iter().zip([1. / 9., -8. / 9., 7. / 9.]) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_min_reduction_ties_and_empty() {
    let client = client();
    let g = Graph::default();
    let x = g.input(&[2, 3]).unwrap();
    let y = x.min(&[1], true).unwrap();
    let gradient = y.sum(&[0, 1], false).unwrap().grad(&[x]).unwrap().remove(0);
    let exe = g.compile_many(&client, &[y, gradient]).unwrap();
    assert_eq!(
        exe.run_many(&[&[1., -2., -2., 0., -0., 3.]]).unwrap(),
        [vec![-2., 0.], vec![0., 0.5, 0.5, 0.5, 0.5, 0.]]
    );
    let g = Graph::default();
    let x = g.input(&[2, 0]).unwrap();
    let y = x.min(&[1], false).unwrap();
    let gradient = y.sum(&[0], false).unwrap().grad(&[x]).unwrap().remove(0);
    let exe = g.compile_many(&client, &[y, gradient]).unwrap();
    assert_eq!(
        exe.run_many(&[&[]]).unwrap(),
        [vec![f32::INFINITY; 2], vec![]]
    );
}
