use rxla_core::{Client, Graph};

fn client() -> Client {
    unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_minmax_broadcast_ties_and_shared_operands() {
    let client = client();
    let g = Graph::default();
    let a = g.input(&[2, 3]).unwrap();
    let b = g.input(&[3]).unwrap();
    let seed = g.input(&[2, 3]).unwrap();
    let broadcast_b = b.broadcast_to(a.shape()).unwrap();
    let mut outputs = a
        .maximum(&broadcast_b)
        .unwrap()
        .vjp(&[a.clone(), b.clone()], &seed)
        .unwrap();
    outputs.extend(
        a.minimum(&broadcast_b)
            .unwrap()
            .vjp(&[a.clone(), b.clone()], &seed)
            .unwrap(),
    );
    outputs.extend(
        a.maximum(&a)
            .unwrap()
            .vjp(std::slice::from_ref(&a), &seed)
            .unwrap(),
    );
    let exe = g.compile_many(&client, &outputs).unwrap();
    let av = [-2., 1., 4., 0., 3., 2.];
    let bv = [0., 1., 2.];
    let sv = [2., -4., 1., -2., 3., 6.];
    let actual = exe.run_many(&[&av, &bv, &sv]).unwrap();
    for (group, maximum) in [true, false].into_iter().enumerate() {
        let mut da = vec![0.; 6];
        let mut db = vec![0.; 3];
        for i in 0..6 {
            let fraction = if av[i] == bv[i % 3] {
                0.5
            } else if (av[i] > bv[i % 3]) == maximum {
                1.
            } else {
                0.
            };
            da[i] = fraction * sv[i];
            db[i % 3] += (1. - fraction) * sv[i];
        }
        assert_eq!(actual[2 * group], da);
        assert_eq!(actual[2 * group + 1], db);
    }
    assert_eq!(actual[4], sv);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_abs_clamp_and_nonfinite_conventions() {
    let client = client();
    let g = Graph::default();
    let x = g.input(&[8]).unwrap();
    let seed = g.constant(&[8], &[2.; 8]).unwrap();
    let mut outputs = x
        .abs()
        .unwrap()
        .vjp(std::slice::from_ref(&x), &seed)
        .unwrap();
    outputs.extend(
        x.clamp(-1., 1.)
            .unwrap()
            .vjp(std::slice::from_ref(&x), &seed)
            .unwrap(),
    );
    outputs.extend(
        x.clamp(0., 0.)
            .unwrap()
            .vjp(std::slice::from_ref(&x), &seed)
            .unwrap(),
    );
    let exe = g.compile_many(&client, &outputs).unwrap();
    let actual = exe
        .run_many(&[&[-2., -1., -0., 0., 1., 2., f32::NEG_INFINITY, f32::INFINITY]])
        .unwrap();
    assert_eq!(actual[0], [-2., -2., 0., 0., 2., 2., -2., 2.]);
    assert_eq!(actual[1], [0., 1., 2., 2., 1., 0., 0., 0.]);
    assert_eq!(actual[2], [0.; 8]);

    let g = Graph::default();
    let a = g.input(&[6]).unwrap();
    let b = g.input(&[6]).unwrap();
    let seed = g.constant(&[6], &[2.; 6]).unwrap();
    let mut roots = a
        .maximum(&b)
        .unwrap()
        .vjp(&[a.clone(), b.clone()], &seed)
        .unwrap();
    roots.extend(
        a.minimum(&b)
            .unwrap()
            .vjp(&[a.clone(), b.clone()], &seed)
            .unwrap(),
    );
    roots.extend(a.abs().unwrap().vjp(&[a], &seed).unwrap());
    let exe = g.compile_many(&client, &roots).unwrap();
    let actual = exe
        .run_many(&[
            &[
                f32::NAN,
                1.,
                f32::NAN,
                f32::INFINITY,
                -0.,
                f32::NEG_INFINITY,
            ],
            &[1., f32::NAN, f32::NAN, f32::INFINITY, 0., f32::NEG_INFINITY],
        ])
        .unwrap();
    for gradient in &actual[..4] {
        assert!(gradient[..3].iter().all(|v| v.is_nan()));
        assert_eq!(gradient[3..], [1., 1., 1.]);
    }
    assert!(actual[4][0].is_nan() && actual[4][2].is_nan());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_softplus_gradient_at_zero_and_smooth_second_derivative() {
    let client = client();
    let g = Graph::default();
    let x = g.input(&[7]).unwrap();
    let y = x.softplus().unwrap();
    let gradient = y
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let exe = g.compile_many(&client, &[y, gradient]).unwrap();
    let xv = [-1000., -10., -0., 0., 1., 10., 1000.];
    let actual = exe.run_many(&[&xv]).unwrap();
    for (i, x) in xv.into_iter().map(f64::from).enumerate() {
        let forward = x.max(0.) + (-x.abs()).exp().ln_1p();
        let gradient = 1. / (1. + (-x).exp());
        assert!((f64::from(actual[0][i]) - forward).abs() < 1e-6);
        assert!((f64::from(actual[1][i]) - gradient).abs() < 1e-6);
    }
    assert_eq!(&actual[1][2..4], &[0.5, 0.5]);
    for shape in [vec![], vec![0, 2]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let seed = g.input(&shape).unwrap();
        let roots = x.softplus().unwrap().vjp(&[x], &seed).unwrap();
        let exe = g.compile_many(&client, &roots).unwrap();
        let values = vec![0.; usize::from(shape.is_empty())];
        let seed = vec![2.; values.len()];
        assert_eq!(
            exe.run_many(&[&values, &seed]).unwrap(),
            [vec![1.; values.len()]]
        );
    }
    let g = Graph::default();
    let x = g.input(&[3]).unwrap();
    let first = x
        .softplus()
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let second = first
        .sum(&[0], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g.compile_many(&client, &[second]).unwrap();
    let actual = exe.run_many(&[&[-3., 0., 3.]]).unwrap();
    for (actual, x) in actual[0].iter().zip([-3.0_f64, 0., 3.]) {
        let sigmoid = 1. / (1. + (-x).exp());
        assert!((f64::from(*actual) - sigmoid * (1. - sigmoid)).abs() < 1e-6);
    }
}
