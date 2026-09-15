use rxla_core::{Client, Graph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_sigmoid_scalar_empty_and_nonfinite() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[]).unwrap();
    let y = x.sigmoid().unwrap();
    let dx = y.grad(&[x]).unwrap().remove(0);
    let exe = g.compile_many(&client, &[y, dx]).unwrap();
    for (x, y, dx) in [
        (f32::NEG_INFINITY, 0., 0.),
        (f32::INFINITY, 1., 0.),
        (-0., 0.5, 0.25),
        (0., 0.5, 0.25),
    ] {
        assert_eq!(exe.run_many(&[&[x]]).unwrap(), [vec![y], vec![dx]]);
    }
    assert!(
        exe.run_many(&[&[f32::NAN]])
            .unwrap()
            .iter()
            .all(|v| v[0].is_nan())
    );
    let g = Graph::default();
    let x = g.input(&[0, 2]).unwrap();
    let y = x.sigmoid().unwrap();
    let dx = y.sum(&[0, 1], false).unwrap().grad(&[x]).unwrap().remove(0);
    let exe = g.compile_many(&client, &[y, dx]).unwrap();
    assert_eq!(exe.run_many(&[&[]]).unwrap(), [Vec::<f32>::new(), vec![]]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_sigmoid_silu_softplus_finite_extremes_and_higher_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let values = [
        -1000., -100., -80., -20., -3., -0., 0., 3., 20., 80., 100., 1000.,
    ];
    let x = g.input(&[values.len() as i64]).unwrap();
    let mut roots = Vec::new();
    for y in [
        x.sigmoid().unwrap(),
        x.silu().unwrap(),
        x.softplus().unwrap(),
    ] {
        let first = y
            .sum(&[0], false)
            .unwrap()
            .grad(std::slice::from_ref(&x))
            .unwrap()
            .remove(0);
        let second = first
            .sum(&[0], false)
            .unwrap()
            .grad(std::slice::from_ref(&x))
            .unwrap()
            .remove(0);
        roots.extend([y, first, second]);
    }
    let exe = g.compile_many(&client, &roots).unwrap();
    let actual = exe.run_many(&[&values]).unwrap();
    for (i, x) in values.into_iter().map(f64::from).enumerate() {
        let z = (-x.abs()).exp();
        let s = if x >= 0. { 1. / (1. + z) } else { z / (1. + z) };
        let ds = z / (1. + z).powi(2);
        let dds = ds * (1. - 2. * s);
        let expected = [
            s,
            ds,
            dds,
            x * s,
            s + x * ds,
            2. * ds + x * dds,
            x.max(0.) + z.ln_1p(),
            s,
            ds,
        ];
        for (output, expected) in expected.into_iter().enumerate() {
            let actual = f64::from(actual[output][i]);
            assert!(
                actual.is_finite(),
                "output={output}, x={x}, actual={actual}"
            );
            assert!(
                (actual - expected).abs() <= 2e-5 * expected.abs() + 1e-37,
                "output={output}, x={x}, actual={actual}, expected={expected}"
            );
        }
    }
}
