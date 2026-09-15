use rxla_core::{Client, Graph};

#[test]
fn stable_unary_shape_and_native_lowering() {
    let g = Graph::default();
    for shape in [vec![], vec![2, 3], vec![0]] {
        let x = g.input(&shape).unwrap();
        for (y, opcode) in [
            (x.log1p().unwrap(), "stablehlo.log"),
            (x.expm1().unwrap(), "stablehlo.exponential"),
            (x.abs().unwrap(), "abs"),
        ] {
            assert_eq!(y.shape(), shape);
            assert!(g.stablehlo(&y).unwrap().contains(opcode));
        }
        assert_eq!(x.softplus().unwrap().shape(), shape);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_small_log1p_and_expm1_match_f64() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let values = [
        -0.999999, -0.5, -1e-4, -1e-8, -1e-12, 0., 1e-12, 1e-8, 1e-4, 0.5, 10.,
    ];
    let g = Graph::default();
    let x = g.input(&[values.len() as i64]).unwrap();
    let outputs = g
        .compile_many(&client, &[x.log1p().unwrap(), x.expm1().unwrap()])
        .unwrap()
        .run_many(&[&values])
        .unwrap();
    for (i, &x) in values.iter().enumerate() {
        for (actual, expected) in [
            (outputs[0][i], (x as f64).ln_1p()),
            (outputs[1][i], (x as f64).exp_m1()),
        ] {
            assert!(actual.is_finite());
            assert!(
                (actual as f64 - expected).abs() <= 2e-7 * expected.abs() + 1e-20,
                "x={x}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_softplus_extremes_and_unary_special_values() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let values = [-1000., -80., -20., -1., 0., 1., 20., 80., 1000.];
    let g = Graph::default();
    let x = g.input(&[values.len() as i64]).unwrap();
    let outputs = g
        .compile(&client, &x.softplus().unwrap())
        .unwrap()
        .run(&[&values])
        .unwrap();
    for (&x, &actual) in values.iter().zip(&outputs) {
        let x = x as f64;
        let expected = if x > 0. {
            x + (-x).exp().ln_1p()
        } else {
            x.exp().ln_1p()
        };
        assert!(
            actual.is_finite() && (actual as f64 - expected).abs() <= 3e-7 * expected.abs() + 1e-40
        );
    }
    let g = Graph::default();
    let x = g.input(&[4]).unwrap();
    let outputs = g
        .compile_many(
            &client,
            &[
                x.log1p().unwrap(),
                x.expm1().unwrap(),
                x.abs().unwrap(),
                x.softplus().unwrap(),
            ],
        )
        .unwrap()
        .run_many(&[&[-1., f32::NEG_INFINITY, f32::INFINITY, f32::NAN]])
        .unwrap();
    assert_eq!(outputs[0][0], f32::NEG_INFINITY);
    assert!(outputs[0][1].is_nan());
    assert_eq!(outputs[0][2], f32::INFINITY);
    assert_eq!(outputs[1][1], -1.);
    assert_eq!(outputs[1][2], f32::INFINITY);
    assert_eq!(&outputs[2][..3], &[1., f32::INFINITY, f32::INFINITY]);
    assert_eq!(outputs[3][1], 0.);
    assert_eq!(outputs[3][2], f32::INFINITY);
    for output in &outputs {
        assert!(output[3].is_nan());
    }
}
