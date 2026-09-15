use rxla_core::{Client, Graph};

// Independent f64 Simpson integration of 2/sqrt(pi) * exp(-t*t).
// Only used on the bounded finite interval exercised below.
fn reference(x: f64) -> f64 {
    let steps = 512;
    let h = x / steps as f64;
    let mut sum = 1. + (-x * x).exp();
    for i in 1..steps {
        let t = i as f64 * h;
        sum += if i % 2 == 0 { 2. } else { 4. } * (-t * t).exp();
    }
    2. / std::f64::consts::PI.sqrt() * h / 3. * sum
}

#[test]
fn erf_shape_and_lowering() {
    let g = Graph::default();
    for shape in [vec![], vec![2, 3], vec![0]] {
        let x = g.input(&shape).unwrap();
        let y = x.erf().unwrap();
        assert_eq!(y.shape(), shape);
        assert!(g.stablehlo(&y).unwrap().contains("chlo.erf"));
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_erf_matches_integral_and_special_values() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let values: Vec<f32> = (-120..=120).map(|i| i as f32 / 20.).collect();
    let g = Graph::default();
    let x = g.input(&[values.len() as i64]).unwrap();
    let y = x.erf().unwrap();
    let actual = g.compile(&client, &y).unwrap().run(&[&values]).unwrap();
    let mut maximum_error = 0f64;
    for (&x, &y) in values.iter().zip(&actual) {
        let error = (y as f64 - reference(x as f64)).abs();
        maximum_error = maximum_error.max(error);
        assert!(
            y.is_finite() && error < 3e-7,
            "erf({x}) = {y}, error {error}"
        );
    }
    eprintln!("Erf maximum absolute error vs f64 quadrature: {maximum_error}");
    let g = Graph::default();
    let x = g.input(&[5]).unwrap();
    let values = [f32::NEG_INFINITY, f32::INFINITY, f32::NAN, 0., -0.];
    let result = g
        .compile(&client, &x.erf().unwrap())
        .unwrap()
        .run(&[&values])
        .unwrap();
    assert_eq!(result[0], -1.);
    assert_eq!(result[1], 1.);
    assert!(result[2].is_nan());
    assert_eq!(&result[3..], &[0., 0.]);
}
