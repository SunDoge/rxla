use rxla_core::{Client, Graph};

// Independent F64 integration of the Gaussian PDF, not a backend erf call.
fn exact_reference(x: f64) -> f64 {
    let steps = 1024;
    let h = x / steps as f64;
    let pdf = |t: f64| (-0.5 * t * t).exp() / (2. * std::f64::consts::PI).sqrt();
    let mut sum = pdf(0.) + pdf(x);
    for i in 1..steps {
        sum += if i % 2 == 0 { 2. } else { 4. } * pdf(i as f64 * h);
    }
    x * (0.5 + h / 3. * sum)
}

#[test]
fn gelu_preserves_shapes_and_explicit_formula_choice() {
    let graph = Graph::default();
    for shape in [vec![], vec![2, 3], vec![0, 4]] {
        let x = graph.input(&shape).unwrap();
        for (output, expected_op, absent_op) in [
            (x.gelu().unwrap(), "erf", "tanh"),
            (x.gelu_tanh().unwrap(), "tanh", "erf"),
        ] {
            assert_eq!(output.shape(), shape);
            let stablehlo = graph.stablehlo(&output).unwrap();
            assert!(stablehlo.contains(expected_op));
            assert!(!stablehlo.contains(absent_op));
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_gelu_variants_match_independent_references() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let values: Vec<f32> = (-200..=200).map(|i| i as f32 / 20.).collect();
    let graph = Graph::default();
    let x = graph.input(&[values.len() as i64]).unwrap();
    let outputs = graph
        .compile_many(&client, &[x.gelu().unwrap(), x.gelu_tanh().unwrap()])
        .unwrap()
        .run_many(&[&values])
        .unwrap();
    let mut largest_difference = 0f32;
    let mut max_errors = [0f64; 2];
    for (i, &x) in values.iter().enumerate() {
        let x = x as f64;
        let references = [
            exact_reference(x),
            0.5 * x
                * (1. + ((2. / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()),
        ];
        for variant in 0..2 {
            let actual = outputs[variant][i];
            let error = (actual as f64 - references[variant]).abs();
            max_errors[variant] = max_errors[variant].max(error);
            assert!(
                actual.is_finite() && error < 2e-6,
                "variant {variant}, x={x}, error={error}"
            );
        }
        largest_difference = largest_difference.max((outputs[0][i] - outputs[1][i]).abs());
    }
    assert!(
        largest_difference > 1e-4,
        "variants must not silently collapse into one formula"
    );
    eprintln!(
        "GELU max absolute errors: erf={}, tanh={}",
        max_errors[0], max_errors[1]
    );
    let graph = Graph::default();
    let x = graph.input(&[]).unwrap();
    assert_eq!(
        graph
            .compile(&client, &x.gelu().unwrap())
            .unwrap()
            .run(&[&[0.]])
            .unwrap(),
        [0.]
    );
    let graph = Graph::default();
    let x = graph.input(&[0, 2]).unwrap();
    assert!(
        graph
            .compile(&client, &x.gelu_tanh().unwrap())
            .unwrap()
            .run(&[&[]])
            .unwrap()
            .is_empty()
    );
}
