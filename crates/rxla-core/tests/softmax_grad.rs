use rxla_core::{Client, Tracer};

#[test]
fn detach_only_stops_its_own_path() {
    let g = Tracer::default();
    let x = g.input(&[2]).unwrap();
    // Direct mask derivatives remain unsupported; max reductions now have a rule.
    let finite_count = x.is_finite_mask().unwrap().sum(&[0], false).unwrap();
    assert!(
        finite_count
            .detach()
            .unwrap()
            .grad(std::slice::from_ref(&x))
            .is_ok()
    );
    let mixed = finite_count.add(&finite_count.detach().unwrap()).unwrap();
    assert!(mixed.grad(&[x]).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_detach_keeps_forward_values_and_other_gradient_paths() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[]).unwrap();
    let detached = x.mul(&x).unwrap().detach().unwrap();
    let loss = detached.mul(&x).unwrap();
    let gradient = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let second = gradient.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let exe = g.compile_many(&client, &[loss, gradient, second]).unwrap();
    for value in [2., 3.] {
        assert_eq!(
            exe.run_many(&[&[value]]).unwrap(),
            [vec![value * value * value], vec![value * value], vec![0.]]
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_softmax_and_log_softmax_gradients_match_f64_formulas() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axis in [0, 1] {
        let g = Tracer::default();
        let x = g.input(&[2, 3]).unwrap();
        let weights = g.input(&[2, 3]).unwrap();
        let soft = x.softmax(axis).unwrap();
        let log = x.log_softmax(axis).unwrap();
        let mut roots = vec![soft.clone(), log.clone()];
        for result in [soft, log] {
            let loss = result.mul(&weights).unwrap().sum(&[0, 1], false).unwrap();
            roots.push(loss.grad(std::slice::from_ref(&x)).unwrap().remove(0));
        }
        let exe = g.compile_many(&client, &roots).unwrap();
        for values in [
            [1., 1., -2., 0., 3., 3.],
            [1000., 999., 998., -1000., -999., -998.],
        ] {
            let weights = [0.2, -0.4, 0.7, 1., -0.3, 0.5];
            let actual = exe.run_many(&[&values, &weights]).unwrap();
            let (groups, length) = if axis == 0 { (3, 2) } else { (2, 3) };
            for group in 0..groups {
                let index = |i| {
                    if axis == 0 {
                        i * 3 + group
                    } else {
                        group * 3 + i
                    }
                };
                let max = (0..length)
                    .map(|i| values[index(i)] as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = (0..length)
                    .map(|i| (values[index(i)] as f64 - max).exp())
                    .sum();
                let p: Vec<_> = (0..length)
                    .map(|i| (values[index(i)] as f64 - max).exp() / total)
                    .collect();
                let weighted: f64 = (0..length).map(|i| p[i] * weights[index(i)] as f64).sum();
                let sum: f64 = (0..length).map(|i| weights[index(i)] as f64).sum();
                for (i, &probability) in p.iter().enumerate() {
                    let j = index(i);
                    let expected = [
                        probability,
                        values[j] as f64 - max - total.ln(),
                        probability * (weights[j] as f64 - weighted),
                        weights[j] as f64 - probability * sum,
                    ];
                    for (root, expected) in expected.into_iter().enumerate() {
                        assert!(
                            (actual[root][j] as f64 - expected).abs() < 2e-4,
                            "axis {axis}, root {root}, element {j}: {} != {expected}",
                            actual[root][j]
                        );
                    }
                }
            }
        }
    }
}
