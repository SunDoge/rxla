use rxla_core::{Client, Graph};

#[test]
fn dense_cross_entropy_validates_shapes_axis_and_owner() {
    let g = Graph::default();
    let x = g.input(&[2, 3]).unwrap();
    let y = g.input(&[2, 3]).unwrap();
    assert_eq!(x.cross_entropy_with_probs(&y, 1).unwrap().shape(), [2]);
    assert_eq!(x.cross_entropy_with_probs(&y, 0).unwrap().shape(), [3]);
    assert!(x.cross_entropy_with_probs(&y, 2).is_err());
    assert!(
        x.cross_entropy_with_probs(&g.input(&[3]).unwrap(), 1)
            .is_err()
    );
    assert!(
        x.cross_entropy_with_probs(&Graph::default().input(&[2, 3]).unwrap(), 1)
            .is_err()
    );
    let empty = g.input(&[2, 0]).unwrap();
    assert!(empty.cross_entropy_with_probs(&empty, 1).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_dense_cross_entropy_and_both_gradients_match_f64() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axis in [0, 1] {
        let g = Graph::default();
        let x = g.input(&[2, 3]).unwrap();
        let targets = g.input(&[2, 3]).unwrap();
        let per_group = x.cross_entropy_with_probs(&targets, axis).unwrap();
        let mut roots = vec![per_group.clone()];
        roots.extend(
            per_group
                .sum(&[0], false)
                .unwrap()
                .grad(&[x, targets])
                .unwrap(),
        );
        let exe = g.compile_many(&client, &roots).unwrap();
        for values in [
            [0., 1., -2., 3., 3., 3.],
            [1000., 999., -1000., -1000., 0., 1000.],
        ] {
            // Includes zeros and unnormalized coefficients: no hidden normalization.
            let labels = [0., 0.7, 0.3, 1., 0., 0.5];
            let actual = exe.run_many(&[&values, &labels]).unwrap();
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
                let norm: f64 = (0..length)
                    .map(|i| (values[index(i)] as f64 - max).exp())
                    .sum();
                let mass: f64 = (0..length).map(|i| labels[index(i)] as f64).sum();
                let mut loss = 0.;
                for i in 0..length {
                    let j = index(i);
                    let logp = values[j] as f64 - max - norm.ln();
                    loss -= labels[j] as f64 * logp;
                    assert!(
                        (actual[1][j] as f64 - (mass * logp.exp() - labels[j] as f64)).abs() < 2e-4
                    );
                    assert!((actual[2][j] as f64 + logp).abs() < 2e-4);
                }
                assert!((actual[0][group] as f64 - loss).abs() < 3e-4);
            }
        }
    }
    let g = Graph::default();
    let empty = g.input(&[0, 3]).unwrap();
    let loss = empty.cross_entropy_with_probs(&empty, 1).unwrap();
    assert!(
        g.compile(&client, &loss)
            .unwrap()
            .run(&[&[]])
            .unwrap()
            .is_empty()
    );
}
