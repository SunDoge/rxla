use rxla_core::{Client, Graph};

#[test]
fn attention_validation_and_broadcast_shapes() {
    let g = Graph::default();
    let q = g.input(&[2, 1, 2]).unwrap();
    let k = g.input(&[1, 3, 2]).unwrap();
    let v = g.input(&[2, 3, 4]).unwrap();
    let mask = g.input(&[3]).unwrap();
    assert_eq!(
        q.scaled_dot_product_attention(&k, &v, Some(&mask), None)
            .unwrap()
            .shape(),
        [2, 1, 4]
    );
    let empty_query = g.input(&[2, 0, 2]).unwrap();
    assert_eq!(
        empty_query
            .scaled_dot_product_attention(&k, &v, None, None)
            .unwrap()
            .shape(),
        [2, 0, 4]
    );
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            q.scaled_dot_product_attention(&k, &v, None, Some(scale))
                .is_err()
        );
    }
    for bad_key in [
        g.input(&[3, 1]).unwrap(),
        g.input(&[0, 2]).unwrap(),
        g.input(&[2]).unwrap(),
        g.input(&[3, 3, 2]).unwrap(),
    ] {
        assert!(
            q.scaled_dot_product_attention(&bad_key, &v, None, None)
                .is_err()
        );
    }
    let bad_value = g.input(&[2, 4, 4]).unwrap();
    assert!(
        q.scaled_dot_product_attention(&k, &bad_value, None, None)
            .is_err()
    );
    let bad_mask = g.input(&[4]).unwrap();
    assert!(
        q.scaled_dot_product_attention(&k, &v, Some(&bad_mask), None)
            .is_err()
    );
    let foreign = Graph::default().input(&[3]).unwrap();
    assert!(
        q.scaled_dot_product_attention(&k, &v, Some(&foreign), None)
            .is_err()
    );
    let zero_depth = g.input(&[1, 0]).unwrap();
    assert!(
        zero_depth
            .scaled_dot_product_attention(&zero_depth, &g.input(&[1, 2]).unwrap(), None, None)
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_attention_broadcast_mask_and_explicit_scales() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let q = g.input(&[2, 1, 2]).unwrap();
    let k = g.input(&[1, 3, 2]).unwrap();
    let v = g.input(&[2, 3, 1]).unwrap();
    let mask = g.input(&[3]).unwrap();
    let uniform = q
        .scaled_dot_product_attention(&k, &v, Some(&mask), Some(0.))
        .unwrap();
    let negative = q
        .scaled_dot_product_attention(&k, &v, Some(&mask), Some(-1.))
        .unwrap();
    let exe = g.compile_many(&client, &[uniform, negative]).unwrap();
    let bias = 2f32.ln();
    let outputs = exe
        .run_many(&[
            &[1., 0., 1., 0.],
            &[1., 0., 0., 0., 0., 1.],
            &[3., 6., 1000., 9., 12., 1000.],
            &[0., bias, f32::NEG_INFINITY],
        ])
        .unwrap();
    let b = (bias as f64).exp();
    for (output, scale) in outputs.iter().zip([0f64, -1.]) {
        let a = scale.exp();
        for (actual, (first, second)) in output.iter().zip([(3., 6.), (9., 12.)]) {
            let expected = (first * a + second * b) / (a + b);
            assert!((*actual as f64 - expected).abs() < 2e-6);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_multihead_attention() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    // Two batches, three heads, four queries, five keys, head dimension two.
    let q = g.input(&[2, 3, 4, 2]).unwrap();
    let k = g.input(&[2, 3, 5, 2]).unwrap();
    let v = g.input(&[2, 3, 5, 2]).unwrap();
    let probabilities = q
        .matmul(&k.transpose(&[0, 1, 3, 2]).unwrap())
        .unwrap()
        .mul_scalar(1. / 2f32.sqrt())
        .unwrap()
        .softmax(3)
        .unwrap();
    let y = q.scaled_dot_product_attention(&k, &v, None, None).unwrap();
    let exe = g.compile_many(&client, &[y, probabilities]).unwrap();
    let qv: Vec<_> = (0..48).map(|i| ((i * 7 % 19) as f32 - 9.) / 8.).collect();
    let kv: Vec<_> = (0..60).map(|i| ((i * 3 % 17) as f32 - 8.) / 7.).collect();
    let vv: Vec<_> = (0..60).map(|i| ((i * 11 % 23) as f32 - 11.) / 6.).collect();
    let actual = exe.run_many(&[&qv, &kv, &vv]).unwrap();
    let mut expected_y = Vec::new();
    let mut expected_p = Vec::new();
    for bh in 0..6 {
        for query in 0..4 {
            let mut logits: Vec<f64> = (0..5)
                .map(|key| {
                    (0..2)
                        .map(|d| {
                            qv[bh * 8 + query * 2 + d] as f64 * kv[bh * 10 + key * 2 + d] as f64
                        })
                        .sum::<f64>()
                        / 2f64.sqrt()
                })
                .collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            logits.iter_mut().for_each(|x| *x = (*x - max).exp());
            let sum: f64 = logits.iter().sum();
            logits.iter_mut().for_each(|x| *x /= sum);
            for d in 0..2 {
                expected_y.push(
                    (0..5)
                        .map(|key| logits[key] * vv[bh * 10 + key * 2 + d] as f64)
                        .sum::<f64>(),
                );
            }
            expected_p.extend(logits);
        }
    }
    for (actual, expected) in actual.iter().zip([expected_y, expected_p]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, e) in actual.iter().zip(expected) {
            assert!((a as f64 - e).abs() < 2e-6, "actual {a}, expected {e}");
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_attention_gradients_match_portable_decomposition() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let q = g.input(&[1, 2, 2]).unwrap();
    let k = g.input(&[1, 3, 2]).unwrap();
    let v = g.input(&[1, 3, 2]).unwrap();
    let weights = g.input(&[1, 2, 2]).unwrap();
    let scale = 0.7;
    let semantic = q
        .scaled_dot_product_attention(&k, &v, None, Some(scale))
        .unwrap();
    let portable = q
        .matmul(&k.swap_axes(1, 2).unwrap())
        .unwrap()
        .mul_scalar(scale)
        .unwrap()
        .softmax(2)
        .unwrap()
        .matmul(&v)
        .unwrap();
    let mut roots = Vec::new();
    for output in [semantic, portable] {
        let loss = output
            .mul(&weights)
            .unwrap()
            .sum(&[0, 1, 2], false)
            .unwrap();
        roots.extend(loss.grad(&[q.clone(), k.clone(), v.clone()]).unwrap());
    }
    let executable = g.compile_many(&client, &roots).unwrap();
    let outputs = executable
        .run_many(&[
            &[0.2, -0.5, 1.0, 0.3],
            &[0.4, 0.7, -0.2, 0.9, 0.5, -0.6],
            &[1.2, -0.1, 0.3, 0.8, -0.7, 0.2],
            &[0.5, -0.4, 0.9, 0.1],
        ])
        .unwrap();
    for (semantic, portable) in outputs[..3].iter().zip(&outputs[3..]) {
        for (&actual, &expected) in semantic.iter().zip(portable) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
    }
}
