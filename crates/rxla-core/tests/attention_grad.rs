use rxla_core::{Client, Tracer};

// Independent scalar reference, deliberately without tensor library operations.
// Two batches, four Q heads, two queries, three keys, depth 2, value depth 3.
fn reference(data: &[Vec<f64>], kv_heads: usize, weights: &[f32]) -> f64 {
    let mut loss = 0.;
    for batch in 0..2 {
        for head in 0..4 {
            let kv_head = head / (4 / kv_heads);
            for query in 0..2 {
                let scores: Vec<_> = (0..3)
                    .map(|key| {
                        let dot: f64 = (0..2)
                            .map(|d| {
                                data[0][((batch * 4 + head) * 2 + query) * 2 + d]
                                    * data[1][((batch * kv_heads + kv_head) * 3 + key) * 2 + d]
                            })
                            .sum();
                        dot * 0.7f32 as f64 + data[3][(head * 2 + query) * 3 + key]
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                for d in 0..3 {
                    let output: f64 = (0..3)
                        .map(|key| {
                            (scores[key] - max).exp() / total
                                * data[2][((batch * kv_heads + kv_head) * 3 + key) * 3 + d]
                        })
                        .sum();
                    loss += output * weights[((batch * 4 + head) * 2 + query) * 3 + d] as f64;
                }
            }
        }
    }
    loss
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_mqa_gqa_mha_gradients_match_independent_f64_differences() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for kv_heads in [1, 2, 4] {
        let g = Tracer::default();
        let q = g.input(&[2, 4, 2, 2]).unwrap();
        let k = g.input(&[2, kv_heads as i64, 3, 2]).unwrap();
        let v = g.input(&[2, kv_heads as i64, 3, 3]).unwrap();
        let mask = g.input(&[1, 4, 2, 3]).unwrap();
        let weights: Vec<_> = (0..48).map(|i| ((i * 7 % 17) as f32 - 8.) / 16.).collect();
        let output = q
            .grouped_query_attention(&k, &v, Some(&mask), Some(0.7))
            .unwrap();
        let loss = output
            .mul(&g.constant(&[2, 4, 2, 3], &weights).unwrap())
            .unwrap()
            .sum(&[0, 1, 2, 3], false)
            .unwrap();
        let mut roots = vec![loss.clone()];
        roots.extend(loss.grad(&[q, k, v, mask]).unwrap());
        let exe = g.compile_many(&client, &roots).unwrap();
        for causal in [false, true] {
            let mut data: Vec<Vec<f64>> = [32, 12 * kv_heads, 18 * kv_heads, 24]
                .into_iter()
                .enumerate()
                .map(|(group, len)| {
                    (0..len)
                        .map(|i| (((i * 5 + group * 3) % 19) as f64 - 9.) / 16.)
                        .collect()
                })
                .collect();
            if causal {
                for head in 0..4 {
                    for query in 0..2 {
                        for key in query + 1..3 {
                            data[3][(head * 2 + query) * 3 + key] = f64::NEG_INFINITY;
                        }
                    }
                }
            }
            let f32data: Vec<Vec<f32>> = data
                .iter()
                .map(|v| v.iter().map(|v| *v as f32).collect())
                .collect();
            let actual = exe
                .run_many(&f32data.iter().map(Vec::as_slice).collect::<Vec<_>>())
                .unwrap();
            assert!((actual[0][0] as f64 - reference(&data, kv_heads, &weights)).abs() < 2e-6);
            for group in 0..4 {
                for element in 0..data[group].len() {
                    let original = data[group][element];
                    if !original.is_finite() {
                        assert_eq!(actual[group + 1][element], 0.);
                        continue;
                    }
                    data[group][element] = original + 1e-5;
                    let plus = reference(&data, kv_heads, &weights);
                    data[group][element] = original - 1e-5;
                    let minus = reference(&data, kv_heads, &weights);
                    data[group][element] = original;
                    let expected = (plus - minus) / 2e-5;
                    let result = actual[group + 1][element] as f64;
                    assert!(
                        (result - expected).abs() < 2e-6,
                        "KV heads {kv_heads}, causal {causal}, group {group}, element {element}: {result} != {expected}"
                    );
                }
            }
        }
    }
}
