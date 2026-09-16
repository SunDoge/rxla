use rxla_core::{Client, Tracer};

#[test]
fn gqa_layout_validation() {
    let g = Tracer::default();
    let q = g.input(&[2, 4, 3, 2]).unwrap();
    let k = g.input(&[2, 2, 5, 2]).unwrap();
    let v = g.input(&[2, 2, 5, 7]).unwrap();
    assert_eq!(
        q.grouped_query_attention(&k, &v, None, None)
            .unwrap()
            .shape(),
        [2, 4, 3, 7]
    );
    for shape in [&[2, 3, 5, 2][..], &[1, 2, 5, 2], &[2, 5, 2], &[2, 0, 5, 2]] {
        assert!(
            q.grouped_query_attention(&g.input(shape).unwrap(), &v, None, None)
                .is_err()
        );
    }
    let mask = g.input(&[3, 5]).unwrap();
    assert!(q.grouped_query_attention(&k, &v, Some(&mask), None).is_ok());
    let bad_mask = g.input(&[4, 5]).unwrap();
    assert!(
        q.grouped_query_attention(&k, &v, Some(&bad_mask), None)
            .is_err()
    );
    let foreign = Tracer::default().input(&[2, 2, 5, 2]).unwrap();
    assert!(q.grouped_query_attention(&foreign, &v, None, None).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_gqa_multiquery_head_masks_match_f64() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let qv: Vec<f32> = (0..48).map(|i| (i as f32 % 9. - 4.) / 5.).collect();
    // Broadcast over batch, but distinct for every query head and query row.
    let mask: Vec<f32> = (0..48)
        .map(|i| {
            if i % 4 == 3 {
                f32::NEG_INFINITY
            } else {
                (i as f32 % 7.) / 8.
            }
        })
        .collect();
    for kv_heads in [1usize, 2, 4] {
        let g = Tracer::default();
        let q = g.input(&[2, 4, 3, 2]).unwrap();
        let k = g.input(&[2, kv_heads as i64, 4, 2]).unwrap();
        let v = g.input(&[2, kv_heads as i64, 4, 3]).unwrap();
        let m = g.input(&[1, 4, 3, 4]).unwrap();
        let output = q.grouped_query_attention(&k, &v, Some(&m), None).unwrap();
        let exe = g.compile(&client, &output).unwrap();
        let keys: Vec<f32> = (0..2 * kv_heads * 4 * 2)
            .map(|i| (i as f32 % 11. - 5.) / 7.)
            .collect();
        let values: Vec<f32> = (0..2 * kv_heads * 4 * 3)
            .map(|i| (i as f32 % 13. - 6.) / 3.)
            .collect();
        let actual = exe.run(&[&qv, &keys, &values, &mask]).unwrap();
        let mut expected = Vec::new();
        for batch in 0..2 {
            for head in 0..4 {
                for row in 0..3 {
                    let kv = head / (4 / kv_heads);
                    let logits: Vec<f64> = (0..4)
                        .map(|col| {
                            let dot = (0..2)
                                .map(|d| {
                                    qv[((batch * 4 + head) * 3 + row) * 2 + d] as f64
                                        * keys[((batch * kv_heads + kv) * 4 + col) * 2 + d] as f64
                                })
                                .sum::<f64>();
                            dot / 2f64.sqrt() + mask[(head * 3 + row) * 4 + col] as f64
                        })
                        .collect();
                    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probs: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
                    let sum = probs.iter().sum::<f64>();
                    for d in 0..3 {
                        expected.push(
                            (0..4)
                                .map(|col| {
                                    probs[col] / sum
                                        * values[((batch * kv_heads + kv) * 4 + col) * 3 + d] as f64
                                })
                                .sum::<f64>(),
                        );
                    }
                }
            }
        }
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (*a as f64 - e).abs() < 2e-6,
                "{a} vs {e}, kv_heads={kv_heads}"
            );
        }
    }
}
