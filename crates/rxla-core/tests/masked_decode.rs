use rxla_core::{CacheLimits, Client, Compiler, Graph, KvCache, StateGraph};

#[test]
fn mask_shape_validation() {
    let g = Graph::default();
    let x = g.input_i32(&[2, 1]).unwrap();
    assert_eq!(x.broadcast_to(&[2, 3]).unwrap().shape(), [2, 3]);
    assert!(x.broadcast_to(&[3]).is_err());
    assert!(x.broadcast_to(&[3, 3]).is_err());
    assert!(x.broadcast_to(&[-1, 3]).is_err());
    assert!(x.le_mask(&g.input_i32_scalar().unwrap()).is_err());
    assert!(
        x.le_mask(&Graph::default().input_i32(&[2, 1]).unwrap())
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_signed_masks_and_integer_broadcast() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let values = g.constant_i32(&[4], &[i32::MIN, -1, 0, i32::MAX]).unwrap();
    let threshold = g.input_i32_scalar().unwrap();
    let mask = values
        .le_mask(&threshold.broadcast_to(&[4]).unwrap())
        .unwrap();
    let exponent = mask.log().unwrap().exp().unwrap();
    let exe = g.compile_many(&client, &[mask, exponent]).unwrap();
    let index = client.buffer(&[], &[-1]).unwrap();
    for output in exe.execute(&[&index]).unwrap() {
        assert_eq!(output.to_vec::<f32>().unwrap(), [1., 1., 0., 0.]);
    }
    let h = Graph::default();
    let x = h
        .constant_i32(&[2, 1], &[1, 4])
        .unwrap()
        .broadcast_to(&[2, 3])
        .unwrap();
    let y = h
        .constant_i32(&[3], &[0, 2, 5])
        .unwrap()
        .broadcast_to(&[2, 3])
        .unwrap();
    assert_eq!(
        h.compile(&client, &x.le_mask(&y).unwrap())
            .unwrap()
            .run(&[])
            .unwrap(),
        [0., 1., 1., 0., 0., 1.]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_stateful_masked_decode_matches_prefix_attention() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let mut cache = KvCache::new(&mut g, &[2, 4, 2]).unwrap();
    let cache_k = cache.key_slot().clone();
    let q = g.input(&[2, 1, 2]).unwrap();
    let cache_v = cache.value_slot().clone();
    let k = g.input(&[2, 1, 2]).unwrap();
    let v = g.input(&[2, 1, 2]).unwrap();
    let position = g.input_i32_scalar().unwrap();
    let zero = g.scalar_i32(0).unwrap();
    let starts = [zero.clone(), position.clone(), zero];
    let (next_k, next_v) = cache.update_at(&mut g, &k, &v, &starts).unwrap();
    let positions = g.constant_i32(&[4], &[0, 1, 2, 3]).unwrap();
    let mask = positions
        .le_mask(&position.broadcast_to(&[4]).unwrap())
        .unwrap()
        .log()
        .unwrap()
        .broadcast_to(&[2, 1, 4])
        .unwrap();
    let probabilities = q
        .matmul(&next_k.transpose(&[0, 2, 1]).unwrap())
        .unwrap()
        .mul_scalar(1. / 2f32.sqrt())
        .unwrap()
        .add(&mask)
        .unwrap()
        .softmax(2)
        .unwrap();
    let output = q
        .scaled_dot_product_attention(&next_k, &next_v, Some(&mask), None)
        .unwrap();
    let program = g.compile(&mut compiler, &[output, probabilities]).unwrap();
    // Deliberately nonzero future entries would dominate the result without a mask.
    let mut session = program
        .session(vec![
            (cache_k, client.buffer(&[2, 4, 2], &[100.; 16]).unwrap()),
            (cache_v, client.buffer(&[2, 4, 2], &[1000.; 16]).unwrap()),
        ])
        .unwrap();
    let mut keys: Vec<[f32; 4]> = Vec::new();
    let mut values: Vec<[f32; 4]> = Vec::new();
    for step in 0..4 {
        let s = step as f32;
        let q = [0.5 + s / 4., -0.25, 0.75, 0.1 + s / 3.];
        let k = [s / 2., 0.5, 1. - s / 3., -0.25];
        let v = [s + 1., -s, 2. * s, 0.25 + s];
        keys.push(k);
        values.push(v);
        let qb = client.buffer(&[2, 1, 2], &q).unwrap();
        let kb = client.buffer(&[2, 1, 2], &k).unwrap();
        let vb = client.buffer(&[2, 1, 2], &v).unwrap();
        let pb = client.buffer(&[], &[step as i32]).unwrap();
        let outputs = session.run(&[&qb, &kb, &vb, &pb]).unwrap();
        let actual = outputs[0].to_vec::<f32>().unwrap();
        let actual_p = outputs[1].to_vec::<f32>().unwrap();
        for head in 0..2 {
            let logits: Vec<f64> = keys
                .iter()
                .map(|key| {
                    (0..2)
                        .map(|d| q[head * 2 + d] as f64 * key[head * 2 + d] as f64)
                        .sum::<f64>()
                        / 2f64.sqrt()
                })
                .collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let denom: f64 = exp.iter().sum();
            for d in 0..2 {
                let expected: f64 = values
                    .iter()
                    .zip(&exp)
                    .map(|(v, p)| v[head * 2 + d] as f64 * p / denom)
                    .sum();
                assert!((actual[head * 2 + d] as f64 - expected).abs() < 2e-6);
            }
            for t in 0..4 {
                let expected = if t <= step { exp[t] / denom } else { 0. };
                assert!((actual_p[head * 4 + t] as f64 - expected).abs() < 1e-6);
            }
        }
    }
    assert_eq!(compiler.stats().misses, 1);
}
