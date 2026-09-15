use rxla_core::{CacheLimits, Client, Compiler, Graph};

#[test]
fn explicit_positions_validate_rank_and_graph() {
    let g = Graph::default();
    let q = g.input_i32(&[2]).unwrap();
    assert!(
        q.causal_attention_mask(&g.input_i32_scalar().unwrap())
            .is_err()
    );
    assert!(
        q.causal_attention_mask(&Graph::default().input_i32(&[3]).unwrap())
            .is_err()
    );
    assert_eq!(
        q.causal_attention_mask(&g.input_i32(&[0]).unwrap())
            .unwrap()
            .shape(),
        [2, 0]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_runtime_positions_reuse_executable_and_preserve_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let g = Graph::default();
    let positions = g.input_i32(&[2]).unwrap();
    let keys = g.input_i32(&[4]).unwrap();
    let value = g.input(&[4, 1]).unwrap();
    let mask = positions.causal_attention_mask(&keys).unwrap();
    let query = g.constant(&[2, 1], &[0., 0.]).unwrap();
    let key = g.constant(&[4, 1], &[0.; 4]).unwrap();
    let y = query
        .scaled_dot_product_attention(&key, &value, Some(&mask), Some(1.))
        .unwrap();
    let grad = y
        .sum(&[0, 1], false)
        .unwrap()
        .grad(&[value])
        .unwrap()
        .remove(0);
    let exe = compiler.compile_many(&g, &[y, grad, mask]).unwrap();
    let k = client.buffer(&[4], &[0, 1, 2, 3]).unwrap();
    let v = client.buffer(&[4, 1], &[10., 20., 30., 40.]).unwrap();
    for (q, expected, gradient) in [
        ([0, 2], [10., 20.], [4. / 3., 1. / 3., 1. / 3., 0.]),
        ([1, 3], [15., 25.], [0.75, 0.75, 0.25, 0.25]),
    ] {
        let q = client.buffer(&[2], &q).unwrap();
        let out = exe.execute(&[&q, &k, &v]).unwrap();
        for (a, b) in out[0].to_vec::<f32>().unwrap().iter().zip(expected) {
            assert!((a - b).abs() < 2e-5);
        }
        for (a, b) in out[1].to_vec::<f32>().unwrap().iter().zip(gradient) {
            assert!((a - b).abs() < 2e-5);
        }
    }
    let q = client.buffer(&[2], &[16_777_216, i32::MAX]).unwrap();
    let k = client
        .buffer(&[4], &[16_777_216, 16_777_217, i32::MAX, i32::MIN])
        .unwrap();
    let out = exe.execute(&[&q, &k, &v]).unwrap();
    assert_eq!(
        out[2].to_vec::<f32>().unwrap(),
        [0., f32::NEG_INFINITY, f32::NEG_INFINITY, 0., 0., 0., 0., 0.]
    );
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
fn mask_positions_are_checked() {
    let g = Graph::default();
    for (q, k, o) in [
        (-1, 2, 0),
        (2, -1, 0),
        (2, 2, -1),
        (2, 2, i64::from(i32::MAX)),
        (1, i64::MAX, 0),
    ] {
        assert!(g.causal_attention_mask(q, k, o).is_err());
    }
    assert_eq!(g.causal_attention_mask(0, 4, 0).unwrap().shape(), [0, 4]);
    assert_eq!(
        g.causal_attention_mask(1, 1, i64::from(i32::MAX))
            .unwrap()
            .shape(),
        [1, 1]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_full_and_chunked_causal_attention() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let q = g.input(&[5, 2]).unwrap();
    let k = g.input(&[7, 2]).unwrap();
    let v = g.input(&[7, 2]).unwrap();
    let mask = g.causal_attention_mask(5, 7, 0).unwrap();
    let full = q
        .scaled_dot_product_attention(&k, &v, Some(&mask), Some(1.))
        .unwrap();
    let mut outputs = vec![full.clone()];
    for (start, end) in [(0, 2), (2, 5), (4, 5)] {
        let chunk = q.slice(&[start, 0], &[end, 2], &[1, 1]).unwrap();
        let mask = g.causal_attention_mask(end - start, 7, start).unwrap();
        outputs.push(
            chunk
                .scaled_dot_product_attention(&k, &v, Some(&mask), Some(1.))
                .unwrap(),
        );
    }
    outputs.push(
        full.sum(&[0, 1], false)
            .unwrap()
            .grad(&[v])
            .unwrap()
            .remove(0),
    );
    outputs.push(g.causal_attention_mask(2, 4, 1).unwrap());
    let exe = g.compile_many(&client, &outputs).unwrap();
    let qv: Vec<_> = (0..10).map(|i| (i as f32 - 3.) * 0.125).collect();
    let kv: Vec<_> = (0..14).map(|i| (i as f32 - 6.) * 0.125).collect();
    let vv: Vec<_> = (0..14).map(|i| (i as f32 - 5.) * 0.25).collect();
    let qb = client.buffer(&[5, 2], &qv).unwrap();
    let kb = client.buffer(&[7, 2], &kv).unwrap();
    let vb = client.buffer(&[7, 2], &vv).unwrap();
    let out = exe.execute(&[&qb, &kb, &vb]).unwrap();
    let full = out[0].to_vec::<f32>().unwrap();
    for (output, start, end) in [(1, 0, 2), (2, 2, 5), (3, 4, 5)] {
        for (a, b) in out[output]
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(&full[start * 2..end * 2])
        {
            assert!((a - b).abs() < 2e-5);
        }
    }
    for i in 0..5 {
        let scores: Vec<_> = (0..=i)
            .map(|j| {
                (0..2)
                    .map(|d| f64::from(qv[i * 2 + d]) * f64::from(kv[j * 2 + d]))
                    .sum::<f64>()
                    .exp()
            })
            .collect();
        let sum: f64 = scores.iter().sum();
        for d in 0..2 {
            let reference: f64 = scores
                .iter()
                .enumerate()
                .map(|(j, s)| s / sum * f64::from(vv[j * 2 + d]))
                .sum();
            assert!((f64::from(full[i * 2 + d]) - reference).abs() < 2e-5);
        }
    }
    assert_eq!(&out[4].to_vec::<f32>().unwrap()[10..], &[0.; 4]);
    assert_eq!(
        out[5].to_vec::<f32>().unwrap(),
        [
            0.,
            0.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.,
            0.,
            0.,
            f32::NEG_INFINITY
        ]
    );
}
