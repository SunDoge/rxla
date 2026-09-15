//! Fixed-size attention chunks with resident KV/position and checked exhaustion.
//! Not a full-model prefill implementation or a performance benchmark.
use rxla_core::{CacheLimits, Client, Compiler, KvCache, StateGraph};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?)? };
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let q = graph.input(&[2, 2])?;
    let k = graph.input(&[2, 2])?;
    let v = graph.input(&[2, 2])?;
    let mut cache = KvCache::new(&mut graph, &[6, 2])?;
    let position_slot = graph.state_i32(&[])?;
    let position = graph.read(&position_slot)?;
    let starts = [position.clone(), graph.scalar_i32(0)?];
    let update = cache.update_at_checked(&mut graph, &k, &v, &starts)?;
    let next_position = update
        .accepted
        .select(&position.wrapping_add_scalar(2)?, &position)?;
    graph.write(&position_slot, &next_position)?;
    let positions = graph
        .iota_i32(&[2], 0)?
        .wrapping_add(&position.broadcast_to(&[2])?)?;
    let mask = positions.causal_attention_mask(&graph.iota_i32(&[6], 0)?)?;
    let attention =
        q.scaled_dot_product_attention(&update.keys, &update.values, Some(&mask), Some(1.))?;
    // Rejected calls expose zeros, not unusable attention results. This is a
    // graph select, not lazy execution or a native cancellation mechanism.
    let output = update
        .accepted
        .broadcast_to(&[2, 2])?
        .select(&attention, &graph.constant(&[2, 2], &[0.; 4])?)?;
    let program = graph.compile(&mut compiler, &[output, update.accepted])?;
    let mut session = program.session(vec![
        (cache.key_slot().clone(), client.buffer(&[6, 2], &[0.; 12])?),
        (
            cache.value_slot().clone(),
            client.buffer(&[6, 2], &[0.; 12])?,
        ),
        (position_slot.clone(), client.buffer(&[], &[0])?),
    ])?;
    let queries: Vec<_> = (0..12).map(|i| (i as f32 - 4.) * 0.125).collect();
    let keys: Vec<_> = (0..12).map(|i| (i as f32 - 6.) * 0.125).collect();
    let values: Vec<_> = (0..12).map(|i| (i as f32 - 3.) * 0.25).collect();
    for chunk in 0..3 {
        let start = chunk * 4;
        let qb = client.buffer(&[2, 2], &queries[start..start + 4])?;
        let kb = client.buffer(&[2, 2], &keys[start..start + 4])?;
        let vb = client.buffer(&[2, 2], &values[start..start + 4])?;
        let result = session.run(&[&qb, &kb, &vb])?;
        assert_eq!(result[1].to_vec::<f32>()?, [1.]);
        let actual = result[0].to_vec::<f32>()?;
        // Independent full-history causal F64 reference, including both rows.
        for row in 0..2 {
            let query = chunk * 2 + row;
            let scores: Vec<_> = (0..=query)
                .map(|key| {
                    (0..2)
                        .map(|d| f64::from(queries[query * 2 + d]) * f64::from(keys[key * 2 + d]))
                        .sum::<f64>()
                        .exp()
                })
                .collect();
            let sum: f64 = scores.iter().sum();
            for d in 0..2 {
                let expected: f64 = scores
                    .iter()
                    .enumerate()
                    .map(|(j, s)| s / sum * f64::from(values[j * 2 + d]))
                    .sum();
                assert!((f64::from(actual[row * 2 + d]) - expected).abs() < 2e-5);
            }
        }
        assert_eq!(
            session.state(&position_slot)?.to_vec::<i32>()?,
            [(chunk as i32 + 1) * 2]
        );
    }
    assert_eq!(session.state(cache.key_slot())?.to_vec::<f32>()?, keys);
    assert_eq!(session.state(cache.value_slot())?.to_vec::<f32>()?, values);
    let invalid = client.buffer(&[2, 2], &[f32::NAN; 4])?;
    for _ in 0..2 {
        let result = session.run(&[&invalid, &invalid, &invalid])?;
        assert_eq!(result[1].to_vec::<f32>()?, [0.]);
        assert_eq!(result[0].to_vec::<f32>()?, [0.; 4]);
        assert_eq!(session.state(&position_slot)?.to_vec::<i32>()?, [6]);
        assert_eq!(session.state(cache.key_slot())?.to_vec::<f32>()?, keys);
        assert_eq!(session.state(cache.value_slot())?.to_vec::<f32>()?, values);
    }
    assert_eq!(compiler.stats().misses, 1);
    println!(
        "PASS: three resident attention chunks match F64, exhaustion preserves KV/position, one compilation"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_resident_chunked_attention() {
    run().unwrap();
}
