use rxla_core::{CacheLimits, Client, Compiler, KvCache, StateGraph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_checked_chunk_updates_preserve_pair_on_invalid_positions() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let mut cache = KvCache::new(&mut graph, &[4, 2]).unwrap();
    let k = graph.input(&[2, 2]).unwrap();
    let v = graph.input(&[2, 2]).unwrap();
    let row = graph.input_i32_scalar().unwrap();
    let col = graph.input_i32_scalar().unwrap();
    let update = cache
        .update_at_checked(&mut graph, &k, &v, &[row, col])
        .unwrap();
    let grad = update
        .keys
        .sum(&[0, 1], false)
        .unwrap()
        .grad(&[k])
        .unwrap()
        .remove(0);
    let plan = graph
        .compile(
            &mut compiler,
            &[update.accepted, update.keys, update.values, grad],
        )
        .unwrap();
    let mut session = plan
        .session(vec![
            (
                cache.key_slot().clone(),
                client.buffer(&[4, 2], &[0.; 8]).unwrap(),
            ),
            (
                cache.value_slot().clone(),
                client.buffer(&[4, 2], &[0.; 8]).unwrap(),
            ),
        ])
        .unwrap();
    let mut expected_k = [0.; 8];
    let mut expected_v = [0.; 8];
    for (step, (row, col, valid)) in [
        (0, 0, true),
        (-1, 0, false),
        (3, 0, false),
        (2, 0, true),
        (1, 1, false),
        (i32::MAX, 0, false),
        (1, 0, true),
    ]
    .into_iter()
    .enumerate()
    {
        let data = if valid {
            [step as f32 + 1.; 4]
        } else {
            [f32::NAN; 4]
        };
        let values = if valid {
            data.map(|x| -x)
        } else {
            [f32::INFINITY; 4]
        };
        let kb = client.buffer(&[2, 2], &data).unwrap();
        let vb = client.buffer(&[2, 2], &values).unwrap();
        let rb = client.buffer(&[], &[row]).unwrap();
        let cb = client.buffer(&[], &[col]).unwrap();
        let outputs = session.run(&[&kb, &vb, &rb, &cb]).unwrap();
        if valid {
            expected_k[row as usize * 2..row as usize * 2 + 4].copy_from_slice(&data);
            expected_v[row as usize * 2..row as usize * 2 + 4].copy_from_slice(&values);
        }
        assert_eq!(
            outputs[0].to_vec::<f32>().unwrap(),
            [if valid { 1. } else { 0. }]
        );
        assert_eq!(outputs[1].to_vec::<f32>().unwrap(), expected_k);
        assert_eq!(outputs[2].to_vec::<f32>().unwrap(), expected_v);
        assert_eq!(
            outputs[3].to_vec::<f32>().unwrap(),
            [if valid { 1. } else { 0. }; 4]
        );
        assert_eq!(
            session
                .state(cache.key_slot())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            expected_k
        );
        assert_eq!(
            session
                .state(cache.value_slot())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            expected_v
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}
