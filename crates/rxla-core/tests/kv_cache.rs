use rxla_core::{CacheLimits, Client, Compiler, KvCache, StateGraph};
use std::sync::Arc;

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_new_session_shares_weights_but_not_cache_or_rebindings() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let mut cache = KvCache::new(&mut graph, &[3, 1]).unwrap();
    let position = graph.state_i32(&[]).unwrap();
    // Put a fixed parameter between dynamic inputs to check ABI ordering.
    let key = graph.input(&[1, 1]).unwrap();
    let weight = graph.parameter(&[1, 1]).unwrap();
    let value = graph.input(&[1, 1]).unwrap();
    let scaled = value.mul(weight.tensor()).unwrap();
    let old_position = graph.read(&position).unwrap();
    let zero = graph.scalar_i32(0).unwrap();
    cache
        .update_at(&mut graph, &key, &scaled, &[old_position.clone(), zero])
        .unwrap();
    graph
        .write(&position, &old_position.wrapping_add_scalar(1).unwrap())
        .unwrap();
    let program = graph.compile(&mut compiler, &[]).unwrap();
    let fresh = || {
        vec![
            (position.clone(), client.buffer(&[], &[0]).unwrap()),
            (
                cache.value_slot().clone(),
                client.buffer(&[3, 1], &[0.; 3]).unwrap(),
            ),
            (
                cache.key_slot().clone(),
                client.buffer(&[3, 1], &[0.; 3]).unwrap(),
            ),
        ]
    };
    let shared = Arc::new(client.buffer(&[1, 1], &[10.]).unwrap());
    let mut a = program.session(fresh()).unwrap();
    a.bind_parameters(vec![(weight.clone(), shared.clone())])
        .unwrap();
    let run = |session: &mut rxla_core::Session, k, v| {
        let kb = client.buffer(&[1, 1], &[k]).unwrap();
        let vb = client.buffer(&[1, 1], &[v]).unwrap();
        assert!(session.run(&[&kb, &vb]).unwrap().is_empty());
    };
    run(&mut a, 1., 2.);
    let mut b = a.new_session(fresh()).unwrap();
    assert_eq!(Arc::strong_count(&shared), 3);
    assert!(std::ptr::eq(
        a.parameter(&weight).unwrap(),
        b.parameter(&weight).unwrap()
    ));
    assert_eq!(b.input_count(), 2);
    assert_eq!(b.state(&position).unwrap().to_vec::<i32>().unwrap(), [0]);
    assert_eq!(
        b.state(cache.value_slot())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [0.; 3]
    );

    // Invalid child initialization neither mutates the parent nor retains weights.
    assert!(a.new_session(vec![]).is_err());
    let mut invalid = fresh();
    invalid[0].1 = client.buffer(&[], &[0.]).unwrap();
    assert!(a.new_session(invalid).is_err());
    assert_eq!(Arc::strong_count(&shared), 3);
    assert_eq!(a.state(&position).unwrap().to_vec::<i32>().unwrap(), [1]);

    a.bind_parameters(vec![(
        weight.clone(),
        Arc::new(client.buffer(&[1, 1], &[20.]).unwrap()),
    )])
    .unwrap();
    assert_eq!(Arc::strong_count(&shared), 2);
    run(&mut b, 7., 3.);
    run(&mut a, 2., 4.);
    run(&mut b, 8., 5.);
    assert_eq!(
        a.state(cache.key_slot()).unwrap().to_vec::<f32>().unwrap(),
        [1., 2., 0.]
    );
    assert_eq!(
        a.state(cache.value_slot())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [20., 80., 0.]
    );
    assert_eq!(
        b.state(cache.key_slot()).unwrap().to_vec::<f32>().unwrap(),
        [7., 8., 0.]
    );
    assert_eq!(
        b.state(cache.value_slot())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [30., 50., 0.]
    );
    drop((a, program, graph, shared));
    // The child owns the shared code and bindings independently of its parent.
    run(&mut b, 9., 6.);
    assert_eq!(b.state(&position).unwrap().to_vec::<i32>().unwrap(), [3]);
    assert_eq!(
        b.state(cache.value_slot())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [30., 50., 60.]
    );
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_paired_cache_is_session_local_and_clamps_positions() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let mut cache = KvCache::new(&mut graph, &[3, 1]).unwrap();
    let k = graph.input(&[1, 1]).unwrap();
    let v = graph.input(&[1, 1]).unwrap();
    let position = graph.input_i32_scalar().unwrap();
    let zero = graph.scalar_i32(0).unwrap();
    cache
        .update_at(&mut graph, &k, &v, &[position, zero])
        .unwrap();
    // No visible outputs: paired updates must survive dead-code elimination.
    let program = graph.compile(&mut compiler, &[]).unwrap();
    let make_session = || {
        program
            .session(vec![
                (
                    cache.key_slot().clone(),
                    client.buffer(&[3, 1], &[0.; 3]).unwrap(),
                ),
                (
                    cache.value_slot().clone(),
                    client.buffer(&[3, 1], &[0.; 3]).unwrap(),
                ),
            ])
            .unwrap()
    };
    let mut a = make_session();
    let b = make_session();
    drop(graph);
    for (position, key, value) in [(-5, 1., 10.), (99, 3., 30.), (1, 2., 20.)] {
        let kb = client.buffer(&[1, 1], &[key]).unwrap();
        let vb = client.buffer(&[1, 1], &[value]).unwrap();
        let pb = client.buffer(&[], &[position]).unwrap();
        assert!(a.run(&[&kb, &vb, &pb]).unwrap().is_empty());
    }
    let bad = client.buffer(&[1], &[999.]).unwrap();
    let good = client.buffer(&[1, 1], &[999.]).unwrap();
    let pos = client.buffer(&[], &[0]).unwrap();
    assert!(a.run(&[&good, &bad, &pos]).is_err());
    assert_eq!(
        a.state(cache.key_slot()).unwrap().to_vec::<f32>().unwrap(),
        [1., 2., 3.]
    );
    assert_eq!(
        a.state(cache.value_slot())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [10., 20., 30.]
    );
    for slot in [cache.key_slot(), cache.value_slot()] {
        assert_eq!(b.state(slot).unwrap().to_vec::<f32>().unwrap(), [0.; 3]);
    }
    assert_eq!(compiler.stats().misses, 1);
}
