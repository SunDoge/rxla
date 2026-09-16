use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tensor, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_state_program_returns_ordered_mixed_outputs_without_exposing_hidden_roots() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let count = graph.state_i32(&[]).unwrap();
    let total = graph.state(&[]).unwrap();
    let old_count = graph.read(&count).unwrap();
    let next_count = old_count.wrapping_add_scalar(1).unwrap();
    let next_total = graph.read(&total).unwrap().add_scalar(2.).unwrap();
    graph
        .write_outputs(&[(&count, next_count.clone()), (&total, next_total.clone())])
        .unwrap();
    let visible = [old_count, next_total, next_count];
    let program = graph.compile_outputs(&mut compiler, &visible).unwrap();
    graph.compile_outputs(&mut compiler, &visible).unwrap();
    let mut session = program
        .session(vec![
            (total.clone(), client.buffer(&[], &[1.]).unwrap()),
            (count.clone(), client.buffer(&[], &[i32::MAX]).unwrap()),
        ])
        .unwrap();
    drop((graph, program));
    let first = session.run(&[]).unwrap();
    assert_eq!(first.len(), 3);
    assert_eq!(first[0].to_vec::<i32>().unwrap(), [i32::MAX]);
    assert_eq!(first[1].to_vec::<f32>().unwrap(), [3.]);
    assert_eq!(first[2].to_vec::<i32>().unwrap(), [i32::MIN]);
    let second = session.run(&[]).unwrap();
    assert_eq!(second[0].to_vec::<i32>().unwrap(), [i32::MIN]);
    assert_eq!(second[1].to_vec::<f32>().unwrap(), [5.]);
    assert_eq!(second[2].to_vec::<i32>().unwrap(), [i32::MIN + 1]);
    assert_eq!(
        session.state(&count).unwrap().to_vec::<i32>().unwrap(),
        [i32::MIN + 1]
    );
    assert_eq!(
        session.state(&total).unwrap().to_vec::<f32>().unwrap(),
        [5.]
    );
    // Previously returned snapshots remain valid after later state transitions.
    assert_eq!(first[2].to_vec::<i32>().unwrap(), [i32::MIN]);
    assert_eq!(compiler.stats().misses, 1);
    assert_eq!(compiler.stats().hits, 1);
}

#[test]
fn mixed_output_graph_ownership_and_slot_types() {
    let mut g = StateGraph::default();
    let f = g.state(&[]).unwrap();
    let i = g.state_i32(&[]).unwrap();
    assert_eq!(g.read(&i).unwrap().dtype(), rxla_core::DType::I32);
    assert_eq!(g.read(&f).unwrap().dtype(), rxla_core::DType::F32);
    assert!(g.write(&f, &g.scalar_i32(3).unwrap()).is_err());
    assert!(g.write(&i, &g.constant(&[], &[3.]).unwrap()).is_err());
    assert!(g.write(&i, &g.constant_i32(&[1], &[3]).unwrap()).is_err());
    let graph = Tracer::default();
    let foreign: Tensor = Tracer::default().scalar_i32(1).unwrap();
    assert!(graph.stablehlo_many(&[foreign]).is_err());
    assert!(graph.stablehlo_many(&[]).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_mixed_state_position_updates_and_replacement_are_atomic() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let cache = g.state(&[4]).unwrap();
    let value = g.input(&[1]).unwrap();
    let position = g.state_i32(&[]).unwrap();
    let old_position = g.read(&position).unwrap();
    let updated = g
        .read(&cache)
        .unwrap()
        .dynamic_update_slice(&value, std::slice::from_ref(&old_position))
        .unwrap();
    let next = old_position.wrapping_add_scalar(1).unwrap();
    g.write_outputs(&[(&cache, updated), (&position, next)])
        .unwrap();
    let program = g.compile(&mut compiler, &[]).unwrap(); // Both state roots hidden.
    let initial = || {
        vec![
            (position.clone(), client.buffer(&[], &[0]).unwrap()),
            (cache.clone(), client.buffer(&[4], &[0.; 4]).unwrap()),
        ]
    };
    let mut session = program.session(initial()).unwrap();
    let independent = program.session(initial()).unwrap();
    assert_eq!(session.input_count(), 1); // No host position argument.
    drop(g);
    for value in [10., 20., 30., 40.] {
        let input = client.buffer(&[1], &[value]).unwrap();
        assert!(session.run(&[&input]).unwrap().is_empty());
    }
    let invalid_input = client.buffer(&[1], &[50]).unwrap();
    assert!(session.run(&[&invalid_input]).is_err());
    assert!(
        session
            .replace_state(vec![
                (cache.clone(), client.buffer(&[4], &[99.; 4]).unwrap()),
                (position.clone(), client.buffer(&[], &[0.]).unwrap()),
            ])
            .is_err()
    );
    assert_eq!(
        session.state(&cache).unwrap().to_vec::<f32>().unwrap(),
        [10., 20., 30., 40.]
    );
    assert_eq!(
        session.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [4]
    );
    assert_eq!(
        independent
            .state(&position)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [0]
    );
    let saved = session.replace_state(initial()).unwrap();
    assert_eq!(
        session.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [0]
    );
    session.replace_state(saved).unwrap();
    assert_eq!(
        session.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [4]
    );
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_mixed_outputs_survive_executable_restoration() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let g = Tracer::default();
    let indices = g.input_i32(&[2]).unwrap();
    let x = g.input(&[2]).unwrap();
    let outputs = [
        indices.wrapping_add_scalar(1).unwrap(),
        x.add_scalar(2.).unwrap(),
    ];
    let exe = compiler.compile_many(&g, &outputs).unwrap();
    compiler.compile_many(&g, &outputs).unwrap();
    let bytes = exe.serialize_with_metadata().unwrap();
    let restored =
        unsafe { rxla_core::Executable::deserialize_with_metadata(&client, &bytes) }.unwrap();
    let i = client.buffer(&[2], &[i32::MAX, 16_777_217]).unwrap();
    let f = client.buffer(&[2], &[3., 4.]).unwrap();
    drop((g, exe));
    let outputs = restored.execute(&[&i, &f]).unwrap();
    assert_eq!(outputs[0].to_vec::<i32>().unwrap(), [i32::MIN, 16_777_218]);
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [5., 6.]);
    assert_eq!(compiler.stats().misses, 1);
    assert_eq!(compiler.stats().hits, 1);
}
