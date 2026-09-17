use rxla_core::{CacheLimits, Client, Compiler, StateGraph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_pruned_inputs_preserve_identities_hidden_updates_and_state_transfer() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let x = graph.input(&[]).unwrap(); // original visible input 0
    let dead = graph.parameter(&[3]).unwrap(); // 1
    let count = graph.state_i32(&[]).unwrap();
    let weight = graph.parameter(&[]).unwrap(); // 2, interleaved with state
    let next_count = graph.input_i32_scalar().unwrap(); // 3, hidden-state-only use
    let _dead_tail = graph.input(&[2, 0]).unwrap(); // 4
    let constant_state = graph.state(&[]).unwrap();
    graph.write(&count, &next_count).unwrap();
    graph
        .write(&constant_state, &graph.constant(&[], &[7.]).unwrap())
        .unwrap();
    let y = x.mul(weight.tensor()).unwrap();
    let outputs = [y, graph.scalar_i32(9).unwrap()];
    let pruned = graph.compile_pruned(&mut compiler, &outputs).unwrap();
    let cached = graph.compile_pruned(&mut compiler, &outputs).unwrap();
    let legacy = graph.compile(&mut compiler, &outputs).unwrap();
    assert_eq!(pruned.input_indices(), [0, 2, 3]);
    assert_eq!(cached.input_indices(), [0, 2, 3]);
    assert_eq!(legacy.input_indices(), [0, 1, 2, 3, 4]);
    let fresh = || {
        vec![
            (constant_state.clone(), client.buffer(&[], &[99.]).unwrap()),
            (count.clone(), client.buffer(&[], &[16_777_217]).unwrap()),
        ]
    };
    assert!(pruned.session(vec![]).is_err()); // old state inputs pruned, schema intact
    let mut session = pruned.session(fresh()).unwrap();
    assert_eq!(session.input_count(), 3);
    let shared = client.buffer(&[], &[2.]).unwrap();
    session
        .bind_parameters(vec![(weight.clone(), shared.clone())])
        .unwrap();
    assert_eq!(session.input_count(), 2);
    assert!(session.parameter(&dead).is_err());
    assert!(
        session
            .bind_parameters(vec![(dead.clone(), client.buffer(&[3], &[0.; 3]).unwrap())])
            .is_err()
    );
    assert_eq!(session.input_count(), 2);
    assert!(
        session
            .parameter(&weight)
            .unwrap()
            .shares_allocation_with(&shared)
    );
    let mut child = session.new_session(fresh()).unwrap();
    let xb = client.buffer(&[], &[3.]).unwrap();
    let ib = client.buffer(&[], &[17]).unwrap();
    assert!(session.run(&[&xb]).is_err());
    assert_eq!(
        session.state(&count).unwrap().to_vec::<i32>().unwrap(),
        [16_777_217]
    );
    let result = session.run(&[&xb, &ib]).unwrap();
    assert_eq!(result[0].to_vec::<f32>().unwrap(), [6.]);
    assert_eq!(result[1].to_vec::<i32>().unwrap(), [9]);
    assert_eq!(
        session.state(&count).unwrap().to_vec::<i32>().unwrap(),
        [17]
    );
    assert_eq!(
        session
            .state(&constant_state)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [7.]
    );
    assert_eq!(
        child.state(&count).unwrap().to_vec::<i32>().unwrap(),
        [16_777_217]
    );
    child.bind_inputs(vec![]).unwrap();
    assert_eq!(child.input_count(), 3);
    child.bind_inputs(vec![(2, shared.clone())]).unwrap(); // original, NOT compact index
    assert_eq!(
        child.run(&[&xb, &ib]).unwrap()[0].to_vec::<f32>().unwrap(),
        [6.]
    );
    let mut resumed = legacy.session(session.into_state()).unwrap();
    resumed.bind_parameters(vec![(weight, shared)]).unwrap();
    let dead_buffer = client.buffer(&[3], &[0.; 3]).unwrap();
    let tail_buffer = client.buffer::<f32>(&[2, 0], &[]).unwrap();
    assert_eq!(
        resumed
            .run(&[&xb, &dead_buffer, &ib, &tail_buffer])
            .unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [6.]
    );
    assert_eq!(compiler.stats().misses, 2);
    assert_eq!(compiler.stats().hits, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_pruned_constant_program_has_no_inputs() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client, CacheLimits::default());
    let mut graph = StateGraph::default();
    let _unused = graph.input(&[100]).unwrap();
    let value = graph.constant(&[], &[5.]).unwrap();
    let program = graph.compile_pruned(&mut compiler, &[value]).unwrap();
    assert!(program.input_indices().is_empty());
    let mut session = program.session(vec![]).unwrap();
    assert_eq!(session.input_count(), 0);
    session.bind_inputs(vec![]).unwrap();
    assert_eq!(session.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(), [5.]);
    let clean = StateGraph::default();
    let same = clean.constant(&[], &[5.]).unwrap();
    let equivalent = clean
        .compile_pruned(&mut compiler, std::slice::from_ref(&same))
        .unwrap();
    assert_eq!(
        equivalent.session(vec![]).unwrap().run(&[]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [5.]
    );
    assert!(graph.compile_pruned(&mut compiler, &[same]).is_err());
    assert_eq!(compiler.stats().misses, 1);
    assert_eq!(compiler.stats().hits, 1);
}
