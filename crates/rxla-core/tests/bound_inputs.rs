use rxla_core::{CacheLimits, Client, Compiler, StateGraph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_many_interleaved_fixed_inputs() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let mut initial = Vec::new();
    let mut bindings = Vec::new();
    let mut sum = graph.constant(&[], &[0.]).unwrap();
    for index in 0..64 {
        let slot = graph.state(&[]).unwrap();
        initial.push((slot, client.buffer(&[], &[-1.]).unwrap()));
        let input = graph.input(&[]).unwrap();
        sum = sum.add(&input).unwrap();
        bindings.push((index, client.buffer(&[], &[index as f32]).unwrap()));
    }
    let program = graph.compile(&mut compiler, &[sum]).unwrap();
    let mut session = program.session(initial).unwrap();
    bindings.reverse();
    session.bind_inputs(bindings.clone()).unwrap();
    assert_eq!(session.input_count(), 0);
    assert_eq!(
        session.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(),
        [2016.]
    );
    // Leave only index 37 dynamic; the input list must not include hidden state.
    bindings.retain(|(index, _)| *index != 37);
    session.bind_inputs(bindings).unwrap();
    assert_eq!(session.input_count(), 1);
    let replacement = client.buffer(&[], &[100.]).unwrap();
    assert_eq!(
        session.run(&[&replacement]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [2079.]
    );
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_fixed_inputs_are_shared_validated_and_rebindable() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let foreign = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let x = graph.input(&[]).unwrap();
    let state = graph.state(&[]).unwrap(); // Not part of visible binding indices.
    let weight = graph.input(&[2]).unwrap();
    let position = graph.input_i32_scalar().unwrap();
    let next = graph
        .read(&state)
        .unwrap()
        .add(&x.mul(&weight.take(&position, 0).unwrap()).unwrap())
        .unwrap();
    graph.write(&state, &next).unwrap();
    let program = graph.compile(&mut compiler, &[next]).unwrap();
    let mut first = program
        .session(vec![(state.clone(), client.buffer(&[], &[0.]).unwrap())])
        .unwrap();
    let mut second = program
        .session(vec![(state.clone(), client.buffer(&[], &[100.]).unwrap())])
        .unwrap();
    let weight = client.buffer(&[2], &[2., 3.]).unwrap();
    let one = client.buffer(&[], &[1.]).unwrap();
    let position = client.buffer(&[], &[1]).unwrap();
    assert_eq!(first.input_count(), 3);
    first
        .bind_inputs(vec![(2, position.clone()), (1, weight.clone())])
        .unwrap();
    second.bind_inputs(vec![(1, weight.clone())]).unwrap();
    assert_eq!(first.input_count(), 1);
    assert_eq!(second.input_count(), 2);
    let invalid = vec![
        vec![(1, weight.clone()), (1, weight.clone())],
        vec![(1, weight.clone()), (3, position.clone())],
        vec![(1, one.clone())],
        vec![(2, one.clone())],
        vec![(1, foreign.buffer(&[2], &[4., 5.]).unwrap())],
    ];
    for binding in invalid {
        assert!(first.bind_inputs(binding).is_err());
        assert_eq!(first.input_count(), 1);
        assert_eq!(first.state(&state).unwrap().to_vec::<f32>().unwrap(), [0.]);
    }
    assert!(first.run(&[]).is_err());
    assert_eq!(
        first.run(&[&one]).unwrap()[0].to_vec::<f32>().unwrap(),
        [3.]
    );
    let zero_position = client.buffer(&[], &[0]).unwrap();
    assert_eq!(
        second.run(&[&one, &zero_position]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [102.]
    );
    assert_eq!(first.state(&state).unwrap().to_vec::<f32>().unwrap(), [3.]);
    // Rebinding values does not specialize/compile a different graph.
    first
        .bind_inputs(vec![
            (0, one.clone()),
            (1, client.buffer(&[2], &[4., 5.]).unwrap()),
            (2, position.clone()),
        ])
        .unwrap();
    assert_eq!(first.input_count(), 0);
    assert_eq!(first.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(), [8.]);
    // Replacing mutable state preserves the fixed inputs.
    first
        .replace_state(vec![(state.clone(), client.buffer(&[], &[0.]).unwrap())])
        .unwrap();
    assert_eq!(first.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(), [5.]);
    first.bind_inputs(vec![]).unwrap();
    assert_eq!(first.input_count(), 3);
    assert_eq!(
        first.run(&[&one, &weight, &zero_position]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [7.]
    );
    assert_eq!(compiler.stats().misses, 1);
    // A session owns its bindings even after all caller-side owners disappear.
    drop(weight);
    drop(program);
    drop(compiler);
    drop(client);
    assert_eq!(
        second.run(&[&one, &zero_position]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [104.]
    );
}
