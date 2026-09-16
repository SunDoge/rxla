use rxla_core::{CacheLimits, Client, Compiler, StateGraph};
use std::rc::Rc;

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_session_state_transfer_and_rebinding() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let count = graph.state_i32(&[]).unwrap();
    let total = graph.state(&[]).unwrap();
    let weight = graph.parameter(&[]).unwrap();
    let next = graph.read(&count).unwrap().wrapping_add_scalar(1).unwrap();
    let sum = graph.read(&total).unwrap().add(weight.tensor()).unwrap();
    graph
        .write_many(&[(&count, &next), (&total, &sum)])
        .unwrap();
    let program = graph.compile(&mut compiler, &[sum]).unwrap();
    let shared = Rc::new(client.buffer(&[], &[2.]).unwrap());
    let mut session = program
        .session(vec![
            (total.clone(), client.buffer(&[], &[1.]).unwrap()),
            (count.clone(), client.buffer(&[], &[16_777_217]).unwrap()),
        ])
        .unwrap();
    session
        .bind_parameters(vec![(weight.clone(), shared.clone())])
        .unwrap();
    let first = session.run(&[]).unwrap();
    assert_eq!(Rc::strong_count(&shared), 2);
    let saved = session.into_state();
    assert_eq!(Rc::strong_count(&shared), 1);
    assert_eq!(saved.len(), 2);
    assert_eq!(saved[0].1.to_vec::<i32>().unwrap(), [16_777_218]);
    assert_eq!(saved[1].1.to_vec::<f32>().unwrap(), [3.]);
    let mut resumed = program.session(saved).unwrap();
    assert_eq!(resumed.input_count(), 1); // Fixed bindings are not state.
    assert!(resumed.run(&[]).is_err());
    assert_eq!(
        resumed.state(&count).unwrap().to_vec::<i32>().unwrap(),
        [16_777_218]
    );
    resumed.bind_parameters(vec![(weight, shared)]).unwrap();
    assert_eq!(resumed.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(), [5.]);
    assert_eq!(first[0].to_vec::<f32>().unwrap(), [3.]);
    assert_eq!(compiler.stats().misses, 1);
    let saved = resumed.into_state();
    drop((program, graph, compiler, client));
    assert_eq!(saved[0].1.to_vec::<i32>().unwrap(), [16_777_219]);
    assert_eq!(saved[1].1.to_vec::<f32>().unwrap(), [5.]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_empty_state_transfer() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client, CacheLimits::default());
    let graph = StateGraph::default();
    let value = graph.constant(&[], &[7.]).unwrap();
    let program = graph.compile(&mut compiler, &[value]).unwrap();
    let saved = program.session(vec![]).unwrap().into_state();
    assert!(saved.is_empty());
    let mut resumed = program.session(saved).unwrap();
    assert_eq!(resumed.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(), [7.]);
}
