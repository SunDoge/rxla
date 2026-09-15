use rxla_core::{CacheLimits, Client, Compiler, StateGraph};
use std::rc::Rc;

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_parameter_identity_binding_and_replacement() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let a = graph.parameter(&[]).unwrap();
    let state = graph.state(&[]).unwrap();
    let x = graph.input(&[]).unwrap();
    let b = graph.parameter(&[]).unwrap();
    let tied = a.clone();
    let next = x
        .mul(a.tensor())
        .unwrap()
        .add(b.tensor())
        .unwrap()
        .add(tied.tensor())
        .unwrap()
        .add(&graph.read(&state).unwrap())
        .unwrap();
    graph.write(&state, &next).unwrap();
    let program = graph.compile(&mut compiler, &[next]).unwrap();
    // Same owner but registered after compilation: not a valid plan parameter.
    let late = graph.parameter(&[]).unwrap();
    let mut other = StateGraph::default();
    let foreign = other.parameter(&[]).unwrap();
    let make_session = || {
        program
            .session(vec![(state.clone(), client.buffer(&[], &[0.]).unwrap())])
            .unwrap()
    };
    let mut first = make_session();
    let mut second = make_session();
    let two = Rc::new(client.buffer(&[], &[2.]).unwrap());
    let three = Rc::new(client.buffer(&[], &[3.]).unwrap());
    first
        .bind_parameters(vec![(b.clone(), three.clone()), (a.clone(), two.clone())])
        .unwrap();
    second
        .bind_parameters(vec![(a.clone(), two.clone()), (b.clone(), three.clone())])
        .unwrap();
    assert_eq!(first.input_count(), 1);
    let wrong = Rc::new(client.buffer(&[1], &[2.]).unwrap());
    let integer = Rc::new(client.buffer(&[], &[2]).unwrap());
    let foreign_client =
        unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let foreign_buffer = Rc::new(foreign_client.buffer(&[], &[2.]).unwrap());
    for bindings in [
        vec![(a.clone(), two.clone()), (tied, three.clone())],
        vec![(a.clone(), two.clone()), (foreign, three.clone())],
        vec![(late, two.clone())],
        vec![(a.clone(), wrong)],
        vec![(a.clone(), integer)],
        vec![(a.clone(), foreign_buffer)],
    ] {
        assert!(first.bind_parameters(bindings).is_err());
        assert_eq!(first.input_count(), 1);
        assert_eq!(first.state(&state).unwrap().to_vec::<f32>().unwrap(), [0.]);
    }
    let four = client.buffer(&[], &[4.]).unwrap();
    assert_eq!(
        first.run(&[&four]).unwrap()[0].to_vec::<f32>().unwrap(),
        [13.]
    );
    assert_eq!(second.state(&state).unwrap().to_vec::<f32>().unwrap(), [0.]);
    // Full replacement, not merging: b becomes dynamic again, after x.
    first
        .bind_parameters(vec![(a.clone(), three.clone())])
        .unwrap();
    assert_eq!(first.input_count(), 2);
    assert_eq!(
        first.run(&[&four, &two]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [30.]
    );
    first.bind_parameters(vec![]).unwrap();
    assert_eq!(first.input_count(), 3);
    assert_eq!(
        first.run(&[&two, &four, &three]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [43.]
    );
    drop((a, b, graph, program, two, three));
    assert_eq!(
        second.run(&[&four]).unwrap()[0].to_vec::<f32>().unwrap(),
        [13.]
    );
    assert_eq!(compiler.stats().misses, 1);
}
