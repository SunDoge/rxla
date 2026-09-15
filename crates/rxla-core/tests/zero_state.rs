use rxla_core::{CacheLimits, Client, Compiler, StateGraph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_zero_state_validates_layout_and_initializes_independent_sessions() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client, CacheLimits::default());
    let mut graph = StateGraph::default();
    let value = graph.state(&[2]).unwrap();
    let position = graph.state_i32(&[]).unwrap();
    let empty = graph.state(&[0, 2]).unwrap();
    let next = graph.read(&value).unwrap().add_scalar(2.).unwrap();
    graph.write(&value, &next).unwrap();
    let next = graph
        .read(&position)
        .unwrap()
        .wrapping_add_scalar(1)
        .unwrap();
    graph.write(&position, &next).unwrap();
    let plan = graph.compile(&mut compiler, &[]).unwrap();
    // Arbitrary order must be preserved, including scalar and empty shapes.
    let slots = [empty.clone(), position.clone(), value.clone()];
    let mut parent = plan.session(plan.zero_state(&slots).unwrap()).unwrap();
    parent.run(&[]).unwrap();
    let mut child = parent
        .new_session(plan.zero_state(&slots).unwrap())
        .unwrap();
    assert_eq!(
        child.state(&value).unwrap().to_vec::<f32>().unwrap(),
        [0.; 2]
    );
    assert_eq!(
        child.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [0]
    );
    assert!(
        child
            .state(&empty)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .is_empty()
    );
    let foreign = StateGraph::default().state(&[2]).unwrap();
    let late = graph.state(&[2]).unwrap();
    assert!(plan.zero_state(&[]).is_err());
    assert!(
        plan.zero_state(&[empty.clone(), position.clone(), position.clone()])
            .is_err()
    );
    assert!(
        plan.zero_state(&[empty.clone(), position.clone(), foreign])
            .is_err()
    );
    assert!(plan.zero_state(&[empty, position.clone(), late]).is_err());
    assert_eq!(
        parent.state(&value).unwrap().to_vec::<f32>().unwrap(),
        [2.; 2]
    );
    assert_eq!(
        parent.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [1]
    );
    child.run(&[]).unwrap();
    child.run(&[]).unwrap();
    assert_eq!(
        child.state(&value).unwrap().to_vec::<f32>().unwrap(),
        [4.; 2]
    );
    assert_eq!(
        parent.state(&value).unwrap().to_vec::<f32>().unwrap(),
        [2.; 2]
    );
    assert_eq!(compiler.stats().misses, 1);
}
