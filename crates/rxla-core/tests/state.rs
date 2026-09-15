use rxla_core::{CacheLimits, Client, Compiler, Graph, StateGraph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_batch_state_swap_is_simultaneous() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let a = graph.state(&[]).unwrap();
    let b = graph.state(&[]).unwrap();
    let untouched = graph.state(&[]).unwrap();
    let old_a = graph.read(&a).unwrap();
    let old_b = graph.read(&b).unwrap();
    graph.write_many(&[(&b, &old_a), (&a, &old_b)]).unwrap();
    let program = graph.compile(&mut compiler, &[]).unwrap();
    let mut session = program
        .session(vec![
            (a.clone(), client.buffer(&[], &[1.]).unwrap()),
            (b.clone(), client.buffer(&[], &[2.]).unwrap()),
            (untouched.clone(), client.buffer(&[], &[3.]).unwrap()),
        ])
        .unwrap();
    for (expected_a, expected_b) in [(2., 1.), (1., 2.), (2., 1.)] {
        assert!(session.run(&[]).unwrap().is_empty());
        assert_eq!(
            session.state(&a).unwrap().to_vec::<f32>().unwrap(),
            [expected_a]
        );
        assert_eq!(
            session.state(&b).unwrap().to_vec::<f32>().unwrap(),
            [expected_b]
        );
        assert_eq!(
            session.state(&untouched).unwrap().to_vec::<f32>().unwrap(),
            [3.]
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
fn symbolic_state_validation() {
    let mut graph = StateGraph::default();
    let slot = graph.state(&[2]).unwrap();
    let value = graph.read(&slot).unwrap();
    assert_eq!(value.shape(), [2]);
    assert!(
        graph
            .write(&slot, &Graph::default().input(&[2]).unwrap())
            .is_err()
    );
    let wrong_shape = graph.constant(&[1], &[1.]).unwrap();
    assert!(graph.write(&slot, &wrong_shape).is_err());
    let mut foreign = StateGraph::default();
    let foreign_slot = foreign.state(&[2]).unwrap();
    assert!(graph.read(&foreign_slot).is_err());
    assert!(graph.write(&foreign_slot, &value).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_state_sessions_and_read_after_write() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    // Interleave state and user arguments to check the parameter binding plan.
    let a = graph.state(&[]).unwrap();
    let delta = graph.input(&[]).unwrap();
    let b = graph.state(&[]).unwrap();
    let intermediate = graph.read(&a).unwrap().add(&delta).unwrap();
    graph.write(&a, &intermediate).unwrap();
    graph
        .write(&b, &graph.read(&a).unwrap().mul_scalar(2.).unwrap())
        .unwrap();
    graph
        .write(&a, &graph.read(&a).unwrap().add_scalar(1.).unwrap())
        .unwrap();
    let visible = graph.constant(&[], &[7.]).unwrap();
    let plan = graph.compile(&mut compiler, &[visible]).unwrap();
    let mut first = plan
        .session(vec![
            (b.clone(), client.buffer(&[], &[10.]).unwrap()),
            (a.clone(), client.buffer(&[], &[0.]).unwrap()),
        ])
        .unwrap();
    let second = plan
        .session(vec![
            (a.clone(), client.buffer(&[], &[100.]).unwrap()),
            (b.clone(), client.buffer(&[], &[200.]).unwrap()),
        ])
        .unwrap();
    let one = client.buffer(&[], &[1.]).unwrap();
    for step in 1..=3 {
        assert_eq!(
            first.run(&[&one]).unwrap()[0].to_vec::<f32>().unwrap(),
            [7.]
        );
        assert_eq!(
            first.state(&a).unwrap().to_vec::<f32>().unwrap(),
            [2. * step as f32]
        );
        assert_eq!(
            first.state(&b).unwrap().to_vec::<f32>().unwrap(),
            [4. * step as f32 - 2.]
        );
    }
    assert_eq!(second.state(&a).unwrap().to_vec::<f32>().unwrap(), [100.]);
    assert_eq!(second.state(&b).unwrap().to_vec::<f32>().unwrap(), [200.]);
    assert_eq!(compiler.stats().misses, 1);
    let malformed = client.buffer(&[1], &[1.]).unwrap();
    assert!(first.run(&[&malformed]).is_err());
    assert!(first.run(&[]).is_err());
    let foreign = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let foreign_buffer = foreign.buffer(&[], &[1.]).unwrap();
    assert!(first.run(&[&foreign_buffer]).is_err());
    assert_eq!(first.state(&a).unwrap().to_vec::<f32>().unwrap(), [6.]);
    assert_eq!(first.state(&b).unwrap().to_vec::<f32>().unwrap(), [10.]);
    assert!(plan.session(vec![]).is_err());
    assert!(
        plan.session(vec![
            (a.clone(), client.buffer(&[], &[1.]).unwrap()),
            (a.clone(), client.buffer(&[], &[2.]).unwrap()),
        ])
        .is_err()
    );
    drop(compiler);
    drop(client);
    drop(plan);
    assert_eq!(
        first.run(&[&one]).unwrap()[0].to_vec::<f32>().unwrap(),
        [7.]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_hidden_state_only_counter() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let counter = graph.state(&[]).unwrap();
    graph
        .write(
            &counter,
            &graph.read(&counter).unwrap().add_scalar(1.).unwrap(),
        )
        .unwrap();
    let program = graph.compile(&mut compiler, &[]).unwrap();
    let mut session = program
        .session(vec![(counter.clone(), client.buffer(&[], &[0.]).unwrap())])
        .unwrap();
    for expected in 1..=5 {
        assert!(session.run(&[]).unwrap().is_empty());
        assert_eq!(
            session.state(&counter).unwrap().to_vec::<f32>().unwrap(),
            [expected as f32]
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_state_replacement_is_atomic_and_restorable() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let foreign_client =
        unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let a = graph.state(&[]).unwrap();
    let b = graph.state(&[]).unwrap();
    graph
        .write(&a, &graph.read(&a).unwrap().add_scalar(1.).unwrap())
        .unwrap();
    graph
        .write(&b, &graph.read(&b).unwrap().add_scalar(2.).unwrap())
        .unwrap();
    let program = graph.compile(&mut compiler, &[]).unwrap();
    let scalar = |value| client.buffer(&[], &[value]).unwrap();
    let mut session = program
        .session(vec![(a.clone(), scalar(1.)), (b.clone(), scalar(10.))])
        .unwrap();
    let mut foreign_graph = StateGraph::default();
    let foreign_slot = foreign_graph.state(&[]).unwrap();
    // A valid first binding must not commit when a later binding fails.
    let invalid = vec![
        vec![],
        vec![(a.clone(), scalar(100.)), (a.clone(), scalar(200.))],
        vec![(a.clone(), scalar(100.)), (foreign_slot, scalar(200.))],
        vec![
            (a.clone(), scalar(100.)),
            (b.clone(), client.buffer(&[1], &[200.]).unwrap()),
        ],
        vec![
            (a.clone(), scalar(100.)),
            (b.clone(), client.buffer(&[], &[200]).unwrap()),
        ],
        vec![
            (a.clone(), scalar(100.)),
            (b.clone(), foreign_client.buffer(&[], &[200.]).unwrap()),
        ],
    ];
    for replacement in invalid {
        assert!(session.replace_state(replacement).is_err());
        assert_eq!(session.state(&a).unwrap().to_vec::<f32>().unwrap(), [1.]);
        assert_eq!(session.state(&b).unwrap().to_vec::<f32>().unwrap(), [10.]);
    }
    // Reversed binding order is valid; old state retains slot identities.
    let saved = session
        .replace_state(vec![(b.clone(), scalar(200.)), (a.clone(), scalar(100.))])
        .unwrap();
    session.run(&[]).unwrap();
    assert_eq!(session.state(&a).unwrap().to_vec::<f32>().unwrap(), [101.]);
    assert_eq!(session.state(&b).unwrap().to_vec::<f32>().unwrap(), [202.]);
    let suspended = session.replace_state(saved).unwrap();
    session.run(&[]).unwrap();
    assert_eq!(session.state(&a).unwrap().to_vec::<f32>().unwrap(), [2.]);
    assert_eq!(session.state(&b).unwrap().to_vec::<f32>().unwrap(), [12.]);
    let mut other = program.session(suspended).unwrap();
    other.run(&[]).unwrap();
    assert_eq!(other.state(&a).unwrap().to_vec::<f32>().unwrap(), [102.]);
    assert_eq!(other.state(&b).unwrap().to_vec::<f32>().unwrap(), [204.]);
    assert_eq!(session.state(&a).unwrap().to_vec::<f32>().unwrap(), [2.]);
    assert_eq!(compiler.stats().misses, 1);
}
