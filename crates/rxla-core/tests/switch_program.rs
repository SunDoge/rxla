use rxla_core::{CacheLimits, Client, Compiler, StateGraph, StateProgram, StateSlot};
use std::rc::Rc;

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_switch_rejects_foreign_client_without_losing_source_state() {
    let path = std::env::var("PJRT_PLUGIN_PATH").unwrap();
    let client = unsafe { Client::load(&path) }.unwrap();
    let other = unsafe { Client::load(&path) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut other_compiler = Compiler::new(other, CacheLimits::default());
    let (first, a, _) = build(&mut compiler, false);
    let (second, b, _) = build(&mut other_compiler, false);
    let mut session = first
        .session(vec![
            (a[0].clone(), client.buffer(&[], &[3.]).unwrap()),
            (a[1].clone(), client.buffer(&[], &[9]).unwrap()),
        ])
        .unwrap();
    let mapping = [(a[0].clone(), b[0].clone()), (a[1].clone(), b[1].clone())];
    assert!(session.switch_program(&second, &mapping, vec![]).is_err());
    assert_eq!(session.state(&a[0]).unwrap().to_vec::<f32>().unwrap(), [3.]);
    let output = session
        .run(&[
            &client.buffer(&[], &[2.]).unwrap(),
            &client.buffer(&[], &[1.]).unwrap(),
        ])
        .unwrap();
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [5.]);
    assert_eq!(output[1].to_vec::<i32>().unwrap(), [10]);
}

fn build(compiler: &mut Compiler, reordered: bool) -> (StateProgram, [StateSlot; 2], usize) {
    let mut graph = StateGraph::default();
    let (value, count) = if reordered {
        let count = graph.state_i32(&[]).unwrap();
        (graph.state(&[]).unwrap(), count)
    } else {
        (graph.state(&[]).unwrap(), graph.state_i32(&[]).unwrap())
    };
    if reordered {
        graph.input(&[5]).unwrap();
    }
    let scale = graph.input(&[]).unwrap();
    let input = graph.input(&[]).unwrap();
    let next = graph
        .read(&value)
        .unwrap()
        .add(&input.mul(&scale).unwrap())
        .unwrap();
    let next_count = graph.read(&count).unwrap().wrapping_add_scalar(1).unwrap();
    graph
        .write_many(&[(&value, &next), (&count, &next_count)])
        .unwrap();
    (
        graph.compile_pruned(compiler, &[next, next_count]).unwrap(),
        [value, count],
        usize::from(reordered),
    )
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_program_switch_remaps_state_and_validates_before_replacing_bindings() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let (first, a, first_scale) = build(&mut compiler, false);
    let (second, b, second_scale) = build(&mut compiler, true);
    assert_eq!(first.input_indices(), [0, 1]);
    assert_eq!(second.input_indices(), [1, 2]);
    let mut session = first
        .session(vec![
            (a[0].clone(), client.buffer(&[], &[1.]).unwrap()),
            (a[1].clone(), client.buffer(&[], &[16_777_217]).unwrap()),
        ])
        .unwrap();
    let two = Rc::new(client.buffer(&[], &[2.]).unwrap());
    session
        .bind_inputs(vec![(first_scale, two.clone())])
        .unwrap();
    let source_names = [("count", a[1].clone()), ("value", a[0].clone())];
    let target_names = [("value", b[0].clone()), ("count", b[1].clone())];
    let mapping = first
        .map_state_by_name(&source_names, &second, &target_names)
        .unwrap();
    for bad in [
        vec![("", a[1].clone()), ("value", a[0].clone())],
        vec![("count", a[1].clone()), ("count", a[0].clone())],
        vec![("unknown", a[1].clone()), ("value", a[0].clone())],
        vec![("count", a[0].clone()), ("value", a[1].clone())],
        vec![("count", a[1].clone())],
        vec![("count", a[1].clone()), ("value", a[1].clone())],
        vec![("count", b[1].clone()), ("value", a[0].clone())],
    ] {
        assert!(
            first
                .map_state_by_name(&bad, &second, &target_names)
                .is_err()
        );
    }
    for bad in [
        vec![("value", b[0].clone()), ("value", b[1].clone())],
        vec![("value", b[0].clone()), ("", b[1].clone())],
        vec![("value", b[0].clone()), ("Count", b[1].clone())],
    ] {
        assert!(
            first
                .map_state_by_name(&source_names, &second, &bad)
                .is_err()
        );
    }
    // Every failure must preserve state AND the source plan's fixed scale.
    let mut expected = 1.;
    let mut counter = 16_777_217;
    let bad_mappings = [
        vec![],
        vec![mapping[0].clone(), mapping[0].clone()],
        vec![(a[0].clone(), b[0].clone()), (a[1].clone(), b[0].clone())],
        vec![(b[0].clone(), b[0].clone()), (a[1].clone(), b[1].clone())],
        vec![(a[0].clone(), b[1].clone()), (a[1].clone(), b[0].clone())],
    ];
    for bad in bad_mappings {
        assert!(session.switch_program(&second, &bad, vec![]).is_err());
        assert_eq!(session.input_count(), 1);
        assert_eq!(
            session.state(&a[0]).unwrap().to_vec::<f32>().unwrap(),
            [expected]
        );
        let output = session.run(&[&client.buffer(&[], &[3.]).unwrap()]).unwrap();
        expected += 6.;
        counter += 1;
        assert_eq!(output[0].to_vec::<f32>().unwrap(), [expected]);
        assert_eq!(output[1].to_vec::<i32>().unwrap(), [counter]);
    }
    for bindings in [
        vec![(0, two.clone())], // pruned destination input
        vec![(second_scale, two.clone()), (second_scale, two.clone())],
        vec![(second_scale, Rc::new(client.buffer(&[], &[2]).unwrap()))],
    ] {
        assert!(session.switch_program(&second, &mapping, bindings).is_err());
        assert_eq!(session.input_count(), 1);
        let output = session.run(&[&client.buffer(&[], &[1.]).unwrap()]).unwrap();
        expected += 2.;
        counter += 1;
        assert_eq!(output[0].to_vec::<f32>().unwrap(), [expected]);
        assert_eq!(output[1].to_vec::<i32>().unwrap(), [counter]);
    }
    session
        .switch_program(
            &second,
            &mapping,
            vec![(second_scale, Rc::new(client.buffer(&[], &[3.]).unwrap()))],
        )
        .unwrap();
    assert!(session.state(&a[0]).is_err());
    assert_eq!(
        session.state(&b[0]).unwrap().to_vec::<f32>().unwrap(),
        [expected]
    );
    assert_eq!(session.input_count(), 1);
    let output = session.run(&[&client.buffer(&[], &[2.]).unwrap()]).unwrap();
    expected += 6.;
    counter += 1;
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [expected]);
    assert_eq!(output[1].to_vec::<i32>().unwrap(), [counter]);
    // Explicit empty fixed set makes both source inputs dynamic on switch back.
    let reverse = second
        .map_state_by_name(&target_names, &first, &source_names)
        .unwrap();
    session.switch_program(&first, &reverse, vec![]).unwrap();
    assert_eq!(session.input_count(), 2);
    assert!(session.run(&[&two]).is_err());
    let output = session
        .run(&[
            &client.buffer(&[], &[4.]).unwrap(),
            &client.buffer(&[], &[1.]).unwrap(),
        ])
        .unwrap();
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [expected + 4.]);
    assert_eq!(output[1].to_vec::<i32>().unwrap(), [counter + 1]);
    assert_eq!(compiler.stats().misses, 2);
}
