use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer};

#[test]
fn invalid_slot_rejects_before_running_closure() {
    let mut graph = StateGraph::default();
    let float = graph.state(&[]).unwrap();
    let integer = graph.state_i32(&[]).unwrap();
    let foreign = StateGraph::default().state(&[]).unwrap();
    assert!(graph.update(&foreign, |_| panic!("must not run")).is_err());
    assert_eq!(
        graph
            .update(&integer, |old| old.wrapping_add_scalar(1))
            .unwrap()
            .dtype(),
        rxla_core::DType::I32
    );
    assert_eq!(
        graph
            .update(&float, |old| old.add_scalar(1.))
            .unwrap()
            .dtype(),
        rxla_core::DType::F32
    );
}

#[test]
fn conditional_updates_validate_before_invoking_closure() {
    let mut graph = StateGraph::default();
    let float = graph.state(&[]).unwrap();
    let integer = graph.state_i32(&[]).unwrap();
    let foreign = StateGraph::default().state(&[]).unwrap();
    let mask = graph.constant(&[], &[1.]).unwrap();
    assert!(
        graph
            .update_if(&mask, &foreign, |_| panic!("invalid slot"))
            .is_err()
    );
    assert_eq!(
        graph
            .update_if(&mask, &integer, |old| old.wrapping_add_scalar(1))
            .unwrap()
            .dtype(),
        rxla_core::DType::I32
    );
    assert_eq!(
        graph
            .update_if(&mask, &float, |old| old.add_scalar(1.))
            .unwrap()
            .dtype(),
        rxla_core::DType::F32
    );
    assert!(
        graph
            .update_if(&mask, &foreign, |_| panic!("invalid slot"))
            .is_err()
    );
    for invalid in [
        graph.constant(&[1], &[1.]).unwrap(),
        Tracer::default().constant(&[], &[1.]).unwrap(),
    ] {
        assert!(
            graph
                .update_if(&invalid, &float, |_| panic!("invalid mask"))
                .is_err()
        );
        assert!(
            graph
                .update_if(&invalid, &integer, |_| panic!("invalid mask"))
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_conditional_updates_return_selected_versions_and_preserve_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let value = graph.state(&[]).unwrap();
    let count = graph.state_i32(&[]).unwrap();
    let mask = graph.input(&[]).unwrap();
    let delta = graph.input(&[]).unwrap();
    let mut calls = 0;
    let selected = graph
        .update_if(&mask, &value, |old| {
            calls += 1;
            old.add(&delta)
        })
        .unwrap();
    graph
        .update_if(&mask, &count, |old| old.wrapping_add_scalar(1))
        .unwrap();
    // Failed proposals must preserve the selected version, not the original input.
    let wrong = graph.constant(&[1], &[99.]).unwrap();
    assert!(graph.update_if(&mask, &value, |_| Ok(wrong)).is_err());
    assert!(
        graph
            .update_if(&mask, &count, |_| Tracer::default().scalar_i32(99))
            .is_err()
    );
    let grad = selected
        .grad(std::slice::from_ref(&delta))
        .unwrap()
        .remove(0);
    let after = graph.update(&value, |old| old.add(&delta)).unwrap();
    let plan = graph
        .compile(&mut compiler, &[selected, after, grad])
        .unwrap();
    let mut session = plan
        .session(vec![
            (value.clone(), client.buffer(&[], &[10.]).unwrap()),
            (count.clone(), client.buffer(&[], &[0]).unwrap()),
        ])
        .unwrap();
    for (mask, expected, count_value) in [
        (0., [10., 12., 0.], 0),
        (1., [14., 16., 1.], 1),
        (-0., [16., 18., 0.], 1),
        (-2., [20., 22., 1.], 2),
        (f32::NAN, [24., 26., 1.], 3),
    ] {
        let mask = client.buffer(&[], &[mask]).unwrap();
        let delta = client.buffer(&[], &[2.]).unwrap();
        let outputs = session.run(&[&mask, &delta]).unwrap();
        for (output, expected) in outputs.iter().zip(expected) {
            assert_eq!(output.to_vec::<f32>().unwrap(), [expected]);
        }
        assert_eq!(
            session.state(&value).unwrap().to_vec::<f32>().unwrap(),
            [expected[1]]
        );
        assert_eq!(
            session.state(&count).unwrap().to_vec::<i32>().unwrap(),
            [count_value]
        );
    }
    assert_eq!(calls, 1);
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_updates_are_recorded_once_and_errors_preserve_current_version() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let value = graph.state(&[]).unwrap();
    let counter = graph.state_i32(&[]).unwrap();
    let delta = graph.input(&[]).unwrap();
    let mut calls = 0;
    let first = graph
        .update(&value, |old| {
            calls += 1;
            old.add(&delta)
        })
        .unwrap();
    let second = graph.update(&value, |old| old.add(&delta)).unwrap();
    graph
        .update(&counter, |old| old.wrapping_add_scalar(1))
        .unwrap();
    // Each error must preserve the already-recorded updates, not reset to inputs.
    assert!(graph.update(&value, |old| old.reshape(&[2])).is_err());
    assert!(
        graph
            .update(&value, |_| Tracer::default().input(&[]))
            .is_err()
    );
    let wrong = graph.constant(&[1], &[99.]).unwrap();
    assert!(graph.update(&value, |_| Ok(wrong)).is_err());
    assert!(
        graph
            .update(&counter, |_| Tracer::default().input_i32(&[]))
            .is_err()
    );
    let plan = graph.compile(&mut compiler, &[first, second]).unwrap();
    let mut session = plan
        .session(vec![
            (value.clone(), client.buffer(&[], &[10.]).unwrap()),
            (counter.clone(), client.buffer(&[], &[0]).unwrap()),
        ])
        .unwrap();
    for (step, (delta, expected)) in [(2., [12., 14.]), (-1., [13., 12.])]
        .into_iter()
        .enumerate()
    {
        let input = client.buffer(&[], &[delta]).unwrap();
        let outputs = session.run(&[&input]).unwrap();
        for (output, expected) in outputs.iter().zip(expected) {
            assert_eq!(output.to_vec::<f32>().unwrap(), [expected]);
        }
        assert_eq!(
            session.state(&value).unwrap().to_vec::<f32>().unwrap(),
            [expected[1]]
        );
        assert_eq!(
            session.state(&counter).unwrap().to_vec::<i32>().unwrap(),
            [step as i32 + 1]
        );
    }
    assert_eq!(calls, 1);
    assert_eq!(compiler.stats().misses, 1);
}
