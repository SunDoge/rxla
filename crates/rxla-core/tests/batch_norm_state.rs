use rxla_core::{BatchNormState, BatchNormTraining, CacheLimits, Client, Compiler, StateGraph};

#[test]
fn statistic_proposals_reject_stale_versions_and_preserve_state_on_errors() {
    let mut graph = StateGraph::default();
    let statistics = BatchNormState::new(&mut graph, 2).unwrap();
    let values = graph.input(&[2]).unwrap();
    let batch = BatchNormTraining {
        output: values.clone(),
        mean: values.clone(),
        variance: values,
    };
    let first = statistics.prepare(&graph, &batch, 0.5).unwrap();
    let stale = statistics.prepare(&graph, &batch, 0.5).unwrap();
    first.commit(&mut graph).unwrap();
    assert!(stale.commit(&mut graph).is_err());
    let survivor = statistics.prepare(&graph, &batch, 0.5).unwrap();
    let invalid = graph.constant(&[1], &[1.]).unwrap();
    assert!(
        statistics
            .prepare(&graph, &batch, 0.5)
            .unwrap()
            .commit_if(&mut graph, &invalid)
            .is_err()
    );
    let mut foreign = StateGraph::default();
    assert!(
        statistics
            .prepare(&graph, &batch, 0.5)
            .unwrap()
            .commit(&mut foreign)
            .is_err()
    );
    survivor.commit(&mut graph).unwrap();
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_proposed_statistic_finiteness_can_gate_other_state() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    for rate in [0., 0.5, 1.] {
        let mut graph = StateGraph::default();
        let statistics = BatchNormState::new(&mut graph, 2).unwrap();
        let count = graph.state_i32(&[]).unwrap();
        let mean = graph.input(&[2]).unwrap();
        let variance = graph.input(&[2]).unwrap();
        let batch = BatchNormTraining {
            output: mean.clone(),
            mean,
            variance,
        };
        let proposal = statistics.prepare(&graph, &batch, rate).unwrap();
        let finite = proposal.finite_mask().unwrap();
        let next_count = graph.read(&count).unwrap().wrapping_add_scalar(1).unwrap();
        graph
            .write_many_if(&finite, &[(&count, next_count)])
            .unwrap();
        proposal.commit_if(&mut graph, &finite).unwrap();
        let program = graph.compile(&mut compiler, &[finite]).unwrap();
        let mut initial = statistics.initial_state(&client).unwrap();
        initial.push((count.clone(), client.buffer(&[], &[0]).unwrap()));
        let mut session = program.session(initial).unwrap();
        let (mut expected_mean, mut expected_variance, mut accepted_count) =
            ([0_f32; 2], [1_f32; 2], 0);
        for (mean, variance) in [
            ([2., 4.], [3., 5.]),
            ([f32::INFINITY, 1.], [3., 5.]),
            ([1., 2.], [f32::NAN, 3.]),
            ([1., 2.], [3., f32::INFINITY]),
            ([-2., -4.], [1., 2.]),
        ] {
            let output = session
                .run(&[
                    &client.buffer(&[2], &mean).unwrap(),
                    &client.buffer(&[2], &variance).unwrap(),
                ])
                .unwrap();
            let accepted = rate == 0. || mean.iter().chain(&variance).all(|x| x.is_finite());
            assert_eq!(
                output[0].to_vec::<f32>().unwrap(),
                [if accepted { 1. } else { 0. }]
            );
            if accepted {
                accepted_count += 1;
                if rate != 0. {
                    for axis in 0..2 {
                        expected_mean[axis] = (1. - rate) * expected_mean[axis] + rate * mean[axis];
                        expected_variance[axis] =
                            (1. - rate) * expected_variance[axis] + rate * variance[axis];
                    }
                }
            }
            assert_eq!(
                session
                    .state(statistics.mean_slot())
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                expected_mean
            );
            assert_eq!(
                session
                    .state(statistics.variance_slot())
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                expected_variance
            );
            assert_eq!(
                session.state(&count).unwrap().to_vec::<i32>().unwrap(),
                [accepted_count]
            );
        }
    }
    assert_eq!(compiler.stats().misses, 3);
}

#[test]
fn running_statistics_validate_configuration_shape_and_graph() {
    let mut g = StateGraph::default();
    assert!(BatchNormState::new(&mut g, 0).is_err());
    let state = BatchNormState::new(&mut g, 2).unwrap();
    let x = g.input(&[2, 2]).unwrap();
    let affine = g.constant(&[2], &[1., 1.]).unwrap();
    let batch = x.batch_norm_training(1, &affine, &affine, 0.1).unwrap();
    for rate in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(state.update(&mut g, &batch, rate).is_err());
    }
    let wrong = BatchNormTraining {
        output: batch.output.clone(),
        mean: batch.mean.clone(),
        variance: g.input(&[3]).unwrap(),
    };
    assert!(state.update(&mut g, &wrong, 0.5).is_err());
    let mut foreign = StateGraph::default();
    assert!(state.update(&mut foreign, &batch, 0.5).is_err());
    let foreign_condition = foreign.constant(&[], &[1.]).unwrap();
    assert!(
        state
            .update_if(&mut g, &batch, 0., &foreign_condition)
            .is_err()
    );
    let vector_condition = g.constant(&[1], &[1.]).unwrap();
    assert!(
        state
            .update_if(&mut g, &batch, 0.5, &vector_condition)
            .is_err()
    );
    state.update(&mut g, &batch, 0.5).unwrap();
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_running_statistics_ema_rejection_detach_and_session_isolation() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for rate in [0., 0.25, 1.] {
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let mut g = StateGraph::default();
        let state = BatchNormState::new(&mut g, 2).unwrap();
        let x = g.input(&[2, 2]).unwrap();
        let condition = g.input(&[]).unwrap();
        let weight = g.constant(&[2], &[1., 1.]).unwrap();
        let bias = g.constant(&[2], &[0., 0.]).unwrap();
        let batch = x.batch_norm_training(1, &weight, &bias, 0.125).unwrap();
        state.update_if(&mut g, &batch, rate, &condition).unwrap();
        let (mean, variance) = state.read(&g).unwrap();
        let gradient = mean
            .add(&variance)
            .unwrap()
            .sum(&[0], false)
            .unwrap()
            .grad(&[x])
            .unwrap()
            .remove(0);
        let program = g
            .compile(&mut compiler, &[batch.output, mean, variance, gradient])
            .unwrap();
        let mut a = program
            .session(state.initial_state(&client).unwrap())
            .unwrap();
        let b = program
            .session(state.initial_state(&client).unwrap())
            .unwrap();
        let mut expected_mean = [0.; 2];
        let mut expected_variance = [1.; 2];
        for (values, accept) in [
            ([1., 3., 5., 7.], 1.),
            ([f32::NAN; 4], 0.),
            ([-2., 2., 2., 6.], -0.),
            ([2., 0., 6., 8.], 1.),
        ] {
            let input = client.buffer(&[2, 2], &values).unwrap();
            let mask = client.buffer(&[], &[accept]).unwrap();
            let outputs = a.run(&[&input, &mask]).unwrap();
            if accept != 0. && rate != 0. {
                for c in 0..2 {
                    let mean = (values[c] + values[c + 2]) * 0.5;
                    let variance =
                        ((values[c] - mean).powi(2) + (values[c + 2] - mean).powi(2)) * 0.5;
                    expected_mean[c] = (1. - rate) * expected_mean[c] + rate * mean;
                    expected_variance[c] = (1. - rate) * expected_variance[c] + rate * variance;
                }
            }
            assert_eq!(
                a.state(state.mean_slot()).unwrap().to_vec::<f32>().unwrap(),
                expected_mean
            );
            assert_eq!(
                a.state(state.variance_slot())
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                expected_variance
            );
            assert_eq!(outputs[1].to_vec::<f32>().unwrap(), expected_mean);
            assert_eq!(outputs[2].to_vec::<f32>().unwrap(), expected_variance);
            assert_eq!(outputs[3].to_vec::<f32>().unwrap(), [0.; 4]);
            assert_eq!(
                b.state(state.mean_slot()).unwrap().to_vec::<f32>().unwrap(),
                [0.; 2]
            );
            assert_eq!(
                b.state(state.variance_slot())
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                [1.; 2]
            );
        }
        if rate == 0. {
            let bad = client.buffer(&[2, 2], &[f32::NAN; 4]).unwrap();
            let accept = client.buffer(&[], &[1.]).unwrap();
            a.run(&[&bad, &accept]).unwrap();
            assert_eq!(
                a.state(state.mean_slot()).unwrap().to_vec::<f32>().unwrap(),
                [0.; 2]
            );
            assert_eq!(
                a.state(state.variance_slot())
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                [1.; 2]
            );
        }
        if rate == 1. {
            let mut recovered = program
                .session(vec![
                    (
                        state.mean_slot().clone(),
                        client.buffer(&[2], &[f32::NAN; 2]).unwrap(),
                    ),
                    (
                        state.variance_slot().clone(),
                        client.buffer(&[2], &[f32::NAN; 2]).unwrap(),
                    ),
                ])
                .unwrap();
            let input = client.buffer(&[2, 2], &[1., 3., 5., 7.]).unwrap();
            let accept = client.buffer(&[], &[1.]).unwrap();
            recovered.run(&[&input, &accept]).unwrap();
            assert_eq!(
                recovered
                    .state(state.mean_slot())
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                [3., 5.]
            );
            assert_eq!(
                recovered
                    .state(state.variance_slot())
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                [4., 4.]
            );
        }
        assert_eq!(compiler.stats().misses, 1);
    }
}
