use rxla_core::{CacheLimits, Client, Compiler, F32, I32, State, StateGraph};

#[test]
fn prepared_parameter_metadata_handles_pruning_storage_dtype_and_identity() {
    use rxla_core::DType;
    let mut graph = StateGraph::default();
    let unused = graph.parameter(&[4]).unwrap();
    let trainable = graph.trainable_parameter(&[2]).unwrap();
    let weight = graph.parameter_bf16_as_f32(&[2]).unwrap();
    let empty = graph.trainable_parameter(&[2, 0]).unwrap();
    let output = trainable.tensor().add(weight.tensor()).unwrap();
    let full = graph.prepare(std::slice::from_ref(&output)).unwrap();
    let pruned = graph.prepare_pruned(&[output]).unwrap();
    let late = graph.parameter(&[2]).unwrap();
    let late_state = graph.trainable_parameter(&[2]).unwrap();
    let foreign = StateGraph::default().parameter_bf16_as_f32(&[2]).unwrap();
    assert_eq!(full.parameter_type(&unused).unwrap(), (DType::F32, vec![4]));
    assert!(pruned.parameter_type(&unused).is_err());
    for prepared in [full, pruned] {
        assert_eq!(
            prepared.parameter_type(&trainable).unwrap(),
            (DType::F32, vec![2])
        );
        assert_eq!(
            prepared.parameter_type(&weight).unwrap(),
            (DType::BF16, vec![2])
        );
        assert_eq!(
            prepared.parameter_type(&empty).unwrap(),
            (DType::F32, vec![2, 0])
        );
        for invalid in [&late, &late_state, &foreign] {
            assert!(prepared.parameter_type(invalid).is_err());
        }
    }
}

#[test]
fn prepared_state_metadata_validates_schema_without_plugin() {
    let mut graph = StateGraph::default();
    graph.input(&[3]).unwrap();
    let empty = graph.state(&[2, 0]).unwrap();
    let count = graph.state_i32(&[]).unwrap();
    let replacement = graph.input_i32_scalar().unwrap();
    graph.write(&count, &replacement).unwrap();
    let full = graph.prepare(&[]).unwrap();
    let pruned = graph.prepare_pruned(&[]).unwrap();
    assert_eq!(full.input_indices(), [0, 1]);
    assert!(full.output_spec(0).is_none()); // hidden state is not a visible output
    assert!(pruned.output_spec(usize::MAX).is_none());
    let visible = graph.prepare(&[graph.read(&count).unwrap()]).unwrap();
    assert_eq!(
        visible.output_spec(0),
        Some(rxla_core::OutputSpec {
            shape: &[],
            dtype: rxla_core::DType::I32
        })
    );
    assert!(visible.output_spec(1).is_none());
    assert_eq!(pruned.input_indices(), [1]);
    let late = graph.state(&[]).unwrap();
    let foreign = StateGraph::default().state_i32(&[]).unwrap();
    drop(graph);
    for prepared in [full, pruned] {
        assert_eq!(
            prepared.state_type(&empty).unwrap(),
            (rxla_core::DType::F32, vec![2, 0])
        );
        let slots = [count.clone(), empty.clone()];
        assert_eq!(
            prepared.state_layout(&slots).unwrap(),
            [
                (rxla_core::DType::I32, vec![]),
                (rxla_core::DType::F32, vec![2, 0]),
            ]
        );
        for invalid in [
            vec![],
            vec![count.clone()],
            vec![count.clone(), count.clone()],
            vec![foreign.clone(), empty.clone()],
            vec![late.clone(), empty.clone()],
        ] {
            assert!(prepared.state_layout(&invalid).is_err());
        }
        assert!(prepared.state_type(&foreign).is_err());
        assert!(prepared.state_type(&late).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn prepared_state_keeps_bf16_binding_identity_and_session_isolation() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut graph = StateGraph::default();
    let unused = graph.parameter(&[3]).unwrap();
    let total = graph.state(&[]).unwrap();
    let weight = graph.parameter_bf16_as_f32(&[]).unwrap();
    let input = graph.input(&[]).unwrap();
    let next = graph
        .read(&total)
        .unwrap()
        .add(&input.mul(weight.tensor()).unwrap())
        .unwrap();
    graph.write(&total, &next).unwrap();
    let prepared = graph.prepare_pruned(&[next]).unwrap();
    drop(graph);
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let program = prepared.compile(&mut compiler).unwrap();
    assert_eq!(program.input_indices(), [1, 2]);
    assert_eq!(
        prepared.parameter_type(&weight).unwrap(),
        program.parameter_type(&weight).unwrap()
    );
    let mut first = program
        .session(vec![(total.clone(), client.buffer(&[], &[0.]).unwrap())])
        .unwrap();
    let mut second = program
        .session(vec![(total.clone(), client.buffer(&[], &[10.]).unwrap())])
        .unwrap();
    let first_weight = std::sync::Arc::new(
        client
            .buffer(&[], &[rxla_core::bf16::from_bits(0x4000)])
            .unwrap(),
    ); // 2
    let second_weight = std::sync::Arc::new(
        client
            .buffer(&[], &[rxla_core::bf16::from_bits(0x4040)])
            .unwrap(),
    ); // 3
    first
        .bind_parameters(vec![(weight.clone(), first_weight.clone())])
        .unwrap();
    second
        .bind_parameters(vec![(weight.clone(), second_weight.clone())])
        .unwrap();
    assert_eq!(first.input_count(), 1);
    assert_eq!(second.input_count(), 1);
    let foreign = StateGraph::default().parameter_bf16_as_f32(&[]).unwrap();
    for invalid in [
        vec![(
            weight.clone(),
            std::sync::Arc::new(client.buffer(&[], &[2.]).unwrap()),
        )],
        vec![(
            unused,
            std::sync::Arc::new(client.buffer(&[3], &[0.; 3]).unwrap()),
        )],
        vec![(foreign, second_weight.clone())],
    ] {
        assert!(first.bind_parameters(invalid).is_err());
        assert!(std::ptr::eq(
            first.parameter(&weight).unwrap(),
            first_weight.as_ref()
        ));
        assert_eq!(first.state(&total).unwrap().to_vec::<f32>().unwrap(), [0.]);
    }
    let input = client.buffer(&[], &[2.]).unwrap();
    assert_eq!(
        first.run(&[&input]).unwrap()[0].to_vec::<f32>().unwrap(),
        [4.]
    );
    assert_eq!(
        second.run(&[&input]).unwrap()[0].to_vec::<f32>().unwrap(),
        [16.]
    );
    first
        .bind_parameters(vec![(
            weight.clone(),
            std::sync::Arc::new(
                client
                    .buffer(&[], &[rxla_core::bf16::from_bits(0x40a0)])
                    .unwrap(),
            ),
        )])
        .unwrap(); // 5
    assert_eq!(
        first.run(&[&input]).unwrap()[0].to_vec::<f32>().unwrap(),
        [14.]
    );
    assert_eq!(
        second.state(&total).unwrap().to_vec::<f32>().unwrap(),
        [16.]
    );
    assert!(std::ptr::eq(
        second.parameter(&weight).unwrap(),
        second_weight.as_ref()
    ));
    prepared.compile(&mut compiler).unwrap();
    assert_eq!((compiler.stats().misses, compiler.stats().hits), (1, 1));
}

#[test]
fn state_preparation_is_host_only_and_requires_at_least_one_root() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<rxla_core::PreparedStateGraph>();
    let mut graph = StateGraph::default();
    assert!(graph.prepare(&[]).is_err());
    assert!(graph.prepare_pruned(&[]).is_err());
    State::<I32>::new(&mut graph, &[]).unwrap();
    graph.prepare(&[]).unwrap();
    graph.prepare_pruned(&[]).unwrap();
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn prepared_state_freezes_hidden_updates_schema_and_input_mapping() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for pruned in [false, true] {
        let mut graph = StateGraph::default();
        graph.input(&[2]).unwrap(); // unused visible input 0
        let value = State::<F32>::new(&mut graph, &[]).unwrap();
        let step = graph.input(&[]).unwrap(); // visible input 1, hidden-output-only use
        let count = State::<I32>::new(&mut graph, &[]).unwrap();
        let next_value = value.read(&graph).unwrap().add(&step).unwrap();
        let next_count = count.read(&graph).unwrap().wrapping_add_scalar(1).unwrap();
        graph
            .transaction()
            .with(&value, &next_value)
            .unwrap()
            .with(&count, &next_count)
            .unwrap()
            .commit()
            .unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let (prepared, ordinary) = if pruned {
            (
                graph.prepare_pruned(&[]).unwrap(),
                graph.compile_pruned(&mut compiler, &[]).unwrap(),
            )
        } else {
            (
                graph.prepare(&[]).unwrap(),
                graph.compile(&mut compiler, &[]).unwrap(),
            )
        };
        let compile_time = compiler.stats().compile_time;
        let replacement = graph.constant(&[], &[99.]).unwrap();
        graph.write(value.as_slot(), &replacement).unwrap();
        let late = State::<F32>::new(&mut graph, &[]).unwrap();
        drop(graph);
        let program = prepared.compile(&mut compiler).unwrap();
        assert_eq!(program.input_indices(), ordinary.input_indices());
        assert_eq!(
            program.input_indices(),
            if pruned { vec![1] } else { vec![0, 1] }
        );
        assert!(program.state_type(late.as_slot()).is_err());
        assert_eq!(compiler.stats().misses, 1);
        assert_eq!(compiler.stats().hits, 1);
        assert_eq!(compiler.stats().compile_time, compile_time);
        let initial = || {
            vec![
                (value.as_slot().clone(), client.buffer(&[], &[10.]).unwrap()),
                (count.as_slot().clone(), client.buffer(&[], &[3]).unwrap()),
            ]
        };
        let mut session = program.session(initial()).unwrap();
        let independent = program.session(initial()).unwrap();
        assert!(session.run(&[]).is_err());
        let unused = client.buffer(&[2], &[0., 0.]).unwrap();
        for step in [2., -1., 4.] {
            let step = client.buffer(&[], &[step]).unwrap();
            let inputs = if pruned {
                vec![&step]
            } else {
                vec![&unused, &step]
            };
            assert!(session.run(&inputs).unwrap().is_empty());
        }
        assert_eq!(
            session
                .state(value.as_slot())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            [15.]
        );
        assert_eq!(
            session
                .state(count.as_slot())
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [6]
        );
        assert_eq!(
            independent
                .state(value.as_slot())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            [10.]
        );
        assert_eq!(
            independent
                .state(count.as_slot())
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [3]
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn copied_state_is_complete_independent_and_failure_preserves_source() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut graph = StateGraph::default();
    // The first tensor fits four bytes; preflight rejects the second before copying.
    let count = State::<I32>::new(&mut graph, &[]).unwrap();
    let value = State::<F32>::new(&mut graph, &[2]).unwrap();
    let next_count = count.read(&graph).unwrap().wrapping_add_scalar(1).unwrap();
    let next_value = value.read(&graph).unwrap().add_scalar(2.).unwrap();
    graph
        .transaction()
        .with(&count, &next_count)
        .unwrap()
        .with(&value, &next_value)
        .unwrap()
        .commit()
        .unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let program = graph.compile(&mut compiler, &[]).unwrap();
    let mut source = program
        .session(vec![
            (count.as_slot().clone(), client.buffer(&[], &[7]).unwrap()),
            (
                value.as_slot().clone(),
                client.buffer(&[2], &[1., 3.]).unwrap(),
            ),
        ])
        .unwrap();
    assert!(source.copy_state_to_client_via_host(&client, 4).is_err());
    assert_eq!(
        source
            .state(count.as_slot())
            .unwrap()
            .host_payload_bytes()
            .unwrap(),
        4
    );
    assert_eq!(
        source
            .state(value.as_slot())
            .unwrap()
            .host_payload_bytes()
            .unwrap(),
        8
    );
    assert!(
        source
            .copy_state_to_client_via_host_with_limits(&client, 8, 11)
            .is_err()
    );
    assert_eq!(
        source
            .state(count.as_slot())
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [7]
    );
    assert_eq!(
        source
            .state(value.as_slot())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [1., 3.]
    );
    source.run(&[]).unwrap();
    let copied = source
        .copy_state_to_client_via_host_with_limits(&client, 8, 12)
        .unwrap();
    assert_eq!(copied.len(), 2);
    let mut fork = program.session(copied).unwrap();
    fork.run(&[]).unwrap();
    assert_eq!(
        source
            .state(count.as_slot())
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [8]
    );
    assert_eq!(
        source
            .state(value.as_slot())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [3., 5.]
    );
    drop(source);
    fork.run(&[]).unwrap();
    assert_eq!(
        fork.state(count.as_slot())
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [10]
    );
    assert_eq!(
        fork.state(value.as_slot())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [7., 9.]
    );
    assert_eq!(compiler.stats().misses, 1);
}
