use rxla_core::random::{ThreefryState, threefry2x32_blocks};
use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer};

#[test]
fn sampled_dropout_validates_probability_and_graph() {
    let mut graph = StateGraph::default();
    let random = ThreefryState::new(&mut graph).unwrap();
    let x = graph.input(&[6]).unwrap();
    let mut sequence = random.begin(&graph).unwrap();
    for p in [0., -0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(sequence.dropout(&x, p).is_err());
    }
    let foreign = Tracer::default().input(&[6]).unwrap();
    for p in [0.5, 1.] {
        assert!(sequence.dropout(&foreign, p).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_sampled_dropout_reuses_mask_for_derivatives_and_explicit_recomputation() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let random = ThreefryState::new(&mut graph).unwrap();
    let x = graph.input(&[6]).unwrap();
    let condition = graph.input(&[]).unwrap();
    let words = random
        .slots()
        .iter()
        .map(|s| graph.read(s).unwrap())
        .collect::<Vec<_>>();
    let reference =
        threefry2x32_blocks([&words[0], &words[1]], [&words[2], &words[3]], &[6]).unwrap();
    let mut sequence = random.begin(&graph).unwrap();
    // These attempts must not move the first successful draw's cursor.
    assert!(sequence.dropout(&x, 0.).is_err());
    assert!(
        sequence
            .dropout(&Tracer::default().input(&[6]).unwrap(), 0.5)
            .is_err()
    );
    let sample = sequence.dropout(&x, 0.5).unwrap();
    let recomputed = x.dropout_with_mask(&sample.keep_mask, 0.5).unwrap();
    let loss = sample.output.square().unwrap().sum(&[0], false).unwrap();
    let gradient = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let second = gradient
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let identity = sequence.dropout(&x, 1.).unwrap();
    let empty = graph.input(&[0]).unwrap();
    let empty_sample = sequence.dropout(&empty, 0.5).unwrap();
    let accept = sequence.commit_if(&mut graph, &condition).unwrap();
    let program = graph
        .compile_outputs(
            &mut compiler,
            &[
                sample.output,
                sample.keep_mask,
                recomputed,
                gradient,
                second,
                identity.output,
                identity.keep_mask,
                empty_sample.output,
                accept,
                reference.bits[0].clone(),
            ],
        )
        .unwrap();
    let values = [-3., -1., 0., 1., 2., 7.];
    let input = client.buffer(&[6], &values).unwrap();
    let empty = client.buffer::<f32>(&[0], &[]).unwrap();
    let start = u32::MAX as u64 - 3;
    let mut session = program
        .session(random.initial_state(&client, [42, 999], start).unwrap())
        .unwrap();
    let mut counter = start;
    let mut rejected_mask = None;
    for request in [0., 1., 1., 0., 1.] {
        let output = session
            .run(&[&input, &client.buffer(&[], &[request]).unwrap(), &empty])
            .unwrap();
        let mask: Vec<_> = output[9]
            .to_vec::<i32>()
            .unwrap()
            .iter()
            .map(|&word| if (word as u32) < 0x80000000 { 1. } else { 0. })
            .collect();
        assert_eq!(output[1].to_vec::<f32>().unwrap(), mask);
        if let Some(previous) = rejected_mask.take() {
            assert_eq!(mask, previous);
        }
        for (index, expected) in [
            (
                0,
                values
                    .iter()
                    .zip(&mask)
                    .map(|(x, m)| x * m * 2.)
                    .collect::<Vec<_>>(),
            ),
            (
                2,
                values.iter().zip(&mask).map(|(x, m)| x * m * 2.).collect(),
            ),
            (
                3,
                values.iter().zip(&mask).map(|(x, m)| x * m * 8.).collect(),
            ),
            (4, mask.iter().map(|m| m * 8.).collect()),
            (5, values.to_vec()),
            (6, vec![1.; 6]),
            (7, vec![]),
            (8, vec![request]),
        ] {
            assert_eq!(
                output[index].to_vec::<f32>().unwrap(),
                expected,
                "output {index}"
            );
        }
        if request != 0. {
            counter += 6;
        } else {
            rejected_mask = Some(mask);
        }
        for (slot, word) in
            random
                .slots()
                .iter()
                .zip([42, 999, counter as i32, (counter >> 32) as i32])
        {
            assert_eq!(
                session.state(slot).unwrap().to_vec::<i32>().unwrap(),
                [word]
            );
        }
    }
    assert_eq!(compiler.stats().misses, 1);
}
