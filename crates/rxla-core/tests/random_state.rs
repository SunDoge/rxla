use rxla_core::random::{ThreefryState, threefry2x32_blocks};
use rxla_core::{CacheLimits, Client, Compiler, StateGraph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_distributions_match_raw_bits_and_preserve_cursor_on_invalid_or_endpoint_calls() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    for probability in [1e-10, 0.3, 0.5, 1. - f32::EPSILON] {
        let mut graph = StateGraph::default();
        let random = ThreefryState::new(&mut graph).unwrap();
        let words = random
            .slots()
            .iter()
            .map(|s| graph.read(s).unwrap())
            .collect::<Vec<_>>();
        let reference =
            threefry2x32_blocks([&words[0], &words[1]], [&words[2], &words[3]], &[8]).unwrap();
        let mut sequence = random.begin(&graph).unwrap();
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            assert!(sequence.bernoulli(&[3], invalid).is_err());
        }
        assert!(sequence.uniform_f32(&[-1]).is_err());
        assert!(sequence.bernoulli(&[-1], 0.).is_err());
        assert!(sequence.bernoulli(&[-1], 1.).is_err());
        assert!(sequence.bernoulli(&[-1], 0.5).is_err());
        let zero = sequence.bernoulli(&[], -0.).unwrap();
        let one = sequence.bernoulli(&[2], 1.).unwrap();
        let empty = sequence.bernoulli(&[0], 0.5).unwrap();
        let empty_uniform = sequence.uniform_f32(&[0]).unwrap();
        let uniform = sequence.uniform_f32(&[3]).unwrap();
        let mask = sequence.bernoulli(&[5], probability).unwrap();
        let yes = graph.constant(&[], &[1.]).unwrap();
        let accept = sequence.commit_if(&mut graph, &yes).unwrap();
        let program = graph
            .compile(
                &mut compiler,
                &[
                    uniform,
                    mask,
                    reference.bits[0].clone(),
                    zero,
                    one,
                    empty,
                    empty_uniform,
                    accept,
                ],
            )
            .unwrap();
        for start in [0, u32::MAX as u64 - 3] {
            let mut session = program
                .session(random.initial_state(&client, [42, 999], start).unwrap())
                .unwrap();
            for step in 1..=4 {
                let output = session.run(&[]).unwrap();
                let bits = output[2].to_vec::<i32>().unwrap();
                let expected: Vec<_> = bits
                    .iter()
                    .map(|&word| ((word as u32 >> 8) as f32) / 16_777_216.)
                    .collect();
                assert_eq!(output[0].to_vec::<f32>().unwrap(), expected[..3]);
                assert_eq!(
                    output[1].to_vec::<f32>().unwrap(),
                    expected[3..]
                        .iter()
                        .map(|&value| if value < probability { 1. } else { 0. })
                        .collect::<Vec<_>>()
                );
                assert_eq!(output[3].to_vec::<f32>().unwrap()[0].to_bits(), 0);
                assert_eq!(output[4].to_vec::<f32>().unwrap(), [1., 1.]);
                assert!(output[5].to_vec::<f32>().unwrap().is_empty());
                assert!(output[6].to_vec::<f32>().unwrap().is_empty());
                assert_eq!(output[7].to_vec::<f32>().unwrap(), [1.]);
                let counter = start + step * 8;
                assert_eq!(
                    session
                        .state(&random.slots()[2])
                        .unwrap()
                        .to_vec::<i32>()
                        .unwrap(),
                    [counter as i32]
                );
                assert_eq!(
                    session
                        .state(&random.slots()[3])
                        .unwrap()
                        .to_vec::<i32>()
                        .unwrap(),
                    [(counter >> 32) as i32]
                );
            }
        }
    }
    assert_eq!(compiler.stats().misses, 4);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_endpoint_only_sampling_is_accepted_at_last_counter() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let random = ThreefryState::new(&mut graph).unwrap();
    let mut sequence = random.begin(&graph).unwrap();
    let zeros = sequence.bernoulli(&[2], 0.).unwrap();
    let ones = sequence.bernoulli(&[2], 1.).unwrap();
    let yes = graph.constant(&[], &[1.]).unwrap();
    let accept = sequence.commit_if(&mut graph, &yes).unwrap();
    let program = graph
        .compile(&mut compiler, &[zeros, ones, accept])
        .unwrap();
    let mut session = program
        .session(random.initial_state(&client, [1, 2], u64::MAX).unwrap())
        .unwrap();
    for _ in 0..3 {
        let output = session.run(&[]).unwrap();
        assert_eq!(output[0].to_vec::<f32>().unwrap(), [0., 0.]);
        assert_eq!(output[1].to_vec::<f32>().unwrap(), [1., 1.]);
        assert_eq!(output[2].to_vec::<f32>().unwrap(), [1.]);
        for (slot, word) in random.slots().iter().zip([1, 2, -1, -1]) {
            assert_eq!(
                session.state(slot).unwrap().to_vec::<i32>().unwrap(),
                [word]
            );
        }
    }
}

#[test]
fn proposals_validate_ownership_and_stale_versions_without_partial_writes() {
    let mut graph = StateGraph::default();
    let random = ThreefryState::new(&mut graph).unwrap();
    let yes = graph.constant(&[], &[1.]).unwrap();
    let mut first = random.begin(&graph).unwrap();
    let mut stale = random.begin(&graph).unwrap();
    first.blocks(&[2]).unwrap();
    stale.blocks(&[3]).unwrap();
    first.commit_if(&mut graph, &yes).unwrap();
    assert!(stale.commit_if(&mut graph, &yes).is_err());
    let mut foreign = StateGraph::default();
    assert!(random.begin(&foreign).is_err());
    assert!(
        random
            .begin(&graph)
            .unwrap()
            .commit_if(&mut foreign, &yes)
            .is_err()
    );
    // Failed validation must not change versions: another outstanding proposal
    // must remain usable after the failed commit.
    let survivor = random.begin(&graph).unwrap();
    let invalid = graph.constant(&[1], &[1.]).unwrap();
    assert!(
        random
            .begin(&graph)
            .unwrap()
            .commit_if(&mut graph, &invalid)
            .is_err()
    );
    survivor.commit_if(&mut graph, &yes).unwrap();
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_sequence_chains_draws_and_rejects_whole_sequence_after_wrap() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let random = ThreefryState::new(&mut graph).unwrap();
    let condition = graph.input(&[]).unwrap();
    let words = random
        .slots()
        .iter()
        .map(|s| graph.read(s).unwrap())
        .collect::<Vec<_>>();
    let reference =
        threefry2x32_blocks([&words[0], &words[1]], [&words[2], &words[3]], &[5]).unwrap();
    let mut sequence = random.begin(&graph).unwrap();
    assert!(sequence.blocks(&[-1]).is_err());
    let empty = sequence.blocks(&[0]).unwrap();
    let first = sequence.blocks(&[2]).unwrap();
    let second = sequence.blocks(&[3]).unwrap();
    let accepted = sequence.commit_if(&mut graph, &condition).unwrap();
    let program = graph
        .compile(
            &mut compiler,
            &[
                first[0].clone(),
                second[0].clone(),
                reference.bits[0].clone(),
                accepted,
                empty[0].clone(),
            ],
        )
        .unwrap();
    for start in [0, u32::MAX as u64 - 2, u64::MAX - 1, u64::MAX - 4] {
        let mut session = program
            .session(
                random
                    .initial_state(&client, [0xdeadbeef, 42], start)
                    .unwrap(),
            )
            .unwrap();
        let mut expected = start;
        for request in [0., 1., 1., 0., 1.] {
            let output = session
                .run(&[&client.buffer(&[], &[request]).unwrap()])
                .unwrap();
            let mut actual = output[0].to_vec::<i32>().unwrap();
            actual.extend(output[1].to_vec::<i32>().unwrap());
            assert_eq!(actual, output[2].to_vec::<i32>().unwrap());
            assert!(output[4].to_vec::<i32>().unwrap().is_empty());
            let accept = request != 0. && expected.checked_add(5).is_some();
            assert_eq!(
                output[3].to_vec::<f32>().unwrap(),
                [if accept { 1. } else { 0. }]
            );
            if accept {
                expected += 5;
            }
            for (slot, word) in random.slots().iter().zip([
                0xdeadbeef_u32 as i32,
                42,
                expected as i32,
                (expected >> 32) as i32,
            ]) {
                assert_eq!(
                    session.state(slot).unwrap().to_vec::<i32>().unwrap(),
                    [word]
                );
            }
        }
    }
    assert_eq!(compiler.stats().misses, 1);
}
