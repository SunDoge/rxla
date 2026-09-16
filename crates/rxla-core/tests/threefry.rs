use rxla_core::random::threefry2x32_blocks;
use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer, random::threefry2x32};

#[test]
fn batch_blocks_validate_scalar_words_and_sizes() {
    let graph = Tracer::default();
    let scalar = graph.input_i32_scalar().unwrap();
    let vector = graph.input_i32(&[1]).unwrap();
    let foreign = Tracer::default().input_i32_scalar().unwrap();
    for shape in [vec![-1], vec![i64::MAX, 2], vec![i32::MAX as i64 + 1]] {
        assert!(threefry2x32_blocks([&scalar; 2], [&scalar; 2], &shape).is_err());
    }
    for bad in [&vector, &foreign] {
        assert!(threefry2x32_blocks([&scalar, bad], [&scalar; 2], &[3]).is_err());
        assert!(threefry2x32_blocks([&scalar; 2], [bad, &scalar], &[3]).is_err());
        assert!(threefry2x32_blocks([&scalar; 2], [&scalar, bad], &[3]).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_batch_blocks_allocate_counters_and_commit_conditionally() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let key = [0x12345678_u32, 0xdeadbeef];
    for shape in [vec![], vec![2, 3], vec![0, 3]] {
        let count = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<i64>() as usize
        };
        let mut graph = StateGraph::default();
        let low_slot = graph.state_i32(&[]).unwrap();
        let high_slot = graph.state_i32(&[]).unwrap();
        let low = graph.read(&low_slot).unwrap();
        let high = graph.read(&high_slot).unwrap();
        let keys = key.map(|k| graph.scalar_i32(k as i32).unwrap());
        let accept = graph.input(&[]).unwrap();
        let draw = threefry2x32_blocks([&keys[0], &keys[1]], [&low, &high], &shape).unwrap();
        graph
            .write_many_if(
                &accept,
                &[
                    (&low_slot, draw.next_counter[0].clone()),
                    (&high_slot, draw.next_counter[1].clone()),
                ],
            )
            .unwrap();
        let program = graph
            .compile(
                &mut compiler,
                &[
                    draw.bits[0].clone(),
                    draw.bits[1].clone(),
                    draw.counter_wrapped,
                ],
            )
            .unwrap();
        for start in [0_u64, 0xfffffffe, u64::MAX - 2] {
            let mut counter = start;
            let mut session = program
                .session(vec![
                    (
                        low_slot.clone(),
                        client.buffer(&[], &[counter as i32]).unwrap(),
                    ),
                    (
                        high_slot.clone(),
                        client.buffer(&[], &[(counter >> 32) as i32]).unwrap(),
                    ),
                ])
                .unwrap();
            for accepted in [true, false, true, true] {
                let mask = client
                    .buffer(&[], &[if accepted { 1. } else { 0. }])
                    .unwrap();
                let actual = session.run(&[&mask]).unwrap();
                let expected: Vec<_> = (0..count)
                    .map(|i| {
                        let c = counter.wrapping_add(i as u64);
                        host_reference(key, [c as u32, (c >> 32) as u32])
                    })
                    .collect();
                for word in 0..2 {
                    assert_eq!(actual[word].dimensions().unwrap(), shape);
                    assert_eq!(
                        actual[word].to_vec::<i32>().unwrap(),
                        expected.iter().map(|r| r[word] as i32).collect::<Vec<_>>()
                    );
                }
                let (next, wrapped) = counter.overflowing_add(count as u64);
                assert_eq!(
                    actual[2].to_vec::<f32>().unwrap(),
                    [if wrapped { 1. } else { 0. }]
                );
                if accepted {
                    counter = next;
                }
                assert_eq!(
                    session.state(&low_slot).unwrap().to_vec::<i32>().unwrap(),
                    [counter as i32]
                );
                assert_eq!(
                    session.state(&high_slot).unwrap().to_vec::<i32>().unwrap(),
                    [(counter >> 32) as i32]
                );
                session = program.session(session.into_state()).unwrap();
            }
        }
    }
    assert_eq!(compiler.stats().misses, 3);
}

fn host_reference(key: [u32; 2], counter: [u32; 2]) -> [u32; 2] {
    let keys = [key[0], key[1], key[0] ^ key[1] ^ 0x1bd11bda];
    let mut words = [
        counter[0].wrapping_add(key[0]),
        counter[1].wrapping_add(key[1]),
    ];
    let rotations = [[13, 15, 26, 6], [17, 29, 16, 24]];
    for group in 0..5 {
        for rotation in rotations[group % 2] {
            words[0] = words[0].wrapping_add(words[1]);
            words[1] = words[1].rotate_left(rotation) ^ words[0];
        }
        words[0] = words[0].wrapping_add(keys[(group + 1) % 3]);
        words[1] = words[1]
            .wrapping_add(keys[(group + 2) % 3])
            .wrapping_add(group as u32 + 1);
    }
    words
}

#[test]
fn validates_all_key_and_counter_words_before_building() {
    let graph = Tracer::default();
    let good = graph.input_i32(&[3]).unwrap();
    let scalar = graph.input_i32_scalar().unwrap();
    let foreign = Tracer::default().input_i32(&[3]).unwrap();
    for bad in [&scalar, &foreign] {
        assert!(threefry2x32([&good, bad], [&good, &good]).is_err());
        assert!(threefry2x32([&good, &good], [bad, &good]).is_err());
        assert!(threefry2x32([&good, &good], [&good, bad]).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_threefry_matches_random123_known_answers() {
    // Official Threefry2x32-20 vectors (counter, key, result):
    // https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    let cases: [([u32; 2], [u32; 2], [u32; 2]); 3] = [
        ([0, 0], [0, 0], [0x6b200159, 0x99ba4efe]),
        ([u32::MAX; 2], [u32::MAX; 2], [0x1cb996fc, 0xbb002be7]),
        (
            [0x243f6a88, 0x85a308d3],
            [0x13198a2e, 0x03707344],
            [0xc4923a9c, 0x483df7a0],
        ),
    ];
    for (counter, key, expected) in cases {
        assert_eq!(host_reference(key, counter), expected);
    }
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    // Scalar invocations change the entire key/counter without recompilation.
    // Vector mode executes all three KATs elementwise, not one shared key.
    for shape in [vec![], vec![3], vec![0, 2]] {
        let graph = Tracer::default();
        let k0 = graph.input_i32(&shape).unwrap();
        let k1 = graph.input_i32(&shape).unwrap();
        let c0 = graph.input_i32(&shape).unwrap();
        let c1 = graph.input_i32(&shape).unwrap();
        let result = threefry2x32([&k0, &k1], [&c0, &c1]).unwrap();
        let executable = compiler
            .compile_many(&graph, &result.map(Into::into))
            .unwrap();
        for offset in 0..3 {
            let count = if shape.is_empty() {
                1
            } else {
                shape.iter().product::<i64>() as usize
            };
            let rows: Vec<_> = (0..count).map(|i| cases[(i + offset) % 3]).collect();
            let words: Vec<Vec<i32>> = (0..4)
                .map(|word| {
                    rows.iter()
                        .map(|(c, k, _)| {
                            if word < 2 {
                                k[word] as i32
                            } else {
                                c[word - 2] as i32
                            }
                        })
                        .collect()
                })
                .collect();
            let buffers: Vec<_> = words
                .iter()
                .map(|w| client.buffer(&shape, w).unwrap())
                .collect();
            let output = executable
                .execute(&buffers.iter().collect::<Vec<_>>())
                .unwrap();
            for word in 0..2 {
                let expected: Vec<_> = rows.iter().map(|(_, _, r)| r[word] as i32).collect();
                assert_eq!(output[word].dimensions().unwrap(), shape);
                assert_eq!(output[word].to_vec::<i32>().unwrap(), expected);
            }
        }
    }
    assert_eq!(compiler.stats().misses, 3);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_resident_counter_carry_wrap_and_resume() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let low_slot = graph.state_i32(&[4]).unwrap();
    let high_slot = graph.state_i32(&[4]).unwrap();
    let low = graph.read(&low_slot).unwrap();
    let high = graph.read(&high_slot).unwrap();
    let key = [0x12345678_u32, 0xdeadbeef];
    let keys = key.map(|k| graph.constant_i32(&[4], &[k as i32; 4]).unwrap());
    let bits = threefry2x32([&keys[0], &keys[1]], [&low, &high]).unwrap();
    let next_low = low.wrapping_add_scalar(1).unwrap();
    let zero = graph.constant_i32(&[4], &[0; 4]).unwrap();
    let carry = next_low
        .le_mask(&zero)
        .unwrap()
        .mul(&zero.le_mask(&next_low).unwrap())
        .unwrap();
    let next_high = carry
        .select(&high.wrapping_add_scalar(1).unwrap(), &high)
        .unwrap();
    graph
        .write_many(&[(&low_slot, &next_low), (&high_slot, &next_high)])
        .unwrap();
    let program = graph.compile(&mut compiler, &bits.map(Into::into)).unwrap();
    let initial = [0_u64, (7_u64 << 32) | 0xfffffffe, 0xffffffff, u64::MAX];
    let fresh = || {
        vec![
            (
                low_slot.clone(),
                client.buffer(&[4], &initial.map(|c| c as i32)).unwrap(),
            ),
            (
                high_slot.clone(),
                client
                    .buffer(&[4], &initial.map(|c| (c >> 32) as i32))
                    .unwrap(),
            ),
        ]
    };
    let mut session = program.session(fresh()).unwrap();
    let idle = program.session(fresh()).unwrap();
    let mut counters = initial;
    for step in 0..20 {
        if step == 8 {
            session = program.session(session.into_state()).unwrap();
        }
        let output = session.run(&[]).unwrap();
        let expected = counters.map(|c| host_reference(key, [c as u32, (c >> 32) as u32]));
        for word in 0..2 {
            assert_eq!(
                output[word].to_vec::<i32>().unwrap(),
                expected.map(|r| r[word] as i32)
            );
        }
        counters = counters.map(|c| c.wrapping_add(1));
        assert_eq!(
            session.state(&low_slot).unwrap().to_vec::<i32>().unwrap(),
            counters.map(|c| c as i32)
        );
        assert_eq!(
            session.state(&high_slot).unwrap().to_vec::<i32>().unwrap(),
            counters.map(|c| (c >> 32) as i32)
        );
        assert_eq!(
            idle.state(&low_slot).unwrap().to_vec::<i32>().unwrap(),
            initial.map(|c| c as i32)
        );
        assert_eq!(
            idle.state(&high_slot).unwrap().to_vec::<i32>().unwrap(),
            initial.map(|c| (c >> 32) as i32)
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}
