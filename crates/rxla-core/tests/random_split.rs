use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer, random::ThreefryState};

fn reference(key: [u32; 2], counter: u64) -> [u32; 2] {
    let k = [key[0], key[1], key[0] ^ key[1] ^ 0x1bd11bda];
    let mut x = [
        (counter as u32).wrapping_add(k[0]),
        ((counter >> 32) as u32).wrapping_add(k[1]),
    ];
    let rotations = [13, 15, 26, 6, 17, 29, 16, 24];
    for r in 0..20 {
        x[0] = x[0].wrapping_add(x[1]);
        x[1] = x[1].rotate_left(rotations[r % 8]) ^ x[0];
        if r % 4 == 3 {
            let s = r / 4 + 1;
            x[0] = x[0].wrapping_add(k[s % 3]);
            x[1] = x[1].wrapping_add(k[(s + 1) % 3]).wrapping_add(s as u32);
        }
    }
    x
}

#[test]
fn reset_validation_preserves_versions_and_success_invalidates_sequences() {
    assert_eq!(reference([0; 2], 0), [0x6b200159, 0x99ba4efe]);
    let mut g = StateGraph::default();
    let rng = ThreefryState::new(&mut g).unwrap();
    let one = g.constant(&[], &[1.]).unwrap();
    let key = g.scalar_i32(7).unwrap();
    let seq = rng.begin(&g).unwrap();
    for bad in [
        g.input_i32(&[1]).unwrap(),
        Tracer::default().input_i32_scalar().unwrap(),
    ] {
        assert!(rng.reset_key_if(&mut g, [&key, &bad], &one).is_err());
        assert!(rng.reset_key_if(&mut g, [&bad, &key], &one).is_err());
    }
    for bad in [
        g.constant(&[1], &[1.]).unwrap(),
        Tracer::default().constant(&[], &[1.]).unwrap(),
    ] {
        assert!(rng.reset_key_if(&mut g, [&key; 2], &bad).is_err());
    }
    seq.commit_if(&mut g, &one).unwrap();
    let stale = rng.begin(&g).unwrap();
    rng.reset_key_if(&mut g, [&key; 2], &one).unwrap();
    assert!(stale.commit_if(&mut g, &one).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_split_and_child_reset_share_parent_acceptance() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let parent = ThreefryState::new(&mut g).unwrap();
    let child = ThreefryState::new(&mut g).unwrap();
    let requested = g.input(&[]).unwrap();
    let mut seq = parent.begin(&g).unwrap();
    assert!(seq.split_keys(usize::MAX).is_err());
    let empty = seq.split_keys(0).unwrap();
    seq.blocks(&[]).unwrap(); // block 0 belongs to an ordinary draw
    let keys = seq.split_keys(2).unwrap(); // blocks 1 and 2
    seq.blocks(&[]).unwrap(); // block 3 remains reserved even if its output is unused
    let accepted = seq.commit_if(&mut g, &requested).unwrap();
    let selected = [
        keys[0].narrow(0, 1, 1).unwrap().reshape(&[]).unwrap(),
        keys[1].narrow(0, 1, 1).unwrap().reshape(&[]).unwrap(),
    ];
    child
        .reset_key_if(&mut g, [&selected[0], &selected[1]], &accepted)
        .unwrap();
    let mut seq = child.begin(&g).unwrap();
    let draw = seq.blocks(&[]).unwrap();
    seq.commit_if(&mut g, &accepted).unwrap();
    let program = g
        .compile_outputs(
            &mut compiler,
            &[
                keys[0].clone(),
                keys[1].clone(),
                draw[0].clone(),
                draw[1].clone(),
                accepted,
                empty[0].clone(),
            ],
        )
        .unwrap();
    let yes = client.buffer(&[], &[1.]).unwrap();
    let no = client.buffer(&[], &[0.]).unwrap();
    for start in [0, u32::MAX as u64 - 2, u64::MAX - 2] {
        let mut initial = parent.initial_state(&client, [17, 29], start).unwrap();
        initial.extend(child.initial_state(&client, [5, 6], 9).unwrap());
        let mut session = program.session(initial).unwrap();
        let mut counter = start;
        let mut child_key = [5, 6];
        let mut child_counter = 9;
        for requested in [false, true, true] {
            let expected_keys = [
                reference([17, 29], counter.wrapping_add(1)),
                reference([17, 29], counter.wrapping_add(2)),
            ];
            let accept = requested && counter.checked_add(4).is_some();
            let out = session.run(&[if requested { &yes } else { &no }]).unwrap();
            for w in 0..2 {
                assert_eq!(
                    out[w].to_vec::<i32>().unwrap(),
                    expected_keys.map(|k| k[w] as i32)
                );
            }
            if accept {
                counter += 4;
                child_key = expected_keys[1];
                child_counter = 0;
            }
            let expected_draw = reference(child_key, child_counter);
            for w in 0..2 {
                assert_eq!(
                    out[w + 2].to_vec::<i32>().unwrap(),
                    [expected_draw[w] as i32]
                );
            }
            assert_eq!(
                out[4].to_vec::<f32>().unwrap(),
                [if accept { 1. } else { 0. }]
            );
            assert!(out[5].to_vec::<i32>().unwrap().is_empty());
            if accept {
                child_counter += 1;
            }
            for (rng, values) in [
                (&parent, [17, 29, counter as u32, (counter >> 32) as u32]),
                (
                    &child,
                    [
                        child_key[0],
                        child_key[1],
                        child_counter as u32,
                        (child_counter >> 32) as u32,
                    ],
                ),
            ] {
                for (slot, expected) in rng.slots().iter().zip(values) {
                    assert_eq!(
                        session.state(slot).unwrap().to_vec::<i32>().unwrap(),
                        [expected as i32]
                    );
                }
            }
        }
    }
    assert_eq!(compiler.stats().misses, 1);
}
