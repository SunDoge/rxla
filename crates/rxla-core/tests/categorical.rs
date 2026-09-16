use rxla_core::{Client, Tracer, random::categorical_from_bits};

#[test]
fn topk_sampling_validates_candidate_shape() {
    use rxla_core::random::topk_categorical_from_bits;
    let g = Tracer::default();
    let x = g.input(&[2, 5]).unwrap();
    let bits = g.input_i32(&[2, 2]).unwrap();
    for (k, axis) in [(0, 1), (6, 1), (2, 2)] {
        assert!(topk_categorical_from_bits(&x, &bits, k, axis).is_err());
    }
    assert!(topk_categorical_from_bits(&x, &g.input_i32(&[2, 5]).unwrap(), 2, 1).is_err());
    assert!(
        topk_categorical_from_bits(&x, &Tracer::default().input_i32(&[2, 2]).unwrap(), 2, 1)
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_topk_sampling_validates_discarded_logits_and_matches_f64() {
    use rxla_core::random::topk_categorical_from_bits;
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let rows = [
        [0., 10., 9., 0., 0.],
        [f32::NAN, 10., 9., 0., 0.],
        [0., f32::INFINITY, 9., 0., 0.],
        [f32::NEG_INFINITY; 5],
        [4., 4., 4., 1., 0.],
        [
            f32::NEG_INFINITY,
            3.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
    ];
    for axis in [0, 1] {
        for k in [1, 2, 5] {
            let shape = if axis == 1 { [6, 5] } else { [5, 6] };
            let bits_shape = if axis == 1 {
                [6, k as i64]
            } else {
                [k as i64, 6]
            };
            let g = Tracer::default();
            let x = g.input(&shape).unwrap();
            let bits = g.input_i32(&bits_shape).unwrap();
            let sample = topk_categorical_from_bits(&x, &bits, k, axis).unwrap();
            let exe = g
                .compile_many(&client, &[sample.indices, sample.valid])
                .unwrap();
            let offset = |row: usize, col: usize, width: usize| {
                if axis == 1 {
                    row * width + col
                } else {
                    col * 6 + row
                }
            };
            let mut values = vec![0.; 30];
            let mut words = vec![0; 6 * k];
            let mut expected = vec![0; 6];
            let random = [0_u32, u32::MAX, 0x80000000, 0x40000000, 0xc0000000];
            for (row, logits) in rows.iter().enumerate() {
                for (col, &value) in logits.iter().enumerate() {
                    values[offset(row, col, 5)] = value;
                }
                for col in 0..k {
                    words[offset(row, col, k)] = random[col] as i32;
                }
                if ![0, 4, 5].contains(&row) {
                    continue;
                }
                let mut order: Vec<_> = (0..5).collect();
                order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
                let maximum = logits[order[0]] as f64;
                let mut best = f64::NEG_INFINITY;
                for (rank, &original) in order.iter().take(k).enumerate() {
                    let u = ((random[rank] >> 9) as f64 + 0.5) / 8_388_608.;
                    let score = logits[original] as f64 - maximum - (-u.ln()).ln();
                    if score > best {
                        best = score;
                        expected[row] = original as i32;
                    }
                }
            }
            let x = client.buffer(&shape, &values).unwrap();
            let bits = client.buffer(&bits_shape, &words).unwrap();
            let out = exe.execute(&[&x, &bits]).unwrap();
            assert_eq!(
                out[0].to_vec::<i32>().unwrap(),
                expected,
                "axis={axis}, k={k}"
            );
            assert_eq!(out[1].to_vec::<f32>().unwrap(), [1., 0., 0., 0., 1., 1.]);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_topk_sequence_only_reserves_candidates_and_rejects_invalid_rows() {
    use rxla_core::{CacheLimits, Compiler, StateGraph, random::ThreefryState};
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let rng = ThreefryState::new(&mut g).unwrap();
    let x = g.input(&[2, 4]).unwrap();
    let mut sequence = rng.begin(&g).unwrap();
    assert!(sequence.topk_categorical(&x, 0, 1).is_err());
    assert!(
        sequence
            .topk_categorical(&Tracer::default().input(&[2, 4]).unwrap(), 2, 1)
            .is_err()
    );
    let first = sequence.topk_categorical(&x, 2, 1).unwrap();
    let second = sequence.topk_categorical(&x, 2, 1).unwrap();
    let valid = first
        .valid
        .mul(&second.valid)
        .unwrap()
        .min(&[0], false)
        .unwrap();
    let accepted = sequence.commit_if(&mut g, &valid).unwrap();
    let program = g
        .compile_outputs(&mut compiler, &[first.indices, second.indices, accepted])
        .unwrap();
    let mut session = program
        .session(rng.initial_state(&client, [3, 7], 0).unwrap())
        .unwrap();
    let bad = client
        .buffer(&[2, 4], &[f32::NAN, 3., 2., 1., 0., 3., 2., 1.])
        .unwrap();
    assert_eq!(
        session.run(&[&bad]).unwrap()[2].to_vec::<f32>().unwrap(),
        [0.]
    );
    assert_eq!(
        session
            .state(&rng.slots()[2])
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [0]
    );
    let good = client
        .buffer(&[2, 4], &[0., 3., 2., 1., 0., 3., 2., 1.])
        .unwrap();
    for counter in [8, 16] {
        let out = session.run(&[&good]).unwrap();
        assert_eq!(out[2].to_vec::<f32>().unwrap(), [1.]);
        for indices in &out[..2] {
            assert!(
                indices
                    .to_vec::<i32>()
                    .unwrap()
                    .iter()
                    .all(|&i| i == 1 || i == 2)
            );
        }
        assert_eq!(
            session
                .state(&rng.slots()[2])
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [counter]
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_sequence_rejection_resume_and_wrap_are_atomic() {
    use rxla_core::{CacheLimits, Compiler, StateGraph, random::ThreefryState};
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let rng = ThreefryState::new(&mut graph).unwrap();
    let tokens = graph.state_i32(&[2]).unwrap();
    let logits = graph.input(&[2, 3]).unwrap();
    let requested = graph.input(&[]).unwrap();
    let mut sequence = rng.begin(&graph).unwrap();
    assert!(sequence.categorical(&logits, 2).is_err());
    assert!(
        sequence
            .categorical(&Tracer::default().input(&[2, 3]).unwrap(), 1)
            .is_err()
    );
    let first = sequence.categorical(&logits, 1).unwrap();
    let second = sequence.categorical(&logits, 1).unwrap();
    let condition = first
        .valid
        .min(&[0], false)
        .unwrap()
        .mul(&requested)
        .unwrap();
    let accepted = sequence.commit_if(&mut graph, &condition).unwrap();
    graph
        .write_outputs_if(&accepted, &[(&tokens, second.indices.clone())])
        .unwrap();
    let program = graph
        .compile_outputs(&mut compiler, &[first.indices, second.indices, accepted])
        .unwrap();
    let good = client.buffer(&[2, 3], &[0., 1., -1., 2., 0., 1.]).unwrap();
    let bad = client
        .buffer(&[2, 3], &[0., 1., -1., f32::NAN, 0., 1.])
        .unwrap();
    let yes = client.buffer(&[], &[1.]).unwrap();
    let no = client.buffer(&[], &[0.]).unwrap();
    for start in [u32::MAX as u64 - 5, u64::MAX - 5] {
        let mut initial = rng.initial_state(&client, [17, 29], start).unwrap();
        initial.push((tokens.clone(), client.buffer(&[2], &[-1, -1]).unwrap()));
        let mut session = program.session(initial).unwrap();
        let rejected = session.run(&[&good, &no]).unwrap();
        assert_eq!(rejected[2].to_vec::<f32>().unwrap(), [0.]);
        let invalid = session.run(&[&bad, &yes]).unwrap();
        assert_eq!(invalid[2].to_vec::<f32>().unwrap(), [0.]);
        assert_eq!(
            session.state(&tokens).unwrap().to_vec::<i32>().unwrap(),
            [-1, -1]
        );
        let out = session.run(&[&good, &yes]).unwrap();
        for i in 0..2 {
            assert_eq!(
                out[i].to_vec::<i32>().unwrap(),
                rejected[i].to_vec::<i32>().unwrap()
            );
        }
        let (next, wrapped) = start.overflowing_add(12);
        let counter = if wrapped { start } else { next };
        assert_eq!(
            out[2].to_vec::<f32>().unwrap(),
            [if wrapped { 0. } else { 1. }]
        );
        for (slot, expected) in rng.slots()[2..]
            .iter()
            .zip([counter as i32, (counter >> 32) as i32])
        {
            assert_eq!(
                session.state(slot).unwrap().to_vec::<i32>().unwrap(),
                [expected]
            );
        }
        assert_eq!(
            session.state(&tokens).unwrap().to_vec::<i32>().unwrap(),
            if wrapped {
                vec![-1, -1]
            } else {
                out[1].to_vec::<i32>().unwrap()
            }
        );
        // Snapshot values to the host and restore into independent device buffers.
        // No hidden sampler state or additional compilation is needed.
        let snapshot = rng
            .slots()
            .iter()
            .chain([&tokens])
            .map(|slot| {
                let buffer = session.state(slot).unwrap();
                (
                    slot.clone(),
                    client
                        .buffer(
                            &buffer.dimensions().unwrap(),
                            &buffer.to_vec::<i32>().unwrap(),
                        )
                        .unwrap(),
                )
            })
            .collect();
        let mut restored = program.session(snapshot).unwrap();
        for _ in 0..4 {
            let a = session.run(&[&good, &yes]).unwrap();
            let b = restored.run(&[&good, &yes]).unwrap();
            for i in 0..2 {
                assert_eq!(a[i].to_vec::<i32>().unwrap(), b[i].to_vec::<i32>().unwrap());
            }
            assert_eq!(a[2].to_vec::<f32>().unwrap(), b[2].to_vec::<f32>().unwrap());
        }
    }
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_threefry_categorical_frequency_smoke() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let zero = g.scalar_i32(0).unwrap();
    let draw = rxla_core::random::threefry2x32_blocks([&zero; 2], [&zero; 2], &[8192, 3]).unwrap();
    let logits = g
        .constant(&[3], &[0.1_f32.ln(), 0.3_f32.ln(), 0.6_f32.ln()])
        .unwrap()
        .broadcast_to(&[8192, 3])
        .unwrap();
    let sample = categorical_from_bits(&logits, &draw.bits[0], 1).unwrap();
    let exe = g
        .compile_many(&client, &[sample.indices, sample.valid])
        .unwrap();
    let out = exe.execute(&[]).unwrap();
    let mut counts = [0; 3];
    for index in out[0].to_vec::<i32>().unwrap() {
        counts[index as usize] += 1;
    }
    assert!(out[1].to_vec::<f32>().unwrap().iter().all(|&v| v == 1.));
    for (count, probability) in counts.into_iter().zip([0.1, 0.3, 0.6]) {
        assert!(
            (count as f64 / 8192. - probability).abs() < 0.025,
            "{counts:?}"
        );
    }
}

#[test]
fn validates_shapes_graphs_and_axis() {
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    let bits = g.input_i32(&[2, 3]).unwrap();
    assert!(categorical_from_bits(&x, &bits, 2).is_err());
    assert!(categorical_from_bits(&x, &g.input_i32(&[3]).unwrap(), 1).is_err());
    assert!(categorical_from_bits(&x, &Tracer::default().input_i32(&[2, 3]).unwrap(), 1).is_err());
    let empty = g.input(&[2, 0]).unwrap();
    assert!(categorical_from_bits(&empty, &g.input_i32(&[2, 0]).unwrap(), 1).is_err());
    let draw = categorical_from_bits(&x, &bits, 0).unwrap();
    assert_eq!(draw.indices.shape(), &[3]);
    assert_eq!(draw.valid.shape(), &[3]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_categorical_matches_reference_and_rejects_invalid_rows() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    // Exercise both category-axis positions, including empty batch dimensions.
    for (shape, axis) in [(vec![7, 3], 1), (vec![3, 7], 0), (vec![0, 3], 1)] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let bits = g.input_i32(&shape).unwrap();
        let draw = categorical_from_bits(&x, &bits, axis).unwrap();
        let exe = g
            .compile_many(&client, &[draw.indices, draw.valid])
            .unwrap();
        let rows = [
            [0., 0., 0.],
            [f32::MAX, f32::MAX, f32::MAX],
            [f32::NEG_INFINITY, -100., f32::NEG_INFINITY],
            [f32::NEG_INFINITY; 3],
            [0., f32::INFINITY, 1.],
            [0., f32::NAN, 1.],
            [-1., 0., 1.],
        ];
        let count = shape.iter().product::<i64>() as usize;
        let mut values = vec![0.; count];
        let mut words = vec![0; count];
        let mut expected = Vec::new();
        for (row, logits) in rows.iter().enumerate().take(count / 3) {
            let random = [0_u32, u32::MAX, 0x80000000];
            for col in 0..3 {
                let index = if axis == 1 {
                    row * 3 + col
                } else {
                    col * 7 + row
                };
                values[index] = logits[col];
                words[index] = random[col] as i32;
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let valid = logits.iter().any(|x| x.is_finite())
                && logits
                    .iter()
                    .all(|x| x.is_finite() || *x == f32::NEG_INFINITY);
            let winner = if valid {
                (0..3)
                    .max_by(|&a, &b| {
                        let score = |i: usize| {
                            let u = ((random[i] >> 9) as f64 + 0.5) / 8_388_608.;
                            logits[i] as f64 - max - (-u.ln()).ln()
                        };
                        score(a).total_cmp(&score(b))
                    })
                    .unwrap() as i32
            } else {
                0
            };
            expected.push(winner);
        }
        let x = client.buffer(&shape, &values).unwrap();
        let bits = client.buffer(&shape, &words).unwrap();
        for _ in 0..2 {
            let out = exe.execute(&[&x, &bits]).unwrap();
            assert_eq!(out[0].to_vec::<i32>().unwrap(), expected);
            assert_eq!(
                out[1].to_vec::<f32>().unwrap(),
                [1., 1., 1., 0., 0., 0., 1.][..count / 3]
            );
        }
    }
}
