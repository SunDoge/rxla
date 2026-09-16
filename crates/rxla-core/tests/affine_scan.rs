use rxla_core::{Client, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_streaming_session_preserves_carry_across_calls_and_rejected_updates() {
    use rxla_core::{CacheLimits, Compiler, F32, I32, State, StateGraph};
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let carry = State::<F32>::new(&mut graph, &[2]).unwrap();
    let count = State::<I32>::new(&mut graph, &[]).unwrap();
    let multiplier = graph.input(&[2, 3]).unwrap();
    let offset = graph.input(&[2, 3]).unwrap();
    let accept = graph.input(&[]).unwrap();
    let output = offset
        .affine_scan_from(&multiplier, &carry.read(&graph).unwrap(), 1)
        .unwrap();
    let next = output.narrow(1, 2, 1).unwrap().reshape(&[2]).unwrap();
    let next_count = count.read(&graph).unwrap().wrapping_add_scalar(1).unwrap();
    graph
        .transaction()
        .with(&carry, &next)
        .unwrap()
        .with(&count, &next_count)
        .unwrap()
        .commit_if(&accept)
        .unwrap();
    let program = graph
        .compile(&mut compiler, std::slice::from_ref(&output))
        .unwrap();
    let mut session = program
        .session(vec![
            (
                carry.as_slot().clone(),
                client.buffer(&[2], &[3., -2.]).unwrap(),
            ),
            (count.as_slot().clone(), client.buffer(&[], &[0]).unwrap()),
        ])
        .unwrap();
    let yes = client.buffer(&[], &[1.]).unwrap();
    let no = client.buffer(&[], &[0.]).unwrap();
    let mut reference = [3_f64, -2.];
    let mut accepted_count = 0;
    for (step, accepted) in [true, false, true, true, false, true]
        .into_iter()
        .enumerate()
    {
        let a = [0.5_f32, -1., 0., 1., 0.5, -1.];
        let b: Vec<_> = (0..6).map(|i| (step + i) as f32 - 3.).collect();
        let inputs = [
            client.buffer(&[2, 3], &a).unwrap(),
            client.buffer(&[2, 3], &b).unwrap(),
        ];
        if step == 1 {
            let bad = client.buffer(&[2, 2], &[0.; 4]).unwrap();
            assert!(session.run(&[&bad, &inputs[1], &yes]).is_err());
            assert_eq!(
                session
                    .state(carry.as_slot())
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                reference.map(|x| x as f32)
            );
            assert_eq!(
                session
                    .state(count.as_slot())
                    .unwrap()
                    .to_vec::<i32>()
                    .unwrap(),
                [accepted_count]
            );
        }
        let mut proposed = reference;
        let mut expected = Vec::new();
        for (row, value) in proposed.iter_mut().enumerate() {
            for column in 0..3 {
                let i = row * 3 + column;
                *value = a[i] as f64 * *value + b[i] as f64;
                expected.push(*value as f32);
            }
        }
        let outputs = session
            .run(&[&inputs[0], &inputs[1], if accepted { &yes } else { &no }])
            .unwrap();
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), expected);
        if accepted {
            reference = proposed;
            accepted_count += 1;
        }
        assert_eq!(
            session
                .state(carry.as_slot())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            reference.map(|x| x as f32)
        );
        assert_eq!(
            session
                .state(count.as_slot())
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [accepted_count]
        );
        if step == 2 {
            // Transfer ownership, dropping the old Session between real calls.
            session = program.session(session.into_state()).unwrap();
        }
    }
    graph.compile(&mut compiler, &[output]).unwrap();
    assert_eq!(compiler.stats().misses, 1);
    assert_eq!(compiler.stats().hits, 1);
    assert_eq!(accepted_count, 4);
}

#[test]
fn explicit_initial_state_requires_the_non_scan_shape() {
    let g = Tracer::default();
    let x = g.input(&[2, 5]).unwrap();
    for bad in [
        g.input(&[]).unwrap(),
        g.input(&[2, 1]).unwrap(),
        Tracer::default().input(&[2]).unwrap(),
    ] {
        assert!(x.affine_scan_from(&x, &bad, 1).is_err());
    }
    let initial = g.input(&[2]).unwrap();
    assert!(x.affine_scan_from(&x, &initial, 2).is_err());
    assert!(
        x.affine_scan_from(&g.input(&[5]).unwrap(), &initial, 1)
            .is_err()
    );
    assert!(x.affine_scan_from(&x, &initial, 1).is_ok());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_initial_state_and_symbolic_chunks_preserve_values_and_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let a = g.input(&[2, 5]).unwrap();
    let b = g.input(&[2, 5]).unwrap();
    let initial = g.input(&[2]).unwrap();
    let whole = b.affine_scan_from(&a, &initial, 1).unwrap();
    let first = b
        .narrow(1, 0, 2)
        .unwrap()
        .affine_scan_from(&a.narrow(1, 0, 2).unwrap(), &initial, 1)
        .unwrap();
    let carry = first.narrow(1, 1, 1).unwrap().reshape(&[2]).unwrap();
    let second = b
        .narrow(1, 2, 3)
        .unwrap()
        .affine_scan_from(&a.narrow(1, 2, 3).unwrap(), &carry, 1)
        .unwrap();
    let chunked = rxla_core::Tensor::concatenate(&[first, second], 1).unwrap();
    let targets = [a, b, initial];
    let whole_gradient = whole.sum(&[0, 1], false).unwrap().grad(&targets).unwrap();
    let chunk_gradient = chunked.sum(&[0, 1], false).unwrap().grad(&targets).unwrap();
    let mut outputs = vec![whole, chunked];
    outputs.extend(whole_gradient);
    outputs.extend(chunk_gradient);
    let exe = g.compile_many(&client, &outputs).unwrap();
    let multipliers = [0.5_f32, 0.5, 0., -1., 2., 1., 0.5, 0.5, 0., -1.];
    let offsets = [1_f32, 2., 3., 4., 5., -1., -2., -3., -4., -5.];
    let start = [3_f32, -2.];
    let mut reference = [0_f64; 10];
    let mut da = [0_f64; 10];
    let mut db = [0_f64; 10];
    let mut di = [0_f64; 2];
    for row in 0..2 {
        for column in 0..5 {
            let i = row * 5 + column;
            let previous = if column == 0 {
                start[row] as f64
            } else {
                reference[i - 1]
            };
            reference[i] = multipliers[i] as f64 * previous + offsets[i] as f64;
        }
        for column in (0..5).rev() {
            let i = row * 5 + column;
            db[i] = 1.
                + if column == 4 {
                    0.
                } else {
                    multipliers[i + 1] as f64 * db[i + 1]
                };
            da[i] = db[i]
                * if column == 0 {
                    start[row] as f64
                } else {
                    reference[i - 1]
                };
        }
        di[row] = db[row * 5] * multipliers[row * 5] as f64;
    }
    let inputs = [
        client.buffer(&[2, 5], &multipliers).unwrap(),
        client.buffer(&[2, 5], &offsets).unwrap(),
        client.buffer(&[2], &start).unwrap(),
    ];
    let actual = exe.execute(&[&inputs[0], &inputs[1], &inputs[2]]).unwrap();
    let expected: [&[f64]; 8] = [&reference, &reference, &da, &db, &di, &da, &db, &di];
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(
            actual
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_discounted_returns_stop_at_terminal_boundaries() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let reward = graph.input(&[5]).unwrap();
    let discount = graph.input(&[5]).unwrap();
    let returns = reward
        .flip(&[0])
        .unwrap()
        .affine_scan(&discount.flip(&[0]).unwrap(), 0)
        .unwrap()
        .flip(&[0])
        .unwrap();
    let first = returns.narrow(0, 0, 1).unwrap().sum(&[0], false).unwrap();
    let gradient = first.grad(&[reward]).unwrap().remove(0);
    let exe = graph.compile_many(&client, &[returns, gradient]).unwrap();
    let inputs = [
        client.buffer(&[5], &[1., 2., 3., 4., 5.]).unwrap(),
        client.buffer(&[5], &[0.5, 0., 0.5, 0.5, 0.]).unwrap(),
    ];
    let outputs = exe.execute(&[&inputs[0], &inputs[1]]).unwrap();
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [2., 2., 6.25, 6.5, 5.]);
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [1., 0.5, 0., 0., 0.]);
}

#[test]
fn rejects_invalid_axis_shape_and_owner() {
    let graph = Tracer::default();
    let x = graph.input(&[4]).unwrap();
    assert!(x.affine_scan(&x, 1).is_err());
    assert!(x.affine_scan(&graph.input(&[]).unwrap(), 0).is_err());
    assert!(
        x.affine_scan(&Tracer::default().input(&[4]).unwrap(), 0)
            .is_err()
    );
    let scalar = graph.input(&[]).unwrap();
    assert!(scalar.affine_scan(&scalar, 0).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_recurrence_and_gradients_match_sequential_adjoint() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [[2_i64, 5], [3, 4], [2, 0], [2, 1]] {
        for axis in 0..2 {
            let count = shape.iter().product::<i64>() as usize;
            let a: Vec<_> = (0..count)
                .map(|i| [0_f32, 0.5, -1., 0., 1.][i % 5])
                .collect();
            let b: Vec<_> = (0..count).map(|i| (i % 7) as f32 - 3.).collect();
            let seed: Vec<_> = (0..count).map(|i| [0.5_f32, -1., 2.][i % 3]).collect();
            let stride = shape[axis + 1..].iter().product::<i64>() as usize;
            let length = shape[axis] as usize;
            let mut expected = vec![0_f64; count];
            let mut da = vec![0_f64; count];
            let mut db = vec![0_f64; count];
            for i in 0..count {
                let position = i / stride % length;
                let previous = if position == 0 {
                    0.
                } else {
                    expected[i - stride]
                };
                expected[i] = a[i] as f64 * previous + b[i] as f64;
            }
            for i in (0..count).rev() {
                let position = i / stride % length;
                db[i] = seed[i] as f64
                    + if position + 1 < length {
                        a[i + stride] as f64 * db[i + stride]
                    } else {
                        0.
                    };
                da[i] = db[i]
                    * if position == 0 {
                        0.
                    } else {
                        expected[i - stride]
                    };
            }
            let g = Tracer::default();
            let multiplier = g.input(&shape).unwrap();
            let offset = g.input(&shape).unwrap();
            let cotangent = g.input(&shape).unwrap();
            let output = offset.affine_scan(&multiplier, axis).unwrap();
            let gradients = output.vjp(&[multiplier, offset], &cotangent).unwrap();
            let exe = g
                .compile_many(
                    &client,
                    &[output, gradients[0].clone(), gradients[1].clone()],
                )
                .unwrap();
            let inputs = [
                client.buffer(&shape, &a).unwrap(),
                client.buffer(&shape, &b).unwrap(),
                client.buffer(&shape, &seed).unwrap(),
            ];
            let outputs = exe.execute(&[&inputs[0], &inputs[1], &inputs[2]]).unwrap();
            for (output, reference) in outputs.iter().zip([expected, da, db]) {
                assert_eq!(output.dimensions().unwrap(), shape);
                for (&actual, expected) in output.to_vec::<f32>().unwrap().iter().zip(reference) {
                    assert_eq!(actual as f64, expected, "shape={shape:?} axis={axis}");
                }
            }
        }
    }
}
