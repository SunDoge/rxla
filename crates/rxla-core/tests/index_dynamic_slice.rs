use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer};

#[test]
fn rejects_invalid_integer_dynamic_regions() {
    let g = Tracer::default();
    let x = g.input_i32(&[3, 4]).unwrap();
    let s = g.input_i32_scalar().unwrap();
    for sizes in [vec![1], vec![-1, 2], vec![4, 1], vec![1, 5]] {
        assert!(x.dynamic_slice(&[s.clone(), s.clone()], &sizes).is_err());
    }
    assert!(x.dynamic_slice(&[], &[1, 1]).is_err());
    for bad in [
        g.input_i32(&[1]).unwrap(),
        Tracer::default().input_i32_scalar().unwrap(),
    ] {
        assert!(x.dynamic_slice(&[bad.clone(), s.clone()], &[1, 1]).is_err());
        assert!(x.dynamic_update_slice(&x, &[s.clone(), bad]).is_err());
    }
    for update in [
        g.input_i32(&[1]).unwrap(),
        g.input_i32(&[4, 1]).unwrap(),
        Tracer::default().input_i32(&[1, 1]).unwrap(),
    ] {
        assert!(
            x.dynamic_update_slice(&update, &[s.clone(), s.clone()])
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_integer_runtime_regions_clamp_and_preserve_source() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input_i32(&[3, 4]).unwrap();
    let patch = g.input_i32(&[2, 2]).unwrap();
    let starts = [g.input_i32_scalar().unwrap(), g.input_i32_scalar().unwrap()];
    let read = x.dynamic_slice(&starts, &[2, 2]).unwrap();
    let updated = x.dynamic_update_slice(&patch, &starts).unwrap();
    let exe = g.compile_many(&client, &[read, updated]).unwrap();
    let data = [
        i32::MIN,
        i32::MAX,
        16_777_217,
        -16_777_217,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ];
    let replacement = [i32::MAX, i32::MIN, -16_777_217, 16_777_217];
    let input = client.buffer(&[3, 4], &data).unwrap();
    let patch = client.buffer(&[2, 2], &replacement).unwrap();
    for [a, b] in [[i32::MIN, i32::MAX], [1, 1], [i32::MAX, i32::MIN], [0, 0]] {
        let aa = client.buffer(&[], &[a]).unwrap();
        let bb = client.buffer(&[], &[b]).unwrap();
        let out = exe.execute(&[&input, &patch, &aa, &bb]).unwrap();
        let mut expected = data;
        let mut read = Vec::new();
        for row in 0..2 {
            for col in 0..2 {
                let offset = (a.clamp(0, 1) as usize + row) * 4 + b.clamp(0, 2) as usize + col;
                read.push(data[offset]);
                expected[offset] = replacement[row * 2 + col];
            }
        }
        assert_eq!(out[0].to_vec::<i32>().unwrap(), read);
        assert_eq!(out[1].to_vec::<i32>().unwrap(), expected);
        assert_eq!(input.to_vec::<i32>().unwrap(), data);
    }
    for shape in [vec![], vec![0, 2]] {
        let g = Tracer::default();
        let x = g.input_i32(&shape).unwrap();
        let p = g.input_i32(&shape).unwrap();
        let starts: Vec<_> = (0..shape.len())
            .map(|_| g.scalar_i32(i32::MAX).unwrap())
            .collect();
        let exe = g
            .compile_many(
                &client,
                &[
                    x.dynamic_slice(&starts, &shape).unwrap(),
                    x.dynamic_update_slice(&p, &starts).unwrap(),
                ],
            )
            .unwrap();
        let data = if shape.is_empty() {
            vec![i32::MIN]
        } else {
            vec![]
        };
        let replacement = if shape.is_empty() {
            vec![i32::MAX]
        } else {
            vec![]
        };
        let input = client.buffer(&shape, &data).unwrap();
        let patch = client.buffer(&shape, &replacement).unwrap();
        let out = exe.execute(&[&input, &patch]).unwrap();
        assert_eq!(out[0].to_vec::<i32>().unwrap(), data);
        assert_eq!(out[1].to_vec::<i32>().unwrap(), replacement);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_resident_token_history_rejects_full_capacity_without_overwriting() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let history = g.state_i32(&[4]).unwrap();
    let position = g.state_i32(&[]).unwrap();
    let token = g.input_i32_scalar().unwrap();
    let pos = g.read(&position).unwrap();
    let accepted = g
        .scalar_i32(0)
        .unwrap()
        .le_mask(&pos)
        .unwrap()
        .mul(&pos.le_mask(&g.scalar_i32(3).unwrap()).unwrap())
        .unwrap();
    let next = g
        .read(&history)
        .unwrap()
        .dynamic_update_slice(&token.reshape(&[1]).unwrap(), std::slice::from_ref(&pos))
        .unwrap();
    g.write_outputs_if(
        &accepted,
        &[
            (&history, next),
            (&position, pos.wrapping_add_scalar(1).unwrap()),
        ],
    )
    .unwrap();
    let program = g
        .compile_outputs(&mut compiler, &[g.read(&history).unwrap(), accepted])
        .unwrap();
    let mut session = program
        .session(
            program
                .zero_state(&[history.clone(), position.clone()])
                .unwrap(),
        )
        .unwrap();
    let mut expected = [0; 4];
    for (n, token) in [i32::MAX, 16_777_217, i32::MIN, -16_777_217, 99]
        .into_iter()
        .enumerate()
    {
        let input = client.buffer(&[], &[token]).unwrap();
        let out = session.run(&[&input]).unwrap();
        if n < 4 {
            expected[n] = token;
        }
        assert_eq!(out[0].to_vec::<i32>().unwrap(), expected);
        assert_eq!(
            out[1].to_vec::<f32>().unwrap(),
            [if n < 4 { 1. } else { 0. }]
        );
        assert_eq!(
            session.state(&position).unwrap().to_vec::<i32>().unwrap(),
            [(n + 1).min(4) as i32]
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}
