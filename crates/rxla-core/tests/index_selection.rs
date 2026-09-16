use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer};

#[test]
fn index_selection_validates_operands() {
    let g = Tracer::default();
    let mask = g.input(&[2]).unwrap();
    let value = g.input_i32(&[2]).unwrap();
    let wrong = g.scalar_i32(1).unwrap();
    let foreign = Tracer::default().input_i32(&[2]).unwrap();
    for invalid in [wrong, foreign] {
        assert!(mask.select(&invalid, &value).is_err());
        assert!(mask.select(&value, &invalid).is_err());
    }
    assert_eq!(mask.select(&value, &value).unwrap().shape(), [2]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_integer_selection_preserves_exact_values() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![6], vec![], vec![0, 2]] {
        let g = Tracer::default();
        let mask = g.input(&shape).unwrap();
        let a = g.input_i32(&shape).unwrap();
        let b = g.input_i32(&shape).unwrap();
        let selected = mask.select(&a, &b).unwrap();
        let exe = g.compile_many(&client, &[selected]).unwrap();
        let len = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<i64>() as usize
        };
        let av = [i32::MAX, i32::MIN, 16_777_217, -16_777_217, 123, -789];
        let bv = [i32::MIN, i32::MAX, -16_777_219, 16_777_219, -456, 999];
        let ab = client.buffer(&shape, &av[..len]).unwrap();
        let bb = client.buffer(&shape, &bv[..len]).unwrap();
        for masks in [
            [0., -0., 1., -1., f32::NAN, f32::INFINITY],
            [1.; 6],
            [0.; 6],
        ] {
            let mb = client.buffer(&shape, &masks[..len]).unwrap();
            let expected: Vec<_> = (0..len)
                .map(|i| if masks[i] != 0. { av[i] } else { bv[i] })
                .collect();
            assert_eq!(
                exe.execute(&[&mb, &ab, &bb]).unwrap()[0]
                    .to_vec::<i32>()
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_masked_state_advances_only_active_positions() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let positions = g.state_i32(&[3]).unwrap();
    let active = g.input(&[3]).unwrap();
    let old = g.read(&positions).unwrap();
    let next = active
        .select(&old.wrapping_add_scalar(1).unwrap(), &old)
        .unwrap();
    g.write(&positions, &next).unwrap();
    let program = g.compile_outputs(&mut compiler, &[next]).unwrap();
    let mut session = program
        .session(vec![(
            positions.clone(),
            client.buffer(&[3], &[16_777_217, i32::MAX, 0]).unwrap(),
        )])
        .unwrap();
    for (mask, expected) in [
        ([1., 0., 1.], [16_777_218, i32::MAX, 1]),
        ([0., 1., 1.], [16_777_218, i32::MIN, 2]),
        ([0., 0., 0.], [16_777_218, i32::MIN, 2]),
    ] {
        let input = client.buffer(&[3], &mask).unwrap();
        assert_eq!(
            session.run(&[&input]).unwrap()[0].to_vec::<i32>().unwrap(),
            expected
        );
        assert_eq!(
            session.state(&positions).unwrap().to_vec::<i32>().unwrap(),
            expected
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}
