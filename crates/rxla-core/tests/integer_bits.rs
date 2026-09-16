use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer};

#[test]
fn validates_bitwise_shapes_owners_and_shift_counts() {
    let graph = Tracer::default();
    let x = graph.input_i32(&[2]).unwrap();
    let short = graph.input_i32(&[1]).unwrap();
    let foreign = Tracer::default().input_i32(&[2]).unwrap();
    for rhs in [&short, &foreign] {
        assert!(x.bitwise_and(rhs).is_err());
        assert!(x.bitwise_or(rhs).is_err());
        assert!(x.bitwise_xor(rhs).is_err());
    }
    for bits in [32, 33, u32::MAX] {
        assert!(x.shift_left(bits).is_err());
        assert!(x.shift_right_logical(bits).is_err());
        assert!(x.shift_right_arithmetic(bits).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_integer_bits_match_rust_bit_patterns() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    for shape in [vec![8], vec![], vec![0, 3]] {
        let graph = Tracer::default();
        let x = graph.input_i32(&shape).unwrap();
        let y = graph.input_i32(&shape).unwrap();
        let mut roots = vec![
            x.bitwise_and(&y).unwrap(),
            x.bitwise_or(&y).unwrap(),
            x.bitwise_xor(&y).unwrap(),
            x.bitwise_not().unwrap(),
        ];
        for bits in 0..32 {
            roots.extend([
                x.shift_left(bits).unwrap(),
                x.shift_right_logical(bits).unwrap(),
                x.shift_right_arithmetic(bits).unwrap(),
            ]);
        }
        let executable = compiler.compile_many(&graph, &roots).unwrap();
        let count = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<i64>() as usize
        };
        let values = [
            i32::MIN,
            i32::MAX,
            -1,
            0,
            1,
            16_777_217,
            0x55555555,
            0xaaaaaaaa_u32 as i32,
        ];
        for offset in [0, 3] {
            let a: Vec<_> = (0..count).map(|i| values[(i + offset) % 8]).collect();
            let b: Vec<_> = (0..count).map(|i| values[(i + offset + 1) % 8]).collect();
            let ab = client.buffer(&shape, &a).unwrap();
            let bb = client.buffer(&shape, &b).unwrap();
            let actual = executable.execute(&[&ab, &bb]).unwrap();
            let mut expected: Vec<Vec<i32>> = vec![
                a.iter().zip(&b).map(|(&x, &y)| x & y).collect(),
                a.iter().zip(&b).map(|(&x, &y)| x | y).collect(),
                a.iter().zip(&b).map(|(&x, &y)| x ^ y).collect(),
                a.iter().map(|&x| !x).collect(),
            ];
            for bits in 0..32 {
                expected.extend([
                    a.iter().map(|&x| x.wrapping_shl(bits)).collect(),
                    a.iter().map(|&x| ((x as u32) >> bits) as i32).collect(),
                    a.iter().map(|&x| x >> bits).collect(),
                ]);
            }
            for (buffer, expected) in actual.iter().zip(expected) {
                assert_eq!(buffer.dimensions().unwrap(), shape);
                assert_eq!(buffer.to_vec::<i32>().unwrap(), expected);
            }
        }
    }
    assert_eq!(compiler.stats().misses, 3);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_resident_bitwise_transition_and_nondifferentiable_float_selection() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let state = graph.state_i32(&[3]).unwrap();
    let old = graph.read(&state).unwrap();
    let next = old
        .bitwise_xor(&old.shift_left(13).unwrap())
        .unwrap()
        .bitwise_xor(&old.shift_right_logical(17).unwrap())
        .unwrap()
        .wrapping_add_scalar(3)
        .unwrap();
    graph.write(&state, &next).unwrap();
    let program = graph.compile(&mut compiler, &[next]).unwrap();
    let mut expected = [1_i32, i32::MIN, 0xdeadbeef_u32 as i32];
    let mut session = program
        .session(vec![(
            state.clone(),
            client.buffer(&[3], &expected).unwrap(),
        )])
        .unwrap();
    for _ in 0..20 {
        expected =
            expected.map(|x| (x ^ x.wrapping_shl(13) ^ ((x as u32) >> 17) as i32).wrapping_add(3));
        assert_eq!(
            session.run(&[]).unwrap()[0].to_vec::<i32>().unwrap(),
            expected
        );
        assert_eq!(
            session.state(&state).unwrap().to_vec::<i32>().unwrap(),
            expected
        );
    }
    assert_eq!(compiler.stats().misses, 1);
    let graph = Tracer::default();
    let x = graph.input(&[3]).unwrap();
    let selected = x
        .argmax(0, false)
        .unwrap()
        .shift_left(1)
        .unwrap()
        .bitwise_not()
        .unwrap()
        .to_f32()
        .unwrap();
    let grad = selected.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let executable = compiler.compile_many(&graph, &[selected, grad]).unwrap();
    let x = client.buffer(&[3], &[1., 3., 2.]).unwrap();
    let output = executable.execute(&[&x]).unwrap();
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [-3.]);
    assert_eq!(output[1].to_vec::<f32>().unwrap(), [0.; 3]);
}
