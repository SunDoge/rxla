use rxla_core::{CacheLimits, Client, Compiler, StateGraph, Tracer};

#[test]
fn iota_validates_axis_and_coordinate_range() {
    let g = Tracer::default();
    for (shape, axis) in [
        (vec![], 0),
        (vec![2], 1),
        (vec![-1], 0),
        (vec![i32::MAX as i64 + 2], 0),
    ] {
        assert!(g.iota_i32(&shape, axis).is_err());
    }
    assert_eq!(g.iota_i32(&[2, 0, 3], 1).unwrap().shape(), [2, 0, 3]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_iota_coordinates_on_each_axis_and_empty_arrays() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for dims in [[2, 3, 4], [2, 0, 4]] {
        for axis in 0..3 {
            let g = Tracer::default();
            let indices = g.iota_i32(&dims, axis).unwrap();
            let exe = g.compile_many(&client, &[indices]).unwrap();
            let actual = exe.execute(&[]).unwrap()[0].to_vec::<i32>().unwrap();
            let inner: i64 = dims[axis + 1..].iter().product();
            let expected: Vec<_> = (0..dims.iter().product::<i64>())
                .map(|i| ((i / inner) % dims[axis]) as i32)
                .collect();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_resident_offset_and_iota_generate_position_blocks() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let offset = g.state_i32(&[]).unwrap();
    let start = g.read(&offset).unwrap();
    let positions = g
        .iota_i32(&[4], 0)
        .unwrap()
        .wrapping_add(&start.broadcast_to(&[4]).unwrap())
        .unwrap();
    let features = positions
        .to_f32()
        .unwrap()
        .mul_scalar(0.125)
        .unwrap()
        .sin()
        .unwrap();
    g.write(&offset, &start.wrapping_add_scalar(4).unwrap())
        .unwrap();
    let program = g
        .compile_outputs(&mut compiler, &[positions, features])
        .unwrap();
    let mut session = program
        .session(vec![(offset.clone(), client.buffer(&[], &[3]).unwrap())])
        .unwrap();
    assert_eq!(session.input_count(), 0);
    for block in 0..3 {
        let result = session.run(&[]).unwrap();
        let expected = [3 + block * 4, 4 + block * 4, 5 + block * 4, 6 + block * 4];
        assert_eq!(result[0].to_vec::<i32>().unwrap(), expected);
        for (actual, p) in result[1].to_vec::<f32>().unwrap().into_iter().zip(expected) {
            assert!((actual as f64 - (p as f64 * 0.125).sin()).abs() < 1e-6);
        }
        assert_eq!(
            session.state(&offset).unwrap().to_vec::<i32>().unwrap(),
            [7 + block * 4]
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}
