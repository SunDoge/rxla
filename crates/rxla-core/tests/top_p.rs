use rxla_core::{Client, Tracer, random::top_p_categorical_from_bits};

#[test]
fn validates_top_p_inputs() {
    let g = Tracer::default();
    let x = g.input(&[4]).unwrap();
    let bits = g.input_i32(&[4]).unwrap();
    for p in [0., -1., 1.1, f32::NAN, f32::INFINITY] {
        assert!(top_p_categorical_from_bits(&x, &bits, p, 0).is_err());
    }
    assert!(top_p_categorical_from_bits(&x, &bits, 0.5, 1).is_err());
    assert!(top_p_categorical_from_bits(&x, &g.input_i32(&[3]).unwrap(), 0.5, 0).is_err());
    assert!(
        top_p_categorical_from_bits(&x, &Tracer::default().input_i32(&[4]).unwrap(), 0.5, 0)
            .is_err()
    );
    assert!(
        top_p_categorical_from_bits(&g.input(&[0]).unwrap(), &g.input_i32(&[0]).unwrap(), 0.5, 0)
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_top_p_cutoff_boundaries_original_validity_and_axis_mapping() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let rows = [
        [0.; 4],
        [f32::NAN, 0., 0., 0.],
        [f32::INFINITY, 0., 0., 0.],
        [f32::NEG_INFINITY; 4],
        [f32::NEG_INFINITY, 10., f32::NEG_INFINITY, f32::NEG_INFINITY],
    ];
    for axis in [0, 1] {
        for (p, winner) in [
            (f32::from_bits(1), 0),
            (0.01, 0),
            (0.25, 0),
            (0.5, 1),
            (0.5001, 2),
            (0.9, 3),
            (1., 3),
        ] {
            let shape = if axis == 1 { [5, 4] } else { [4, 5] };
            let g = Tracer::default();
            let x = g.input(&shape).unwrap();
            let bits = g.input_i32(&shape).unwrap();
            let draw = top_p_categorical_from_bits(&x, &bits, p, axis).unwrap();
            let exe = g
                .compile_many(&client, &[draw.indices, draw.valid])
                .unwrap();
            let mut values = vec![0.; 20];
            let mut words = vec![0; 20];
            for (row, logits) in rows.iter().enumerate() {
                for (col, &value) in logits.iter().enumerate() {
                    let i = if axis == 1 {
                        row * 4 + col
                    } else {
                        col * 5 + row
                    };
                    values[i] = value;
                    words[i] = [0, 0x01000000, i32::MIN, -1][col];
                }
            }
            let x = client.buffer(&shape, &values).unwrap();
            let bits = client.buffer(&shape, &words).unwrap();
            let out = exe.execute(&[&x, &bits]).unwrap();
            assert_eq!(
                out[0].to_vec::<i32>().unwrap(),
                [winner, 0, 0, 0, 1],
                "axis={axis} p={p}"
            );
            assert_eq!(out[1].to_vec::<f32>().unwrap(), [1., 0., 0., 0., 1.]);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_top_p_includes_crossing_category_and_sequence_reserves_full_shape() {
    use rxla_core::{CacheLimits, Compiler, StateGraph, random::ThreefryState};
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (p, expected) in [(0.6, 1), (0.8, 2), (1., 3)] {
        let g = Tracer::default();
        let x = g.input(&[4]).unwrap();
        let bits = g.input_i32(&[4]).unwrap();
        let draw = top_p_categorical_from_bits(&x, &bits, p, 0).unwrap();
        let exe = g.compile_many(&client, &[draw.indices]).unwrap();
        let x = client
            .buffer(&[4], &[4_f32.ln(), 2_f32.ln(), 0., 0.])
            .unwrap();
        let bits = client.buffer(&[4], &[0, 0x01000000, i32::MIN, -1]).unwrap();
        assert_eq!(
            exe.execute(&[&x, &bits]).unwrap()[0]
                .to_vec::<i32>()
                .unwrap(),
            [expected]
        );
    }
    let mut g = StateGraph::default();
    let rng = ThreefryState::new(&mut g).unwrap();
    let x = g.input(&[4]).unwrap();
    let mut sequence = rng.begin(&g).unwrap();
    assert!(sequence.top_p_categorical(&x, 0., 0).is_err());
    let sample = sequence.top_p_categorical(&x, 0.5, 0).unwrap();
    let accepted = sequence.commit_if(&mut g, &sample.valid).unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let program = g
        .compile_outputs(&mut compiler, &[sample.indices, accepted])
        .unwrap();
    let mut session = program
        .session(rng.initial_state(&client, [3, 7], 0).unwrap())
        .unwrap();
    for (values, expected_counter) in [([f32::NAN, 0., 0., 0.], 0), ([0.; 4], 4), ([0.; 4], 8)] {
        let x = client.buffer(&[4], &values).unwrap();
        let out = session.run(&[&x]).unwrap();
        assert_eq!(
            out[1].to_vec::<f32>().unwrap(),
            [if expected_counter == 0 { 0. } else { 1. }]
        );
        assert_eq!(
            session
                .state(&rng.slots()[2])
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [expected_counter]
        );
    }
}
