use rxla_core::{CacheLimits, Client, Compiler, StateGraph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_finite_result_gate_preserves_state_and_allows_retry() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let data = g.state(&[2]).unwrap();
    let position = g.state_i32(&[]).unwrap();
    let input = g.input(&[2]).unwrap();
    let proposed = g.read(&data).unwrap().add(&input).unwrap();
    let count = proposed.is_finite_mask().unwrap().sum(&[0], false).unwrap();
    let accepted = count.eq_mask(&g.constant(&[], &[2.]).unwrap()).unwrap();
    let next = g.read(&position).unwrap().wrapping_add_scalar(1).unwrap();
    g.write_many_if(&accepted, &[(&data, proposed.clone()), (&position, next)])
        .unwrap();
    let program = g.compile(&mut compiler, &[count, proposed]).unwrap();
    let mut session = program
        .session(vec![
            (data.clone(), client.buffer(&[2], &[1., 2.]).unwrap()),
            (position.clone(), client.buffer(&[], &[0]).unwrap()),
        ])
        .unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input = client.buffer(&[2], &[3., bad]).unwrap();
        let output = session.run(&[&input]).unwrap();
        assert_eq!(output[0].to_vec::<f32>().unwrap(), [1.]);
        assert!(!output[1].to_vec::<f32>().unwrap()[1].is_finite());
        assert_eq!(
            session.state(&data).unwrap().to_vec::<f32>().unwrap(),
            [1., 2.]
        );
        assert_eq!(
            session.state(&position).unwrap().to_vec::<i32>().unwrap(),
            [0]
        );
    }
    let retry = client.buffer(&[2], &[3., 4.]).unwrap();
    assert_eq!(
        session.run(&[&retry]).unwrap()[0].to_vec::<f32>().unwrap(),
        [2.]
    );
    assert_eq!(
        session.state(&data).unwrap().to_vec::<f32>().unwrap(),
        [4., 6.]
    );
    assert_eq!(
        session.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [1]
    );
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_guarded_cache_and_position_update_together() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let data = g.state(&[2]).unwrap();
    let position = g.state_i32(&[]).unwrap();
    let enabled = g.input(&[]).unwrap();
    let value = g.input(&[1]).unwrap();
    let old_position = g.read(&position).unwrap();
    let in_bounds = old_position.le_mask(&g.scalar_i32(1).unwrap()).unwrap();
    let condition = enabled.mul(&in_bounds).unwrap();
    let next = old_position.wrapping_add_scalar(1).unwrap();
    let updated = g
        .read(&data)
        .unwrap()
        .dynamic_update_slice(&value, &[old_position])
        .unwrap();
    g.write_many_if(&condition, &[(&data, updated), (&position, next)])
        .unwrap();
    let program = g
        .compile(
            &mut compiler,
            &[g.read(&data).unwrap(), g.read(&position).unwrap()],
        )
        .unwrap();
    let mut session = program
        .session(vec![
            (data.clone(), client.buffer(&[2], &[0., 0.]).unwrap()),
            (position.clone(), client.buffer(&[], &[0]).unwrap()),
        ])
        .unwrap();
    for (enabled, value, expected, position_value) in [
        (1., 10., [10., 0.], 1),
        (0., f32::NAN, [10., 0.], 1),
        (1., 20., [10., 20.], 2),
        (1., f32::INFINITY, [10., 20.], 2),
    ] {
        let flag = client.buffer(&[], &[enabled]).unwrap();
        let value = client.buffer(&[1], &[value]).unwrap();
        let output = session.run(&[&flag, &value]).unwrap();
        assert_eq!(output[0].to_vec::<f32>().unwrap(), expected);
        assert_eq!(output[1].to_vec::<i32>().unwrap(), [position_value]);
        assert_eq!(
            session.state(&data).unwrap().to_vec::<f32>().unwrap(),
            expected
        );
        assert_eq!(
            session.state(&position).unwrap().to_vec::<i32>().unwrap(),
            [position_value]
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}
