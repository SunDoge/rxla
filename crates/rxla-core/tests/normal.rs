use rxla_core::{
    CacheLimits, Client, Compiler, StateGraph, Tracer,
    random::{ThreefryState, normal_f32_from_bits},
};

#[test]
fn validates_normal_word_shapes_and_graphs() {
    let g = Tracer::default();
    let a = g.input_i32(&[2]).unwrap();
    for b in [
        g.input_i32_scalar().unwrap(),
        Tracer::default().input_i32(&[2]).unwrap(),
    ] {
        assert!(normal_f32_from_bits([&a, &b]).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_normal_reparameterization_gradients_reuse_the_resident_draw() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let rng = ThreefryState::new(&mut g).unwrap();
    let mean = g.input(&[3]).unwrap();
    let log_std = g.input(&[3]).unwrap();
    let requested = g.input(&[]).unwrap();
    let mut sequence = rng.begin(&g).unwrap();
    let noise = sequence.normal_f32(&[2, 3]).unwrap();
    let sample = mean
        .broadcast_to(&[2, 3])
        .unwrap()
        .add(
            &log_std
                .exp()
                .unwrap()
                .broadcast_to(&[2, 3])
                .unwrap()
                .mul(&noise)
                .unwrap(),
        )
        .unwrap();
    let loss = sample
        .mul(&sample)
        .unwrap()
        .sum(&[0, 1], false)
        .unwrap()
        .mul_scalar(0.5)
        .unwrap();
    let gradients = loss.grad(&[mean.clone(), log_std.clone()]).unwrap();
    let mean_hessian = gradients[0]
        .sum(&[0], false)
        .unwrap()
        .grad(&[mean.clone(), log_std.clone()])
        .unwrap();
    let scale_hessian = gradients[1]
        .sum(&[0], false)
        .unwrap()
        .grad(&[mean, log_std])
        .unwrap();
    let accepted = sequence.commit_if(&mut g, &requested).unwrap();
    let program = g
        .compile(
            &mut compiler,
            &[
                noise,
                sample,
                loss,
                gradients[0].clone(),
                gradients[1].clone(),
                mean_hessian[0].clone(),
                mean_hessian[1].clone(),
                scale_hessian[0].clone(),
                scale_hessian[1].clone(),
                accepted,
            ],
        )
        .unwrap();
    let means = [0.25f32, -1., 2.];
    let log_stds = [-0.7f32, 0., 0.4];
    let inputs = [
        client.buffer(&[3], &means).unwrap(),
        client.buffer(&[3], &log_stds).unwrap(),
    ];
    let yes = client.buffer(&[], &[1.]).unwrap();
    let no = client.buffer(&[], &[0.]).unwrap();
    let mut session = program
        .session(rng.initial_state(&client, [17, 29], 0).unwrap())
        .unwrap();
    let mut previous: Option<Vec<Vec<f32>>> = None;
    for (accept, counter) in [(&no, 0), (&yes, 6), (&yes, 12)] {
        let output: Vec<_> = session
            .run(&[&inputs[0], &inputs[1], accept])
            .unwrap()
            .iter()
            .map(|b| b.to_vec::<f32>().unwrap())
            .collect();
        let mut expected = vec![
            vec![0.; 6],
            vec![0.; 1],
            vec![0.; 3],
            vec![0.; 3],
            vec![2.; 3],
            vec![0.; 3],
            vec![0.; 3],
            vec![0.; 3],
        ];
        for i in 0..6 {
            let j = i % 3;
            let shifted = (log_stds[j] as f64).exp() * output[0][i] as f64;
            let z = means[j] as f64 + shifted;
            expected[0][i] = z;
            expected[1][0] += 0.5 * z * z;
            expected[2][j] += z;
            expected[3][j] += z * shifted;
            expected[5][j] += shifted;
            expected[6][j] += shifted;
            expected[7][j] += shifted * shifted + z * shifted;
        }
        for (actual, reference) in output[1..9].iter().zip(expected) {
            for (&a, e) in actual.iter().zip(reference) {
                assert!((a as f64 - e).abs() < 1e-5 + 1e-5 * e.abs(), "{a} != {e}");
            }
        }
        assert_eq!(
            session
                .state(&rng.slots()[2])
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [counter]
        );
        assert_eq!(
            session
                .state(&rng.slots()[3])
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [0]
        );
        assert_eq!(output[9], [if counter == 0 { 0. } else { 1. }]);
        if counter == 6 {
            assert_eq!(&output[..9], previous.as_ref().unwrap());
        }
        if counter == 12 {
            assert_ne!(output[0], previous.as_ref().unwrap()[0]);
        }
        previous = Some(output[..9].to_vec());
    }
    assert_eq!(compiler.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_normal_matches_f64_formula_at_grid_boundaries() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let words = [0i32, 1, 511, 512, i32::MAX, i32::MIN, -512, -1];
    let mut a = Vec::new();
    let mut b = Vec::new();
    for x in words {
        for y in words {
            a.push(x);
            b.push(y);
        }
    }
    let g = Tracer::default();
    let x = g.input_i32(&[8, 8]).unwrap();
    let y = g.input_i32(&[8, 8]).unwrap();
    let z = normal_f32_from_bits([&x, &y]).unwrap();
    let executable = g.compile(&client, &z).unwrap();
    let inputs = [
        client.buffer(&[8, 8], &a).unwrap(),
        client.buffer(&[8, 8], &b).unwrap(),
    ];
    let output = executable.execute(&[&inputs[0], &inputs[1]]).unwrap()[0]
        .to_vec::<f32>()
        .unwrap();
    for ((x, y), actual) in a.iter().zip(&b).zip(output) {
        let u = ((*x as u32 >> 9) as f64 + 0.5) / 8_388_608.;
        let v = (*y as u32 >> 8) as f64 / 16_777_216.;
        let expected = (-2. * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
        assert!(actual.is_finite());
        assert!(
            (actual as f64 - expected).abs() < 5e-6,
            "{x} {y}: {actual} != {expected}"
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_normal_sequence_replays_and_commits_only_accepted_draws() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let rng = ThreefryState::new(&mut g).unwrap();
    let requested = g.input(&[]).unwrap();
    let mut sequence = rng.begin(&g).unwrap();
    assert!(sequence.normal_f32(&[-1]).is_err());
    let empty = sequence.normal_f32(&[0, 2]).unwrap();
    let noise = sequence.normal_f32(&[4096]).unwrap();
    let scalar = sequence.normal_f32(&[]).unwrap();
    let accepted = sequence.commit_if(&mut g, &requested).unwrap();
    let program = g
        .compile(&mut compiler, &[noise, scalar, empty, accepted])
        .unwrap();
    let yes = client.buffer(&[], &[1.]).unwrap();
    let no = client.buffer(&[], &[0.]).unwrap();
    for start in [0, u32::MAX as u64 - 10, u64::MAX - 10] {
        let mut session = program
            .session(rng.initial_state(&client, [17, 29], start).unwrap())
            .unwrap();
        let rejected = session.run(&[&no]).unwrap();
        let actual = session.run(&[&yes]).unwrap();
        for i in 0..3 {
            assert_eq!(
                rejected[i].to_vec::<f32>().unwrap(),
                actual[i].to_vec::<f32>().unwrap()
            );
        }
        let wraps = start.checked_add(4097).is_none();
        assert_eq!(
            actual[3].to_vec::<f32>().unwrap(),
            [if wraps { 0. } else { 1. }]
        );
        let end = if wraps { start } else { start + 4097 };
        for (slot, expected) in rng.slots()[2..]
            .iter()
            .zip([end as i32, (end >> 32) as i32])
        {
            assert_eq!(
                session.state(slot).unwrap().to_vec::<i32>().unwrap(),
                [expected]
            );
        }
        let values = actual[0].to_vec::<f32>().unwrap();
        assert!(values.iter().all(|x| x.is_finite()));
        if start == 0 {
            let mean = values.iter().map(|&x| x as f64).sum::<f64>() / values.len() as f64;
            let second =
                values.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / values.len() as f64;
            assert!(
                mean.abs() < 0.08 && (second - 1.).abs() < 0.12,
                "{mean} {second}"
            );
        }
    }
    assert_eq!(compiler.stats().misses, 1);
}
