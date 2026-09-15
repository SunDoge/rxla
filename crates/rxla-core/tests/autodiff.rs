use rxla_core::{CacheLimits, Client, Compiler, Graph, StateGraph};

#[test]
fn grad_rejects_unsupported_non_scalar_and_non_leaf_requests() {
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    assert!(x.grad(std::slice::from_ref(&x)).is_err());
    let loss = x.sum(&[0], false).unwrap();
    assert!(loss.grad(&[x.neg().unwrap()]).is_err());
    assert!(loss.grad(&[Graph::default().input(&[2]).unwrap()]).is_err());
    assert!(
        x.is_finite_mask()
            .unwrap()
            .sum(&[0], false)
            .unwrap()
            .grad(&[x])
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_grad_smooth_unary_rules() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[3]).unwrap();
    let ops = [
        x.neg(),
        x.exp(),
        x.expm1(),
        x.log(),
        x.log1p(),
        x.sqrt(),
        x.rsqrt(),
        x.tanh(),
        x.erf(),
    ];
    let outputs: Vec<_> = ops
        .into_iter()
        .map(|op| {
            op.unwrap()
                .sum(&[0], false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0)
        })
        .collect();
    let exe = g.compile_many(&client, &outputs).unwrap();
    let values = [0.2f32, 0.7, 1.3];
    let actual = exe.run_many(&[&values]).unwrap();
    for (i, x) in values.into_iter().map(f64::from).enumerate() {
        let expected = [
            -1.,
            x.exp(),
            x.exp(),
            1. / x,
            1. / (1. + x),
            0.5 / x.sqrt(),
            -0.5 / x.powf(1.5),
            1. - x.tanh().powi(2),
            2. / std::f64::consts::PI.sqrt() * (-x * x).exp(),
        ];
        for (op, value) in expected.into_iter().enumerate() {
            assert!(
                (actual[op][i] as f64 - value).abs() < 3e-6,
                "unary {op}/{i}"
            );
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_grad_batched_matmul_vector_promotion_and_empty_broadcast() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let a = g.input(&[2, 1, 3]).unwrap();
    let b = g.input(&[1, 3, 2]).unwrap();
    let loss = a.matmul(&b).unwrap().sum(&[0, 1, 2], false).unwrap();
    let gradients = loss.grad(&[a, b]).unwrap();
    assert_eq!(
        g.compile_many(&client, &gradients)
            .unwrap()
            .run_many(&[&[1., 2., 3., 4., 5., 6.], &[1., 2., 3., 4., 5., 6.]])
            .unwrap(),
        [vec![3., 7., 11., 3., 7., 11.], vec![5., 5., 7., 7., 9., 9.]]
    );
    let g = Graph::default();
    let a = g.input(&[3]).unwrap();
    let b = g.input(&[3]).unwrap();
    let grads = a.matmul(&b).unwrap().grad(&[a, b]).unwrap();
    assert_eq!(
        g.compile_many(&client, &grads)
            .unwrap()
            .run_many(&[&[1., 2., 3.], &[4., 5., 6.]])
            .unwrap(),
        [vec![4., 5., 6.], vec![1., 2., 3.]]
    );
    let g = Graph::default();
    let x = g.input(&[]).unwrap();
    let loss = x
        .broadcast_to(&[0, 2])
        .unwrap()
        .sum(&[0, 1], false)
        .unwrap();
    let grad = loss.grad(&[x]).unwrap();
    assert_eq!(
        g.compile_many(&client, &grad)
            .unwrap()
            .run_many(&[&[1.]])
            .unwrap(),
        [vec![0.]]
    );
}

fn reference(x: &[f64], w: &[f64], b: &[f64]) -> f64 {
    let mut loss = 0.;
    for row in 0..2 {
        for col in 0..2 {
            let z = b[col] + (0..3).map(|k| x[row * 3 + k] * w[k * 2 + col]).sum::<f64>();
            let y = z / (1. + (-z).exp());
            loss += y * y / 4.;
        }
    }
    loss
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_grad_mlp_matches_independent_f64_finite_differences() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2, 3]).unwrap();
    let w = g.input(&[3, 2]).unwrap();
    let b = g.input(&[2]).unwrap();
    let h = x
        .matmul(&w)
        .unwrap()
        .add(&b.broadcast_to(&[2, 2]).unwrap())
        .unwrap()
        .silu()
        .unwrap();
    let loss = h.mul(&h).unwrap().mean(&[0, 1], false).unwrap();
    let grads = loss.grad(&[x, w, b]).unwrap();
    let exe = g.compile_many(&client, &grads).unwrap();
    for scale in [1., -0.7] {
        let mut data = vec![
            vec![0.2, -0.5, 0.7, 1., 0.3, -0.4],
            vec![0.5, -0.2, 0.3, 0.6, -0.8, 0.7],
            vec![0.1, -0.3],
        ];
        for values in &mut data {
            for v in values {
                *v *= scale;
            }
        }
        let f32data: Vec<Vec<f32>> = data
            .iter()
            .map(|v| v.iter().map(|x| *x as f32).collect())
            .collect();
        let actual = exe
            .run_many(&f32data.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .unwrap();
        for group in 0..3 {
            for index in 0..data[group].len() {
                let old = data[group][index];
                data[group][index] = old + 1e-5;
                let plus = reference(&data[0], &data[1], &data[2]);
                data[group][index] = old - 1e-5;
                let minus = reference(&data[0], &data[1], &data[2]);
                data[group][index] = old;
                let expected = (plus - minus) / 2e-5;
                assert!(
                    (actual[group][index] as f64 - expected).abs() < 2e-5,
                    "gradient {group}/{index}: {} != {expected}",
                    actual[group][index]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_grad_layouts_shared_inputs_disconnected_and_second_derivative() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[1, 2]).unwrap();
    let unused = g.input(&[3]).unwrap();
    let y = x.broadcast_to(&[3, 2]).unwrap().transpose(&[1, 0]).unwrap();
    let loss = y.mul(&y).unwrap().sum(&[0, 1], false).unwrap();
    let grads = loss.grad(&[x.clone(), unused, x]).unwrap();
    assert_eq!(
        g.compile_many(&client, &grads)
            .unwrap()
            .run_many(&[&[2., 3.], &[1., 2., 3.]])
            .unwrap(),
        [vec![12., 18.], vec![0.; 3], vec![12., 18.]]
    );
    let g = Graph::default();
    let x = g.input(&[]).unwrap();
    let cubic = x.mul(&x).unwrap().mul(&x).unwrap();
    let first = cubic.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let second = first.grad(&[x]).unwrap();
    assert_eq!(
        g.compile_many(&client, &second)
            .unwrap()
            .run_many(&[&[2.]])
            .unwrap(),
        [vec![12.]]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_resident_linear_regression_training_compiles_once() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let weight = graph.state(&[]).unwrap();
    let bias = graph.state(&[]).unwrap();
    let w = graph.read(&weight).unwrap();
    let b = graph.read(&bias).unwrap();
    let x = graph.input(&[4]).unwrap();
    let target = graph.input(&[4]).unwrap();
    let error = x
        .mul(&w.broadcast_to(&[4]).unwrap())
        .unwrap()
        .add(&b.broadcast_to(&[4]).unwrap())
        .unwrap()
        .sub(&target)
        .unwrap();
    let loss = error.mul(&error).unwrap().mean(&[0], false).unwrap();
    let grad = loss.grad(&[w.clone(), b.clone()]).unwrap();
    graph
        .write_many(&[
            (&weight, &w.sub(&grad[0].mul_scalar(0.1).unwrap()).unwrap()),
            (&bias, &b.sub(&grad[1].mul_scalar(0.1).unwrap()).unwrap()),
        ])
        .unwrap();
    let program = graph.compile(&mut compiler, &[loss]).unwrap();
    let mut session = program
        .session(vec![
            (weight.clone(), client.buffer(&[], &[0.]).unwrap()),
            (bias.clone(), client.buffer(&[], &[0.]).unwrap()),
        ])
        .unwrap();
    let xb = client.buffer(&[4], &[-1., 0., 1., 2.]).unwrap();
    let yb = client.buffer(&[4], &[-1., 1., 3., 5.]).unwrap();
    let mut last = f32::INFINITY;
    for _ in 0..100 {
        let loss = session.run(&[&xb, &yb]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(loss.is_finite() && loss <= last + 1e-7);
        last = loss;
    }
    assert!(last < 1e-8);
    assert!((session.state(&weight).unwrap().to_vec::<f32>().unwrap()[0] - 2.).abs() < 1e-4);
    assert!((session.state(&bias).unwrap().to_vec::<f32>().unwrap()[0] - 1.).abs() < 1e-4);
    assert_eq!(compiler.stats().misses, 1);
}
