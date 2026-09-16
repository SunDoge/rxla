use rxla_core::{Client, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_signed_prefix_vjp_and_nonuniform_hessian_vector() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for length in [17_usize, 128] {
        let shape = [2, length as i64];
        let data: Vec<f32> = (0..2 * length)
            .map(|i| ((i * 37 % 101) as f32 - 50.) / 12.5)
            .collect();
        let seeds: Vec<f32> = (0..2 * length)
            .map(|i| ((i * 7 % 13) as f32 - 6.) / 7.)
            .collect();
        let direction: Vec<f32> = (0..2 * length)
            .map(|i| ((i * 3 % 11) as f32 - 5.) / 9.)
            .collect();
        let mut expected_gradient = vec![0_f64; data.len()];
        let mut expected_hvp = vec![0_f64; data.len()];
        // Independent dense prefix-softmax formula, only used by the oracle.
        // H*v = sum_j seed_j p_jk (v_k - sum_l p_jl v_l).
        for row in 0..2 {
            let base = row * length;
            for end in 0..length {
                let maximum = data[base..=base + end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max) as f64;
                let exponentials: Vec<_> = data[base..=base + end]
                    .iter()
                    .map(|&x| (x as f64 - maximum).exp())
                    .collect();
                let sum: f64 = exponentials.iter().sum();
                let average: f64 = exponentials
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p / sum * direction[base + i] as f64)
                    .sum();
                for (i, p) in exponentials.iter().enumerate() {
                    let weighted = seeds[base + end] as f64 * p / sum;
                    expected_gradient[base + i] += weighted;
                    expected_hvp[base + i] += weighted * (direction[base + i] as f64 - average);
                }
            }
        }
        assert!(expected_hvp.iter().any(|x| x.abs() > 0.01));
        for tree in [false, true] {
            let graph = Tracer::default();
            let x = graph.input(&shape).unwrap();
            let seed = graph.input(&shape).unwrap();
            let vector = graph.input(&shape).unwrap();
            let y = if tree {
                x.logcumsumexp_tree(1)
            } else {
                x.logcumsumexp(1)
            }
            .unwrap();
            let gradient = y.vjp(std::slice::from_ref(&x), &seed).unwrap().remove(0);
            let hvp = gradient
                .mul(&vector)
                .unwrap()
                .sum(&[0, 1], false)
                .unwrap()
                .grad(&[x])
                .unwrap()
                .remove(0);
            let executable = graph.compile_many(&client, &[gradient, hvp]).unwrap();
            let inputs = [
                client.buffer(&shape, &data).unwrap(),
                client.buffer(&shape, &seeds).unwrap(),
                client.buffer(&shape, &direction).unwrap(),
            ];
            let outputs = executable
                .execute(&[&inputs[0], &inputs[1], &inputs[2]])
                .unwrap();
            for (output, expected) in outputs.iter().zip([&expected_gradient, &expected_hvp]) {
                assert_eq!(output.dimensions().unwrap(), shape);
                for (&actual, &expected) in output.to_vec::<f32>().unwrap().iter().zip(expected) {
                    assert!(
                        (actual as f64 - expected).abs() < 3e-5,
                        "length={length} tree={tree}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_logaddexp_retains_small_increments_and_tie_hessian() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let x = graph.input(&[5]).unwrap();
    let z = graph.input(&[5]).unwrap();
    let y = x.logaddexp(&z).unwrap();
    let gradient = y
        .sum(&[0], false)
        .unwrap()
        .grad(&[x.clone(), z.clone()])
        .unwrap();
    let hessian = gradient[0].sum(&[0], false).unwrap().grad(&[x, z]).unwrap();
    let executable = graph
        .compile_many(
            &client,
            &[
                y,
                gradient[0].clone(),
                gradient[1].clone(),
                hessian[0].clone(),
                hessian[1].clone(),
            ],
        )
        .unwrap();
    let a = [0_f32, 0., 2., 1000., -1000.];
    let b = [-20_f32, -40., 2., 999., -1001.];
    let inputs = [
        client.buffer(&[5], &a).unwrap(),
        client.buffer(&[5], &b).unwrap(),
    ];
    let outputs = executable.execute(&[&inputs[0], &inputs[1]]).unwrap();
    let values: Vec<_> = outputs.iter().map(|o| o.to_vec::<f32>().unwrap()).collect();
    for i in 0..5 {
        let x = a[i] as f64;
        let z = b[i] as f64;
        let expected = x.max(z) + (-(x - z).abs()).exp().ln_1p();
        assert!(
            (values[0][i] as f64 - expected).abs() < 1e-24 + expected.abs() * 1e-6,
            "logaddexp({x}, {z}): {} vs {expected}",
            values[0][i]
        );
        let p = 1. / (1. + (z - x).exp());
        for (output, expected) in [p, 1. - p, p * (1. - p), -p * (1. - p)]
            .into_iter()
            .enumerate()
        {
            assert!((values[output + 1][i] as f64 - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn validates_axes_and_builds_logarithmic_stages() {
    let g = Tracer::default();
    assert!(g.input(&[]).unwrap().logcumsumexp(0).is_err());
    assert!(g.input(&[2]).unwrap().logcumsumexp(1).is_err());
    assert!(g.input(&[]).unwrap().logcumsumexp_tree(0).is_err());
    assert!(g.input(&[2]).unwrap().logcumsumexp_tree(1).is_err());
    let x = g.input(&[2]).unwrap();
    assert!(x.logaddexp(&g.input(&[]).unwrap()).is_err());
    assert!(
        x.logaddexp(&Tracer::default().input(&[2]).unwrap())
            .is_err()
    );
    let counts: Vec<_> = [8, 4096]
        .into_iter()
        .map(|length| {
            let graph = Tracer::default();
            let output = graph
                .input(&[length])
                .unwrap()
                .logcumsumexp_tree(0)
                .unwrap();
            graph
                .stablehlo(&output)
                .unwrap()
                .matches("stablehlo.concatenate")
                .count()
        })
        .collect();
    assert_eq!(counts, [5, 23]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_prefix_values_and_gradients_match_stable_f64_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, tree) in [[2_i64, 5], [2, 0], [2, 1], [3, 4], [2, 7]]
        .into_iter()
        .flat_map(|shape| [false, true].map(|tree| (shape, tree)))
    {
        for axis in 0..2 {
            let g = Tracer::default();
            let x = g.input(&shape).unwrap();
            let y = if tree {
                x.logcumsumexp_tree(axis)
            } else {
                x.logcumsumexp(axis)
            }
            .unwrap();
            let dx = y
                .sum(&[0, 1], false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            let ddx = dx
                .sum(&[0, 1], false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            let exe = g.compile_many(&client, &[y, dx, ddx]).unwrap();
            let count = shape.iter().product::<i64>() as usize;
            // Large magnitudes defeat a naive exp -> cumsum -> log implementation.
            let data: Vec<f32> = (0..count)
                .map(|i| {
                    if i % 2 == 0 {
                        1000. + i as f32
                    } else {
                        -1000. - i as f32
                    }
                })
                .collect();
            let stride = shape[axis + 1..].iter().product::<i64>() as usize;
            let length = shape[axis] as usize;
            let prefix: Vec<f64> = (0..count)
                .map(|i| {
                    let position = i / stride % length;
                    let values: Vec<f64> = (0..=position)
                        .map(|p| data[i - (position - p) * stride] as f64)
                        .collect();
                    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
                })
                .collect();
            let gradient: Vec<f64> = (0..count)
                .map(|i| {
                    let position = i / stride % length;
                    (position..length)
                        .map(|p| (data[i] as f64 - prefix[i + (p - position) * stride]).exp())
                        .sum()
                })
                .collect();
            let input = client.buffer(&shape, &data).unwrap();
            let output = exe.execute(&[&input]).unwrap();
            for (actual, expected) in output.iter().zip([prefix, gradient, vec![0.; count]]) {
                assert_eq!(actual.dimensions().unwrap(), shape);
                for (a, e) in actual.to_vec::<f32>().unwrap().iter().zip(expected) {
                    assert!(
                        (*a as f64 - e).abs() < 2e-4,
                        "shape={shape:?} axis={axis}: {a} vs {e}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_nonfinite_values_only_affect_prefix_successors() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[6]).unwrap();
    let exe = g.compile(&client, &x.logcumsumexp(0).unwrap()).unwrap();
    let input = client
        .buffer(
            &[6],
            &[
                f32::NEG_INFINITY,
                0.,
                f32::NEG_INFINITY,
                f32::INFINITY,
                2.,
                f32::NAN,
            ],
        )
        .unwrap();
    let output = exe.execute(&[&input]).unwrap()[0].to_vec::<f32>().unwrap();
    assert_eq!(
        output[..5],
        [f32::NEG_INFINITY, 0., 0., f32::INFINITY, f32::INFINITY]
    );
    assert!(output[5].is_nan());
}
