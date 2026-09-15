use rxla_core::{CacheLimits, Client, Compiler, Graph, StateGraph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_rectangular_surrogate_routes_dense_vjp_and_cross_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let forward = g.input(&[2, 2]).unwrap();
    let x = g.input(&[2, 3]).unwrap();
    let w = g.input(&[3, 2]).unwrap();
    let seed = g.input(&[2, 2]).unwrap();
    let y = forward.with_gradient_of(&x.matmul(&w).unwrap()).unwrap();
    assert_eq!(
        g.prepare_outputs_pruned(std::slice::from_ref(&y))
            .unwrap()
            .1,
        [0]
    );
    let gradients = y.vjp(&[forward.clone(), x, w.clone()], &seed).unwrap();
    assert_eq!(gradients[1].shape(), [2, 3]);
    assert_eq!(gradients[2].shape(), [3, 2]);
    // d/dW sum(dL/dX) = sum_batch(seed), broadcast to each input feature.
    let cross = gradients[1]
        .sum(&[0, 1], false)
        .unwrap()
        .grad(&[w])
        .unwrap()
        .remove(0);
    let mut outputs = vec![y];
    outputs.extend(gradients);
    outputs.push(cross);
    let exe = g.compile_many(&client, &outputs).unwrap();
    let xv = [1., -2., 3., 4., 0.5, -1.];
    let wv = [2., -1., 0.5, 3., -2., 4.];
    for sv in [[1., -2., 3., 0.5], [-1., 4., 0., -2.]] {
        let fv = [17., -3., 0., 42.];
        let actual = exe.run_many(&[&fv, &xv, &wv, &sv]).unwrap();
        assert_eq!(actual[0], fv);
        assert_eq!(actual[1], [0.; 4]);
        let mut dx = vec![0.; 6];
        let mut dw = vec![0.; 6];
        let mut expected_cross = vec![0.; 6];
        for b in 0..2 {
            for k in 0..3 {
                for j in 0..2 {
                    dx[b * 3 + k] += sv[b * 2 + j] * wv[k * 2 + j];
                    dw[k * 2 + j] += xv[b * 3 + k] * sv[b * 2 + j];
                    expected_cross[k * 2 + j] += sv[b * 2 + j];
                }
            }
        }
        assert_eq!(actual[2], dx);
        assert_eq!(actual[3], dw);
        assert_eq!(actual[4], expected_cross);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_partial_rules_share_forward_cache_but_separate_backward_cache() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let graph = Graph::default();
    let x = graph.input(&[4]).unwrap();
    let one = graph.constant(&[4], &[1.; 4]).unwrap();
    let two = graph.constant(&[4], &[2.; 4]).unwrap();
    let nonfinite = graph.constant(&[4], &[-1.; 4]).unwrap().log().unwrap();
    let first = x.with_elementwise_derivative(&x, &one).unwrap();
    let second = x.with_elementwise_derivative(&x, &two).unwrap();
    let poisoned = x
        .with_elementwise_derivatives(&[(&x, &nonfinite), (&x, &two)])
        .unwrap();
    let forward = compiler.compile(&graph, &first).unwrap();
    for expression in [&second, &poisoned] {
        let reused = compiler.compile(&graph, expression).unwrap();
        assert!(std::rc::Rc::ptr_eq(&forward, &reused));
    }
    assert_eq!(compiler.stats().misses, 1);
    assert_eq!(compiler.stats().hits, 2);
    let values = [
        -0.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc01234),
    ];
    let input = client.buffer(&[4], &values).unwrap();
    let actual = forward.execute(&[&input]).unwrap()[0]
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        values.map(f32::to_bits)
    );
    let gradients = [first, second].map(|y| {
        y.sum(&[0], false)
            .unwrap()
            .grad(std::slice::from_ref(&x))
            .unwrap()
            .remove(0)
    });
    let backward_one = compiler.compile(&graph, &gradients[0]).unwrap();
    let backward_two = compiler.compile(&graph, &gradients[1]).unwrap();
    assert!(!std::rc::Rc::ptr_eq(&backward_one, &backward_two));
    assert_eq!(
        backward_one.execute(&[&input]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [1.; 4]
    );
    assert_eq!(
        backward_two.execute(&[&input]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [2.; 4]
    );
    assert_eq!(compiler.stats().misses, 3);
    assert_eq!(compiler.stats().compile_failures, 0);
}

#[test]
fn multi_input_partials_validate_all_pairs_and_prune_forward_edges() {
    let graph = Graph::default();
    let f = graph.input(&[3]).unwrap();
    let x = graph.input(&[3]).unwrap();
    let z = graph.input(&[3]).unwrap();
    assert!(f.with_elementwise_derivatives(&[]).is_err());
    for bad in [
        graph.input(&[]).unwrap(),
        Graph::default().input(&[3]).unwrap(),
    ] {
        assert!(
            f.with_elementwise_derivatives(&[(&x, &z), (&bad, &x)])
                .is_err()
        );
        assert!(
            f.with_elementwise_derivatives(&[(&x, &z), (&z, &bad)])
                .is_err()
        );
    }
    let y = f
        .with_elementwise_derivatives(&[(&x, &z), (&z, &x)])
        .unwrap();
    assert_eq!(graph.prepare_outputs_pruned(&[y]).unwrap().1, [0]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_multi_input_partials_cross_derivatives_and_alias_accumulation() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Graph::default();
    let f = graph.input(&[3]).unwrap();
    let x = graph.input(&[3]).unwrap();
    let z = graph.input(&[3]).unwrap();
    let y = f
        .with_elementwise_derivatives(&[(&x, &z), (&z, &x)])
        .unwrap();
    let gradients = y
        .mul(&y)
        .unwrap()
        .mul_scalar(0.5)
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(&[f.clone(), x.clone(), z.clone()])
        .unwrap();
    let hessian = gradients[1]
        .sum(&[0], false)
        .unwrap()
        .grad(&[x.clone(), z.clone()])
        .unwrap();
    let alias = f
        .with_elementwise_derivatives(&[(&x, &z), (&x, &z)])
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    // Distinct input edges sharing an ancestor also add through ordinary AD.
    let doubled = x.mul_scalar(2.).unwrap();
    let shared = f
        .with_elementwise_derivatives(&[(&x, &z), (&doubled, &z)])
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let executable = graph
        .compile_many(
            &client,
            &[
                y,
                gradients[0].clone(),
                gradients[1].clone(),
                gradients[2].clone(),
                hessian[0].clone(),
                hessian[1].clone(),
                alias,
                shared,
            ],
        )
        .unwrap();
    let inputs = [
        client.buffer(&[3], &[2., -3., 0.]).unwrap(),
        client.buffer(&[3], &[1., 2., -1.]).unwrap(),
        client.buffer(&[3], &[4., -2., 3.]).unwrap(),
    ];
    let outputs = executable
        .execute(&[&inputs[0], &inputs[1], &inputs[2]])
        .unwrap();
    for (actual, expected) in outputs.iter().zip([
        [2., -3., 0.],
        [0.; 3],
        [8., 6., 0.],
        [2., -6., 0.],
        [16., 4., 9.],
        [6., -7., -3.],
        [8., -4., 6.],
        [12., -6., 9.],
    ]) {
        assert_eq!(actual.to_vec::<f32>().unwrap(), expected);
    }
}

#[test]
fn explicit_elementwise_derivative_validates_and_prunes_forward_only_edges() {
    let graph = Graph::default();
    let forward = graph.input(&[3]).unwrap();
    let x = graph.input(&[3]).unwrap();
    let derivative = x.mul(&x).unwrap().mul_scalar(3.).unwrap();
    for bad in [
        graph.input(&[]).unwrap(),
        Graph::default().input(&[3]).unwrap(),
    ] {
        assert!(
            forward
                .with_elementwise_derivative(&bad, &derivative)
                .is_err()
        );
        assert!(forward.with_elementwise_derivative(&x, &bad).is_err());
    }
    let output = forward
        .with_elementwise_derivative(&x, &derivative)
        .unwrap();
    assert_eq!(
        graph
            .prepare_outputs_pruned(std::slice::from_ref(&output))
            .unwrap()
            .1,
        [0]
    );
    let gradient = output
        .sum(&[0], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    assert_eq!(graph.prepare_outputs_pruned(&[gradient]).unwrap().1, [1]);
    assert_eq!(graph.prepare_outputs_pruned(&[output]).unwrap().1, [0]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_explicit_elementwise_derivative_vjp_and_higher_order() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Graph::default();
    let forward = graph.input(&[3]).unwrap();
    let x = graph.input(&[3]).unwrap();
    let seed = graph.input(&[3]).unwrap();
    let derivative = x.mul(&x).unwrap().mul_scalar(3.).unwrap();
    let output = forward
        .with_elementwise_derivative(&x, &derivative)
        .unwrap();
    let gradients = output
        .mul(&output)
        .unwrap()
        .mul_scalar(0.5)
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(&[forward, x.clone()])
        .unwrap();
    let higher = gradients[1]
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let vjp = output.vjp(&[x], &seed).unwrap().remove(0);
    let executable = graph
        .compile_many(
            &client,
            &[
                output,
                gradients[0].clone(),
                gradients[1].clone(),
                higher,
                vjp,
            ],
        )
        .unwrap();
    let inputs = [
        client.buffer(&[3], &[2., -3., 0.]).unwrap(),
        client.buffer(&[3], &[1., 2., -1.]).unwrap(),
        client.buffer(&[3], &[3., -2., 4.]).unwrap(),
    ];
    let result = executable
        .execute(&[&inputs[0], &inputs[1], &inputs[2]])
        .unwrap();
    // d(y * 3x²)/dx = 9x⁴ + 6xy, including the wrapped y's chosen derivative.
    for (actual, expected) in result.iter().zip([
        [2., -3., 0.],
        [0.; 3],
        [6., -36., 0.],
        [21., 108., 9.],
        [9., -24., 12.],
    ]) {
        assert_eq!(actual.to_vec::<f32>().unwrap(), expected);
    }
}

#[test]
fn validates_contract_and_prunes_only_forward_dependencies() {
    let g = Graph::default();
    let x = g.input(&[3]).unwrap();
    let s = g.input(&[3]).unwrap();
    for bad in [g.input(&[]).unwrap(), Graph::default().input(&[3]).unwrap()] {
        assert!(x.with_gradient_of(&bad).is_err());
    }
    let y = x.with_gradient_of(&s.exp().unwrap()).unwrap();
    let (_, mapping) = g.prepare_outputs_pruned(std::slice::from_ref(&y)).unwrap();
    assert_eq!(mapping, [0]);
    let ds = y.sum(&[0], false).unwrap().grad(&[s]).unwrap().remove(0);
    assert_eq!(g.prepare_outputs_pruned(&[ds]).unwrap().1, [1]);
    // Inspecting/compiling snapshots must not erase the original AD edge.
    assert_eq!(g.prepare_outputs_pruned(&[y]).unwrap().1, [0]);
    let unsupported = x.gt_mask(&x).unwrap();
    assert!(
        x.with_gradient_of(&unsupported)
            .unwrap()
            .sum(&[0], false)
            .unwrap()
            .grad(&[x])
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_surrogate_vjp_and_higher_derivatives_follow_backward_graph() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[3]).unwrap();
    let s = g.input(&[3]).unwrap();
    let seed = g.input(&[3]).unwrap();
    let y = x.with_gradient_of(&s.mul(&s).unwrap()).unwrap();
    let gradients = y
        .mul(&y)
        .unwrap()
        .mul_scalar(0.5)
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(&[x.clone(), s.clone()])
        .unwrap();
    let higher = gradients[1]
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&s))
        .unwrap()
        .remove(0);
    let vjp = y.vjp(&[x.clone(), s], &seed).unwrap();
    let unchanged = x.sum(&[0], false).unwrap().grad(&[x]).unwrap().remove(0);
    let executable = g
        .compile_many(
            &client,
            &[
                y,
                gradients[0].clone(),
                gradients[1].clone(),
                higher,
                vjp[0].clone(),
                vjp[1].clone(),
                unchanged,
            ],
        )
        .unwrap();
    let input = [
        client.buffer(&[3], &[2., -3., 0.]).unwrap(),
        client.buffer(&[3], &[1., 2., -1.]).unwrap(),
        client.buffer(&[3], &[3., -2., 4.]).unwrap(),
    ];
    let output = executable
        .execute(&[&input[0], &input[1], &input[2]])
        .unwrap();
    // d/ds (2*s*y) = 2*y + 4*s², because y itself carries the substituted rule.
    let expected = [
        [2., -3., 0.],
        [0.; 3],
        [4., -12., 0.],
        [8., 10., 4.],
        [0.; 3],
        [6., -8., -8.],
        [1.; 3],
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert_eq!(actual.to_vec::<f32>().unwrap(), expected);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_forward_preserves_nonfinite_bits_and_straight_through_mask() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let x = g.input(&[4]).unwrap();
    let s = g.input(&[4]).unwrap();
    let y = x.with_gradient_of(&s.log().unwrap()).unwrap();
    let plan = g.compile_pruned(&mut compiler, &[y]).unwrap();
    assert_eq!(plan.input_indices(), [0]);
    let mut session = plan.session(vec![]).unwrap();
    let values = [
        -0.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc01234),
    ];
    let input = client.buffer(&[4], &values).unwrap();
    assert_eq!(
        session.run(&[&input]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        values.map(f32::to_bits)
    );
    let zero = g.constant(&[4], &[0.; 4]).unwrap();
    let mask = x.gt_mask(&zero).unwrap().with_gradient_of(&x).unwrap();
    let derivative = mask
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let plan = g
        .compile_pruned(&mut compiler, &[mask, derivative])
        .unwrap();
    assert_eq!(plan.input_indices(), [0]);
    let mut session = plan.session(vec![]).unwrap();
    let input = client.buffer(&[4], &[-2., 0., 1., 3.]).unwrap();
    let output = session.run(&[&input]).unwrap();
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [0., 0., 1., 1.]);
    assert_eq!(output[1].to_vec::<f32>().unwrap(), [1.; 4]);
}
