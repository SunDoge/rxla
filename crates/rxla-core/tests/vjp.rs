use rxla_core::{Client, Graph};

#[test]
fn vjp_requires_matching_shape_owner_and_leaf_targets() {
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let y = x.exp().unwrap();
    assert!(
        y.vjp(std::slice::from_ref(&x), &g.input(&[]).unwrap())
            .is_err()
    );
    assert!(
        y.vjp(
            std::slice::from_ref(&x),
            &Graph::default().input(&[2]).unwrap()
        )
        .is_err()
    );
    assert!(y.vjp(std::slice::from_ref(&y), &x).is_err());
    assert!(y.vjp(&[Graph::default().input(&[2]).unwrap()], &x).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_vjp_runtime_cotangents_and_non_scalar_outputs() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2, 3]).unwrap();
    let w = g.input(&[3, 2]).unwrap();
    let seed = g.input(&[2, 2]).unwrap();
    let unused = g.input(&[]).unwrap();
    let y = x.matmul(&w).unwrap();
    let grads = y.vjp(&[x, w, unused], &seed).unwrap();
    let exe = g.compile_many(&client, &grads).unwrap();
    let xv = [1., 2., 3., 4., 5., 6.];
    let wv = [2., -1., 3., 0., 1., 4.];
    for sv in [[1., -2., 3., 4.], [0., 0., 0., 0.], [-1., 2., 1., -0.5]] {
        let actual = exe.run_many(&[&xv, &wv, &sv, &[7.]]).unwrap();
        let mut dx = vec![0.; 6];
        let mut dw = vec![0.; 6];
        for row in 0..2 {
            for k in 0..3 {
                for col in 0..2 {
                    dx[row * 3 + k] += sv[row * 2 + col] * wv[k * 2 + col];
                    dw[k * 2 + col] += xv[row * 3 + k] * sv[row * 2 + col];
                }
            }
        }
        assert_eq!(actual, [dx, dw, vec![0.]]);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_vjp_does_not_differentiate_seed_as_an_extra_loss_factor() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let y = x.mul(&x).unwrap();
    let gradient = y.vjp(std::slice::from_ref(&x), &x).unwrap().remove(0);
    let second = gradient
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    // VJP seed x is not an extra product term: first result is 2*x*x, not 3*x*x.
    // A subsequent grad sees both explicit x dependencies in that result.
    let exe = g.compile_many(&client, &[gradient, second]).unwrap();
    assert_eq!(
        exe.run_many(&[&[2., 3.]]).unwrap(),
        [vec![8., 18.], vec![8., 12.]]
    );
    let detached = y
        .vjp(std::slice::from_ref(&x), &x.detach().unwrap())
        .unwrap()
        .remove(0);
    let detached_second = detached
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let exe = g
        .compile_many(&client, &[detached, detached_second])
        .unwrap();
    // Detaching only the seed preserves the first VJP, but the next sweep
    // differentiates 2*x*stop_gradient(x), giving 2*x rather than 4*x.
    assert_eq!(
        exe.run_many(&[&[2., 3.]]).unwrap(),
        [vec![8., 18.], vec![4., 6.]]
    );
    for shape in [vec![], vec![0, 2]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let seed = g.input(&shape).unwrap();
        let grads = x.neg().unwrap().vjp(&[x], &seed).unwrap();
        let exe = g.compile_many(&client, &grads).unwrap();
        let input = vec![1.; usize::from(shape.is_empty())];
        let seed = vec![3.; input.len()];
        assert_eq!(
            exe.run_many(&[&input, &seed]).unwrap(),
            [vec![-3.; input.len()]]
        );
    }
}
