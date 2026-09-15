use rxla_core::{Client, Conv2dOptions, Graph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_convolution_mixed_derivatives_through_fourth_order() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[1, 1, 1, 1]).unwrap();
    let w = g.input(&[1, 1, 1, 1]).unwrap();
    let mut current = x
        .conv2d(&w, Default::default())
        .unwrap()
        .square()
        .unwrap()
        .mul_scalar(0.5)
        .unwrap();
    let mut roots = Vec::new();
    for input in [&x, &x, &w, &w, &w] {
        current = current
            .sum(&[0, 1, 2, 3], false)
            .unwrap()
            .grad(std::slice::from_ref(input))
            .unwrap()
            .remove(0);
        roots.push(current.clone());
    }
    let exe = g.compile_many(&client, &roots).unwrap();
    for (x, w) in [(2., 3.), (-1., -2.)] {
        assert_eq!(
            exe.run_many(&[&[x], &[w]]).unwrap(),
            [
                vec![x * w * w],
                vec![w * w],
                vec![2. * w],
                vec![2.],
                vec![0.]
            ]
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_grouped_convolution_joint_hessian_vector_matches_analytic_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[1, 2, 2, 2]).unwrap();
    let w = g.input(&[1, 1, 1, 4]).unwrap();
    let vx = g.input(x.shape()).unwrap();
    let vw = g.input(w.shape()).unwrap();
    let y = x
        .conv2d(
            &w,
            Conv2dOptions {
                groups: 2,
                ..Default::default()
            },
        )
        .unwrap();
    let loss = y
        .square()
        .unwrap()
        .sum(&[0, 1, 2, 3], false)
        .unwrap()
        .mul_scalar(0.5)
        .unwrap();
    let first = loss.grad(&[x.clone(), w.clone()]).unwrap();
    let directional = first[0]
        .mul(&vx)
        .unwrap()
        .sum(&[0, 1, 2, 3], false)
        .unwrap()
        .add(
            &first[1]
                .mul(&vw)
                .unwrap()
                .sum(&[0, 1, 2, 3], false)
                .unwrap(),
        )
        .unwrap();
    let second = directional.grad(&[x, w]).unwrap();
    let exe = g
        .compile_many(
            &client,
            &[
                first[0].clone(),
                first[1].clone(),
                second[0].clone(),
                second[1].clone(),
            ],
        )
        .unwrap();
    let xv = [1., -2., 3., 1., -1., 2., 0.5, -0.5];
    let wv = [0.5, -1., 2., 1.5];
    for (vxv, vwv) in [
        ([1.; 8], [1.; 4]),
        ([1., 0., -1., 2., 0.5, -0.5, 2., 1.], [0.5, -1., 1., 2.]),
    ] {
        let actual = exe.run_many(&[&xv, &wv, &vxv, &vwv]).unwrap();
        let mut dx = vec![0.; 8];
        let mut dw = vec![0.; 4];
        let mut hx = vec![0.; 8];
        let mut hw = vec![0.; 4];
        // Each 1x1 group maps one input channel to two output channels.
        // L = 1/2 * sum_g(sum_p x_pg^2) * (sum_o w_go^2).
        for group in 0..2 {
            let x2: f64 = (0..4).map(|p| f64::from(xv[p * 2 + group]).powi(2)).sum();
            let xvx: f64 = (0..4)
                .map(|p| f64::from(xv[p * 2 + group]) * f64::from(vxv[p * 2 + group]))
                .sum();
            let w2: f64 = (0..2).map(|o| f64::from(wv[group * 2 + o]).powi(2)).sum();
            let wvw: f64 = (0..2)
                .map(|o| f64::from(wv[group * 2 + o]) * f64::from(vwv[group * 2 + o]))
                .sum();
            for p in 0..4 {
                let i = p * 2 + group;
                dx[i] = f64::from(xv[i]) * w2;
                hx[i] = f64::from(vxv[i]) * w2 + 2. * f64::from(xv[i]) * wvw;
            }
            for o in 0..2 {
                let i = group * 2 + o;
                dw[i] = f64::from(wv[i]) * x2;
                hw[i] = f64::from(vwv[i]) * x2 + 2. * f64::from(wv[i]) * xvx;
            }
        }
        for (actual, expected) in actual.iter().zip([dx, dw, hx, hw]) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((f64::from(*actual) - expected).abs() < 1e-5);
            }
        }
    }
}
