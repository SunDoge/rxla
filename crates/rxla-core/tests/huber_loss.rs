use rxla_core::{Client, Graph};

#[test]
fn huber_validates_delta_shape_and_graph() {
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    for delta in [0., -1., f32::INFINITY, f32::NAN] {
        assert!(x.huber_loss(&x, delta).is_err());
    }
    assert!(x.huber_loss(&g.input(&[]).unwrap(), 1.).is_err());
    assert!(
        x.huber_loss(&Graph::default().input(&[2]).unwrap(), 1.)
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_huber_values_gradients_curvature_and_extreme_residuals() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for delta in [0.5_f32, 2.] {
        let values = [
            -f32::MAX,
            -1e30,
            -4.,
            -delta,
            -0.25,
            -0.,
            0.,
            0.25,
            delta,
            4.,
            1e30,
            f32::MAX,
        ];
        let g = Graph::default();
        let x = g.input(&[values.len() as i64]).unwrap();
        let target = g.input(x.shape()).unwrap();
        let loss = x.huber_loss(&target, delta).unwrap();
        let grad = loss
            .sum(&[0], false)
            .unwrap()
            .grad(&[x.clone(), target.clone()])
            .unwrap();
        let curvature = grad[0]
            .sum(&[0], false)
            .unwrap()
            .grad(&[x.clone(), target])
            .unwrap();
        let exe = g
            .compile_many(
                &client,
                &[
                    loss,
                    grad[0].clone(),
                    grad[1].clone(),
                    curvature[0].clone(),
                    curvature[1].clone(),
                ],
            )
            .unwrap();
        for sign in [1., -1.] {
            let input: Vec<_> = values.iter().map(|x| x * sign).collect();
            let actual = exe.run_many(&[&input, &[0.; 12]]).unwrap();
            for (i, &r) in input.iter().enumerate() {
                let r = r as f64;
                let d = delta as f64;
                let quadratic = r.abs() <= d;
                let expected = if quadratic {
                    0.5 * r * r
                } else {
                    d * (r.abs() - 0.5 * d)
                };
                if expected > f32::MAX as f64 {
                    assert!(actual[0][i].is_infinite());
                } else {
                    assert!((actual[0][i] as f64 - expected).abs() <= 2e-6 * expected.abs() + 1e-7);
                }
                let derivative = r.clamp(-d, d) as f32;
                assert_eq!(actual[1][i], derivative, "r={r} delta={delta}");
                assert_eq!(actual[2][i], -derivative);
                let curvature = if quadratic { 1. } else { 0. };
                assert_eq!(actual[3][i], curvature, "r={r} delta={delta}");
                assert_eq!(actual[4][i], -curvature);
            }
        }
    }
    for shape in [vec![], vec![0, 2]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let target = g
            .constant(&[], &[0.])
            .unwrap()
            .broadcast_to(&shape)
            .unwrap();
        let loss = x.huber_loss(&target, 1.).unwrap();
        let axes: Vec<_> = (0..shape.len()).collect();
        let grad = loss
            .sum(&axes, false)
            .unwrap()
            .grad(std::slice::from_ref(&x))
            .unwrap()
            .remove(0);
        let exe = g.compile_many(&client, &[loss, grad]).unwrap();
        let values: Vec<f32> = if shape.is_empty() { vec![2.] } else { vec![] };
        let output = exe.run_many(&[&values]).unwrap();
        assert_eq!(
            output,
            if shape.is_empty() {
                vec![vec![1.5], vec![1.]]
            } else {
                vec![vec![], vec![]]
            }
        );
    }
}
