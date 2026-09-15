use rxla_core::{Client, Graph};

#[test]
fn scalar_padding_validates_graph_rank_and_widths() {
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let fill = g.input(&[]).unwrap();
    assert_eq!(x.pad_with_scalar(&[[1, 2]], &fill).unwrap().shape(), [5]);
    assert!(
        x.pad_with_scalar(&[[0, 0]], &g.input(&[1]).unwrap())
            .is_err()
    );
    assert!(
        x.pad_with_scalar(&[[0, 0]], &Graph::default().input(&[]).unwrap())
            .is_err()
    );
    for widths in [vec![], vec![[-1, 0]], vec![[i64::MAX, 0]]] {
        assert!(x.pad_with_scalar(&widths, &fill).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_runtime_fill_and_both_gradients_with_second_derivative() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let fill = g.input(&[]).unwrap();
    let padded = x.pad_with_scalar(&[[1, 2]], &fill).unwrap();
    let loss = padded.square().unwrap().sum(&[0], false).unwrap();
    let gradients = loss.grad(&[x.clone(), fill.clone()]).unwrap();
    let second = gradients[1]
        .grad(std::slice::from_ref(&fill))
        .unwrap()
        .remove(0);
    let weights = g.constant(&[5], &[1., 1e30, 1e30, 2., 3.]).unwrap();
    let weighted = padded.mul(&weights).unwrap().sum(&[0], false).unwrap();
    let border_gradient = weighted
        .grad(std::slice::from_ref(&fill))
        .unwrap()
        .remove(0);
    let exe = g
        .compile_many(
            &client,
            &[
                padded,
                loss,
                gradients[0].clone(),
                gradients[1].clone(),
                second,
                border_gradient,
            ],
        )
        .unwrap();
    for f in [-3., 0., 4.] {
        let output = exe.run_many(&[&[-2., 3.], &[f]]).unwrap();
        assert_eq!(output[0], [f, -2., 3., f, f]);
        assert_eq!(output[1], [13. + 3. * f * f]);
        assert_eq!(output[2], [-4., 6.]);
        assert_eq!(output[3], [6. * f]);
        assert_eq!(output[4], [6.]);
        assert_eq!(output[5], [6.]);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_empty_and_noop_padding_fill_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (size, widths, expected) in [(0, [2, 1], 3.), (2, [0, 0], 0.)] {
        let g = Graph::default();
        let x = g.input(&[size]).unwrap();
        let fill = g.input(&[]).unwrap();
        let y = x.pad_with_scalar(&[widths], &fill).unwrap();
        let gradient = y
            .sum(&[0], false)
            .unwrap()
            .grad(std::slice::from_ref(&fill))
            .unwrap()
            .remove(0);
        let exe = g.compile_many(&client, &[y, gradient]).unwrap();
        let values = vec![2.; size as usize];
        let out = exe.run_many(&[&values, &[7.]]).unwrap();
        assert_eq!(out[0], if size == 0 { vec![7.; 3] } else { values });
        assert_eq!(out[1], [expected]);
    }
}
