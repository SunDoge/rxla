use rxla_core::{Client, Tracer};

#[test]
fn sparse_targets_require_exact_nonclass_shape_and_owner() {
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    let labels = g.input_i32(&[2]).unwrap();
    assert_eq!(
        x.cross_entropy_with_indices(&labels, 1).unwrap().shape(),
        [2]
    );
    assert!(x.cross_entropy_with_indices(&labels, 0).is_err());
    assert!(x.cross_entropy_with_indices(&labels, 2).is_err());
    assert!(
        x.cross_entropy_with_indices(&g.input_i32(&[2, 1]).unwrap(), 1)
            .is_err()
    );
    assert!(
        x.cross_entropy_with_indices(&Tracer::default().input_i32(&[2]).unwrap(), 1)
            .is_err()
    );
    assert!(
        g.input(&[2, 0])
            .unwrap()
            .cross_entropy_with_indices(&labels, 1)
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_sparse_loss_gradient_and_invalid_labels() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axis in [0, 1] {
        let g = Tracer::default();
        let x = g.input(&[2, 3]).unwrap();
        let groups = if axis == 0 { 3 } else { 2 };
        let classes = if axis == 0 { 2 } else { 3 };
        let ids = g.input_i32(&[groups as i64]).unwrap();
        let loss = x.cross_entropy_with_indices(&ids, axis).unwrap();
        let grad = loss.sum(&[0], false).unwrap().grad(&[x]).unwrap().remove(0);
        let exe = g.compile_many(&client, &[loss, grad]).unwrap();
        let values = [1000.0f32, 999., -1000., 3., 3., 3.];
        let xb = client.buffer(&[2, 3], &values).unwrap();
        for offset in [0, 1] {
            let labels: Vec<_> = (0..groups)
                .map(|i| ((i + offset) % classes) as i32)
                .collect();
            let ib = client.buffer(&[groups as i64], &labels).unwrap();
            let actual = exe.execute(&[&xb, &ib]).unwrap();
            let loss = actual[0].to_vec::<f32>().unwrap();
            let grad = actual[1].to_vec::<f32>().unwrap();
            for group in 0..groups {
                let index = |i| {
                    if axis == 0 {
                        i * 3 + group
                    } else {
                        group * 3 + i
                    }
                };
                let max = (0..classes)
                    .map(|i| values[index(i)] as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = (0..classes)
                    .map(|i| (values[index(i)] as f64 - max).exp())
                    .sum();
                let expected = max + sum.ln() - values[index(labels[group] as usize)] as f64;
                assert!((loss[group] as f64 - expected).abs() < 2e-4);
                for i in 0..classes {
                    let p = (values[index(i)] as f64 - max).exp() / sum;
                    let expected = p - if i as i32 == labels[group] { 1. } else { 0. };
                    assert!((grad[index(i)] as f64 - expected).abs() < 1e-6);
                }
            }
        }
        for bad in [-1, classes as i32, i32::MIN, i32::MAX] {
            let ib = client.buffer(&[groups as i64], &vec![bad; groups]).unwrap();
            assert!(
                exe.execute(&[&xb, &ib]).unwrap()[0]
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_nan())
            );
        }
    }
    for shape in [vec![3], vec![0, 3]] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let ids = g.input_i32(&shape[..shape.len() - 1]).unwrap();
        let loss = x.cross_entropy_with_indices(&ids, shape.len() - 1).unwrap();
        let exe = g.compile(&client, &loss).unwrap();
        let scalar = shape.len() == 1;
        let xb = client
            .buffer(&shape, &vec![0.; if scalar { 3 } else { 0 }])
            .unwrap();
        let ib = client
            .buffer(ids.shape(), &vec![1; usize::from(scalar)])
            .unwrap();
        let result = exe.execute(&[&xb, &ib]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap();
        if scalar {
            assert!((result[0] - 3.0f32.ln()).abs() < 1e-6);
        } else {
            assert!(result.is_empty());
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_select_routes_data_gradients_but_not_mask_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let mask = g.input(&[4]).unwrap();
    let a = g.input(&[4]).unwrap();
    let b = g.input(&[4]).unwrap();
    let loss = mask.select(&a, &b).unwrap().sum(&[0], false).unwrap();
    let grads = loss.grad(&[mask, a, b]).unwrap();
    let exe = g.compile_many(&client, &grads).unwrap();
    assert_eq!(
        exe.run_many(&[&[0., -0., -1., f32::NAN], &[1.; 4], &[2.; 4]])
            .unwrap(),
        [vec![0.; 4], vec![0., 0., 1., 1.], vec![1., 1., 0., 0.]]
    );
}
