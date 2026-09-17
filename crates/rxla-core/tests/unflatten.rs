use rxla_core::{Client, Tracer};

#[test]
fn unflatten_validates_selected_axis_even_when_total_size_is_zero() {
    let g = Tracer::default();
    let x = g.input(&[0, 6]).unwrap();
    let ids = g.input_i32(&[0, 6]).unwrap();
    assert_eq!(x.unflatten(1, &[2, 3]).unwrap().shape(), [0, 2, 3]);
    assert_eq!(ids.unflatten(0, &[2, 0]).unwrap().shape(), [2, 0, 6]);
    for sizes in [
        vec![],
        vec![-1, 6],
        vec![5],
        vec![0, 6],
        vec![i64::MAX, i64::MAX],
    ] {
        assert!(x.unflatten(1, &sizes).is_err());
        assert!(ids.unflatten(1, &sizes).is_err());
    }
    for axis in [2, usize::MAX] {
        assert!(x.unflatten(axis, &[6]).is_err());
        assert!(ids.unflatten(axis, &[6]).is_err());
    }
    assert!(g.input(&[]).unwrap().unflatten(0, &[1]).is_err());
    assert!(g.input_i32_scalar().unwrap().unflatten(0, &[1]).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_unflatten_preserves_typed_values_and_gradient() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, axis, sizes) in [
        (vec![2, 6, 4], 1, vec![2, 3]),
        (vec![1], 0, vec![1, 1]),
        (vec![2, 0, 4], 1, vec![3, 0]),
    ] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let index = g.input_i32(&shape).unwrap();
        let y = x.unflatten(axis, &sizes).unwrap();
        let ids = index.unflatten(axis, &sizes).unwrap();
        let restored = y.flatten(axis, axis + sizes.len() - 1).unwrap();
        assert_eq!(restored.shape(), shape);
        let grad = y
            .mul(&y)
            .unwrap()
            .sum(&(0..y.ndim()).collect::<Vec<_>>(), false)
            .unwrap()
            .grad(&[x])
            .unwrap()
            .remove(0);
        let values: Vec<_> = (0..index.static_numel().unwrap())
            .map(|i| i as f32 - 3.)
            .collect();
        let integers: Vec<_> = (0..index.static_numel().unwrap())
            .map(|i| [i32::MIN, i32::MAX, -1][i % 3])
            .collect();
        let exe = g
            .compile_many(&client, &[y.clone(), ids, grad, restored])
            .unwrap();
        let x = client.buffer(&shape, &values).unwrap();
        let ids = client.buffer(&shape, &integers).unwrap();
        let output = exe.execute(&[&x, &ids]).unwrap();
        assert_eq!(output[0].dimensions().unwrap(), y.shape());
        assert_eq!(output[0].to_vec::<f32>().unwrap(), values);
        assert_eq!(output[1].to_vec::<i32>().unwrap(), integers);
        assert_eq!(
            output[2].to_vec::<f32>().unwrap(),
            values.iter().map(|v| v * 2.).collect::<Vec<_>>()
        );
        assert_eq!(output[3].to_vec::<f32>().unwrap(), values);
    }
}
