use rxla_core::{Client, Graph};

#[test]
fn flatten_checks_ranges_and_dimension_overflow_without_plugin() {
    let g = Graph::default();
    let x = g.input(&[2, 3, 4]).unwrap();
    let ids = g.input_i32(&[2, 3, 4]).unwrap();
    assert_eq!(x.flatten(1, 2).unwrap().shape(), [2, 12]);
    assert_eq!(ids.flatten(0, 1).unwrap().shape(), [6, 4]);
    for (start, end) in [(2, 1), (0, 3), (usize::MAX, usize::MAX)] {
        assert!(x.flatten(start, end).is_err());
        assert!(ids.flatten(start, end).is_err());
    }
    let scalar = g.input(&[]).unwrap();
    assert_eq!(scalar.flatten(0, 0).unwrap().shape(), [1]);
    assert!(scalar.flatten(0, 1).is_err());
    let huge_empty = g.input(&[0, i64::MAX, 2]).unwrap();
    assert!(huge_empty.flatten(1, 2).is_err());
    assert_eq!(huge_empty.flatten(0, 2).unwrap().shape(), [0]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_flatten_preserves_values_integer_bits_and_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![], vec![2, 3, 4], vec![2, 0, 3]] {
        let count = shape.iter().product::<i64>() as usize;
        let values: Vec<_> = (0..count).map(|i| i as f32 - 5.).collect();
        let integers: Vec<_> = (0..count)
            .map(|i| [i32::MIN, i32::MAX, 16_777_217, -1][i % 4])
            .collect();
        for start in 0..shape.len().max(1) {
            for end in start..shape.len().max(1) {
                let g = Graph::default();
                let x = g.input(&shape).unwrap();
                let ids = g.input_i32(&shape).unwrap();
                let y = x.flatten(start, end).unwrap();
                let integer = ids.flatten(start, end).unwrap();
                let axes: Vec<_> = (0..y.ndim()).collect();
                let grad = y
                    .mul(&y)
                    .unwrap()
                    .sum(&axes, false)
                    .unwrap()
                    .grad(&[x])
                    .unwrap()
                    .remove(0);
                let exe = g
                    .compile_outputs(&client, &[y.clone(), integer, grad])
                    .unwrap();
                let x = client.buffer(&shape, &values).unwrap();
                let ids = client.buffer(&shape, &integers).unwrap();
                let output = exe.execute(&[&x, &ids]).unwrap();
                assert_eq!(output[0].dimensions().unwrap(), y.shape());
                assert_eq!(output[0].to_vec::<f32>().unwrap(), values);
                assert_eq!(output[1].to_vec::<i32>().unwrap(), integers);
                assert_eq!(output[2].dimensions().unwrap(), shape);
                assert_eq!(
                    output[2].to_vec::<f32>().unwrap(),
                    values.iter().map(|v| 2. * v).collect::<Vec<_>>()
                );
            }
        }
    }
}
