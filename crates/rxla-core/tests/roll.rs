use rxla_core::{Client, Graph};

#[test]
fn roll_validates_axis_even_for_zero_shift() {
    let graph = Graph::default();
    assert!(graph.input(&[]).unwrap().roll(0, 0).is_err());
    assert!(graph.input(&[0]).unwrap().roll(0, 1).is_err());
    assert_eq!(
        graph
            .input(&[0])
            .unwrap()
            .roll(i64::MIN, 0)
            .unwrap()
            .shape(),
        [0]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_roll_values_weighted_gradients_and_inverse() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![5], vec![2, 3], vec![2, 1, 3], vec![0, 3]] {
        for axis in 0..shape.len() {
            for shift in [0, 1, -1, 6, i64::MIN, i64::MAX] {
                let graph = Graph::default();
                let x = graph.input(&shape).unwrap();
                let y = x.roll(shift, axis).unwrap();
                let len = shape.iter().product::<i64>() as usize;
                let values: Vec<_> = (0..len).map(|i| i as f32 - 3.).collect();
                let weights: Vec<_> = (0..len).map(|i| i as f32 + 1.).collect();
                let w = graph.constant(&shape, &weights).unwrap();
                let axes: Vec<_> = (0..shape.len()).collect();
                let gradient = y
                    .mul(&w)
                    .unwrap()
                    .sum(&axes, false)
                    .unwrap()
                    .grad(std::slice::from_ref(&x))
                    .unwrap()
                    .remove(0);
                let size = shape[axis];
                let normalized = if size == 0 { 0 } else { shift.rem_euclid(size) };
                let restored = y.roll(-normalized, axis).unwrap();
                let exe = graph
                    .compile_many(&client, &[y, gradient, restored])
                    .unwrap();
                let mut expected = vec![0.; len];
                let mut expected_grad = vec![0.; len];
                let stride = shape[axis + 1..].iter().product::<i64>() as usize;
                for i in 0..len {
                    let coordinate = (i / stride) % size as usize;
                    // Widen the arithmetic so the reference also handles MIN/MAX.
                    let source_coordinate =
                        (coordinate as i128 - shift as i128).rem_euclid(size as i128) as usize;
                    let source = i - coordinate * stride + source_coordinate * stride;
                    expected[i] = values[source];
                    expected_grad[source] = weights[i];
                }
                let output = exe.run_many(&[&values]).unwrap();
                assert_eq!(
                    output[0], expected,
                    "shape={shape:?} axis={axis} shift={shift}"
                );
                assert_eq!(output[1], expected_grad);
                assert_eq!(output[2], values);
            }
        }
    }
}
