use rxla_core::{Client, Tracer};

#[test]
fn validates_count_axis_and_overflow() {
    let graph = Tracer::default();
    let x = graph.input(&[2, 3]).unwrap();
    assert!(x.repeat_interleave(-1, 0).is_err());
    assert!(x.repeat_interleave(i64::MAX, 0).is_err());
    assert!(
        graph
            .input(&[1, 3])
            .unwrap()
            .repeat_interleave(i64::MAX, 0)
            .is_err()
    );
    for repeats in [0, 1, 2] {
        assert!(x.repeat_interleave(repeats, 2).is_err());
        assert!(
            graph
                .input(&[])
                .unwrap()
                .repeat_interleave(repeats, 0)
                .is_err()
        );
    }
    assert_eq!(x.repeat_interleave(0, 1).unwrap().shape(), [2, 0]);
    assert_eq!(x.repeat_interleave(2, 1).unwrap().shape(), [2, 6]);
    assert_eq!(
        graph
            .input(&[0, 3])
            .unwrap()
            .repeat_interleave(4, 0)
            .unwrap()
            .shape(),
        [0, 3]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_values_and_first_second_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![3], vec![2, 3], vec![2, 1, 3], vec![0, 3]] {
        for axis in 0..shape.len() {
            for repeats in [0, 1, 2, 3] {
                let graph = Tracer::default();
                let x = graph.input(&shape).unwrap();
                let y = x.repeat_interleave(repeats, axis).unwrap();
                let n = shape.iter().product::<i64>() as usize;
                let m = y.shape().iter().product::<i64>() as usize;
                let values: Vec<_> = (0..n).map(|i| i as f32 - 2.).collect();
                let weights: Vec<_> = (0..m).map(|i| (i % 7 + 1) as f32).collect();
                let w = graph.constant(y.shape(), &weights).unwrap();
                let axes: Vec<_> = (0..shape.len()).collect();
                let loss = y
                    .mul(&y)
                    .unwrap()
                    .mul(&w)
                    .unwrap()
                    .sum(&axes, false)
                    .unwrap();
                let grad = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
                let second = grad
                    .sum(&axes, false)
                    .unwrap()
                    .grad(std::slice::from_ref(&x))
                    .unwrap()
                    .remove(0);
                let mut expected = Vec::new();
                let mut first = vec![0.; n];
                let mut second_expected = vec![0.; n];
                let stride = shape[axis + 1..].iter().product::<i64>() as usize;
                let output_width = shape[axis] as usize * repeats as usize;
                for (i, &weight) in weights.iter().enumerate() {
                    let outer = i / (output_width * stride);
                    let coordinate = (i / stride) % output_width;
                    let source = outer * shape[axis] as usize * stride
                        + coordinate / repeats as usize * stride
                        + i % stride;
                    expected.push(values[source]);
                    first[source] += 2. * values[source] * weight;
                    second_expected[source] += 2. * weight;
                }
                let result = graph
                    .compile_many(&client, &[y, grad, second])
                    .unwrap()
                    .run_many(&[&values])
                    .unwrap();
                assert_eq!(
                    result[0], expected,
                    "shape={shape:?} axis={axis} repeats={repeats}"
                );
                assert_eq!(result[1], first);
                assert_eq!(result[2], second_expected);
            }
        }
    }
}
