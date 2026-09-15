use rxla_core::{Client, Graph, Output};

#[test]
fn argmax_shapes_and_axis_validation() {
    let g = Graph::default();
    let x = g.input(&[2, 3, 4]).unwrap();
    assert_eq!(x.argmax(1, false).unwrap().shape(), [2, 4]);
    assert_eq!(x.argmax(1, true).unwrap().shape(), [2, 1, 4]);
    assert!(x.argmax(3, false).is_err());
    assert!(g.input(&[]).unwrap().argmax(0, false).is_err());
    assert!(g.input(&[2, 0]).unwrap().argmax(1, false).is_err());
    assert!(
        g.input(&[0, i32::MAX as i64 + 1])
            .unwrap()
            .argmax(1, false)
            .is_err()
    );
    assert_eq!(
        g.input(&[0, 3]).unwrap().argmax(1, false).unwrap().shape(),
        [0]
    );
}

fn reference(values: &[f32], outer: usize, length: usize, inner: usize) -> Vec<i32> {
    let mut result = Vec::new();
    for o in 0..outer {
        for j in 0..inner {
            let mut best = 0;
            for i in 1..length {
                let a = values[(o * length + i) * inner + j];
                let b = values[(o * length + best) * inner + j];
                if !b.is_nan() && (a.is_nan() || a > b) {
                    best = i;
                }
            }
            result.push(best as i32);
        }
    }
    result
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_argmax_axes_ties_nan_and_infinity() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2, 3, 4]).unwrap();
    let outputs: Vec<Output> = (0..3)
        .flat_map(|a| [x.argmax(a, false).unwrap(), x.argmax(a, true).unwrap()])
        .collect();
    let exe = g.compile_outputs(&client, &outputs).unwrap();
    let mut values: Vec<f32> = (0..24).map(|i| (i % 7) as f32 - 3.).collect();
    values[1] = f32::NAN;
    values[17] = f32::NAN;
    values[3] = f32::INFINITY;
    values[7] = f32::INFINITY;
    values[10] = f32::NEG_INFINITY;
    let input = client.buffer(&[2, 3, 4], &values).unwrap();
    let actual = exe.execute(&[&input]).unwrap();
    for (axis, (outer, length, inner)) in [(1, 2, 12), (2, 3, 4), (6, 4, 1)].into_iter().enumerate()
    {
        let expected = reference(&values, outer, length, inner);
        for result in &actual[axis * 2..axis * 2 + 2] {
            assert_eq!(result.to_vec::<i32>().unwrap(), expected);
        }
    }
    let g = Graph::default();
    let x = g.input(&[6]).unwrap();
    let exe = g
        .compile_outputs(&client, &[x.argmax(0, false).unwrap()])
        .unwrap();
    for values in [
        [f32::NEG_INFINITY; 6],
        [0., -0., 0., -0., 0., -0.],
        [1., 4., 4., 2., 4., 3.],
        [f32::INFINITY; 6],
        [f32::NAN; 6],
        [f32::INFINITY, 2., f32::NAN, f32::NAN, 5., 1.],
    ] {
        let input = client.buffer(&[6], &values).unwrap();
        assert_eq!(
            exe.execute(&[&input]).unwrap()[0].to_vec::<i32>().unwrap(),
            reference(&values, 1, 6, 1)
        );
    }
    let g = Graph::default();
    let x = g.input(&[0, 3]).unwrap();
    let exe = g
        .compile_outputs(&client, &[x.argmax(1, false).unwrap()])
        .unwrap();
    let empty = client.buffer::<f32>(&[0, 3], &[]).unwrap();
    assert!(
        exe.execute(&[&empty]).unwrap()[0]
            .to_vec::<i32>()
            .unwrap()
            .is_empty()
    );
}
