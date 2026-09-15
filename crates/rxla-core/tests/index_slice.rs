use rxla_core::{Client, Graph};

#[test]
fn integer_and_float_slice_validation_agree() {
    let g = Graph::default();
    let i = g.input_i32(&[2, 3]).unwrap();
    let f = g.input(&[2, 3]).unwrap();
    for (starts, limits, strides) in [
        (vec![0], vec![2, 3], vec![1, 1]),
        (vec![0, 0], vec![2], vec![1, 1]),
        (vec![0, 0], vec![2, 3], vec![1]),
        (vec![-1, 0], vec![2, 3], vec![1, 1]),
        (vec![0, 2], vec![2, 1], vec![1, 1]),
        (vec![0, 0], vec![2, 4], vec![1, 1]),
        (vec![0, 0], vec![2, 3], vec![0, 1]),
        (vec![0, 0], vec![2, 3], vec![1, -1]),
    ] {
        assert!(i.slice(&starts, &limits, &strides).is_err());
        assert!(f.slice(&starts, &limits, &strides).is_err());
    }
    for (axis, start, length) in [
        (2, 0, 0),
        (0, -1, 1),
        (1, 0, -1),
        (1, 2, 2),
        (1, i64::MAX, 1),
    ] {
        assert!(i.narrow(axis, start, length).is_err());
        assert!(f.narrow(axis, start, length).is_err());
    }
    assert_eq!(i.narrow(1, 3, 0).unwrap().shape(), [2, 0]);
    let huge = g.input_i32(&[i64::MAX]).unwrap();
    assert_eq!(
        huge.slice(&[0], &[i64::MAX], &[i64::MAX]).unwrap().shape(),
        [1]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_i32_slices_preserve_full_words_scalar_and_empty_shapes() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input_i32(&[2, 3, 4]).unwrap();
    let mut outputs = vec![x.slice(&[0, 1, 0], &[2, 3, 4], &[1, 1, 2]).unwrap()];
    for axis in 0..3 {
        outputs.push(x.narrow(axis, 1, 1).unwrap());
    }
    outputs.push(x.narrow(2, 4, 0).unwrap());
    let executable = g.compile_outputs(&client, &outputs.to_vec()).unwrap();
    let values: Vec<_> = (0..24)
        .map(|i| match i % 4 {
            0 => i32::MIN + i,
            1 => i32::MAX - i,
            2 => 16_777_217 + i,
            _ => -16_777_217 - i,
        })
        .collect();
    let input = client.buffer(&[2, 3, 4], &values).unwrap();
    let actual = executable.execute(&[&input]).unwrap();
    for (n, buffer) in actual.iter().enumerate() {
        assert_eq!(buffer.dimensions().unwrap(), outputs[n].shape());
        let expected: Vec<_> = values
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                let coords = [i / 12, i / 4 % 3, i % 4];
                let keep = match n {
                    0 => coords[1] >= 1 && coords[2] % 2 == 0,
                    1..=3 => coords[n - 1] == 1,
                    _ => false,
                };
                keep.then_some(v)
            })
            .collect();
        assert_eq!(buffer.to_vec::<i32>().unwrap(), expected);
    }
    for shape in [vec![], vec![2, 0, 4]] {
        let g = Graph::default();
        let x = g.input_i32(&shape).unwrap();
        let sliced = x
            .slice(&vec![0; shape.len()], &shape, &vec![1; shape.len()])
            .unwrap();
        let exe = g.compile_outputs(&client, &[sliced]).unwrap();
        let data = if shape.is_empty() {
            vec![i32::MIN]
        } else {
            vec![]
        };
        let input = client.buffer(&shape, &data).unwrap();
        let out = exe.execute(&[&input]).unwrap();
        assert_eq!(out[0].dimensions().unwrap(), shape);
        assert_eq!(out[0].to_vec::<i32>().unwrap(), data);
        if shape.is_empty() {
            assert!(x.narrow(0, 0, 0).is_err());
        }
    }
}
