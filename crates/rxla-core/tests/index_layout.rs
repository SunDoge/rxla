use rxla_core::{Client, Tracer};

#[test]
fn index_layout_validation() {
    let g = Tracer::default();
    let ids = g.input_i32(&[2, 3]).unwrap();
    assert_eq!(ids.reshape(&[6]).unwrap().shape(), [6]);
    assert_eq!(ids.transpose(&[1, 0]).unwrap().shape(), [3, 2]);
    for shape in [vec![5], vec![-1], vec![i64::MAX, i64::MAX]] {
        assert!(ids.reshape(&shape).is_err());
    }
    for permutation in [vec![0], vec![0, 0], vec![0, 2], vec![0, 1, 2]] {
        assert!(ids.transpose(&permutation).is_err());
    }
    assert_eq!(g.scalar_i32(1).unwrap().transpose(&[]).unwrap().shape(), []);
    assert_eq!(
        g.input_i32(&[0, 3])
            .unwrap()
            .reshape(&[2, 0])
            .unwrap()
            .shape(),
        [2, 0]
    );
}

#[test]
fn index_singleton_axes_validate_before_recording_nodes() {
    let g = Tracer::default();
    for shape in [vec![], vec![1], vec![2, 1, 3], vec![2, 0, 1]] {
        let ids = g.input_i32(&shape).unwrap();
        let before = g.stablehlo_many(std::slice::from_ref(&ids)).unwrap();
        for axis in [shape.len() + 1, usize::MAX] {
            assert!(ids.unsqueeze(axis).is_err());
        }
        for axis in 0..=shape.len() {
            if shape.get(axis) != Some(&1) {
                assert!(ids.squeeze(axis).is_err());
            }
        }
        assert!(ids.squeeze(usize::MAX).is_err());
        assert_eq!(
            before,
            g.stablehlo_many(std::slice::from_ref(&ids)).unwrap()
        );
        for axis in 0..=shape.len() {
            let expanded = ids.unsqueeze(axis).unwrap();
            let mut expected = shape.clone();
            expected.insert(axis, 1);
            assert_eq!(expanded.shape(), expected);
            assert_eq!(expanded.squeeze(axis).unwrap().shape(), shape);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_index_singleton_axes_preserve_integer_bits() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![], vec![1], vec![2, 1, 3], vec![2, 0, 1]] {
        let g = Tracer::default();
        let ids = g.input_i32(&shape).unwrap();
        let values: Vec<_> = (0..ids.numel())
            .map(|i| [i32::MIN, i32::MAX, -1, 0, 16_777_217, 42][i % 6])
            .collect();
        let input = client.buffer(&shape, &values).unwrap();
        let mut outputs = Vec::new();
        for axis in 0..=shape.len() {
            let expanded = ids.unsqueeze(axis).unwrap();
            outputs.push(expanded.clone());
            outputs.push(expanded.squeeze(axis).unwrap());
        }
        let exe = g.compile_many(&client, &outputs).unwrap();
        for output in exe.execute(&[&input]).unwrap() {
            assert_eq!(output.to_vec::<i32>().unwrap(), values);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_index_layouts_feed_batched_log_prob_selection() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let logits = g.input(&[2, 3, 4]).unwrap();
    // Sequence-major IDs supplied by the host; normalize layout inside the graph.
    let ids = g.input_i32(&[3, 2]).unwrap();
    let ids = ids.transpose(&[1, 0]).unwrap().unsqueeze(2).unwrap();
    let output = logits
        .log_softmax(2)
        .unwrap()
        .take_along_axis(&ids, 2)
        .unwrap();
    let exe = g.compile(&client, &output).unwrap();
    let values: Vec<_> = (0..24).map(|i| ((i * 13 % 17) as f32 - 8.) / 4.).collect();
    let input = client.buffer(&[2, 3, 4], &values).unwrap();
    for ids in [[0, 3, 1, 2, 2, 0], [3, 0, 2, 1, 1, 3]] {
        let buffer = client.buffer(&[3, 2], &ids).unwrap();
        let actual = exe.execute(&[&input, &buffer]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap();
        for batch in 0..2 {
            for step in 0..3 {
                let row = batch * 3 + step;
                let values = &values[row * 4..row * 4 + 4];
                let denominator: f64 = values.iter().map(|&x| (x as f64).exp()).sum();
                let expected = values[ids[step * 2 + batch] as usize] as f64 - denominator.ln();
                assert!((actual[row] as f64 - expected).abs() < 2e-6);
            }
        }
    }
    // Constant scalar reshape and empty index layouts remain valid gather inputs.
    let g = Tracer::default();
    let table = g.constant(&[3], &[10., 20., 30.]).unwrap();
    let scalar = g.scalar_i32(1).unwrap().reshape(&[1, 1]).unwrap();
    let empty = g
        .constant_i32(&[0, 2], &[])
        .unwrap()
        .transpose(&[1, 0])
        .unwrap()
        .reshape(&[0])
        .unwrap();
    let outputs = g
        .compile_many(
            &client,
            &[
                table.take(&scalar, 0).unwrap(),
                table.take(&empty, 0).unwrap(),
            ],
        )
        .unwrap()
        .run_many(&[])
        .unwrap();
    assert_eq!(outputs[0], [20.]);
    assert!(outputs[1].is_empty());
}
