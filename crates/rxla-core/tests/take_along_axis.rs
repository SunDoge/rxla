use rxla_core::{Client, Graph};

#[test]
fn rejects_incompatible_indices() {
    let g = Graph::default();
    let x = g.input(&[2, 3]).unwrap();
    let integers = g.input_i32(&[2, 3]).unwrap();
    for shape in [&[2][..], &[1, 1], &[2, 1, 1]] {
        assert!(
            integers
                .take_along_axis(&g.input_i32(shape).unwrap(), 1)
                .is_err()
        );
        assert!(x.take_along_axis(&g.input_i32(shape).unwrap(), 1).is_err());
    }
    let ids = g.input_i32(&[2, 1]).unwrap();
    assert!(integers.take_along_axis(&ids, 2).is_err());
    assert!(
        g.input_i32(&[2, 0])
            .unwrap()
            .take_along_axis(&ids, 1)
            .is_err()
    );
    assert!(
        integers
            .take_along_axis(&Graph::default().input_i32(&[2, 1]).unwrap(), 1)
            .is_err()
    );
    assert!(x.take_along_axis(&ids, 2).is_err());
    assert!(g.input(&[2, 0]).unwrap().take_along_axis(&ids, 1).is_err());
    assert!(
        x.take_along_axis(&Graph::default().input_i32(&[2, 1]).unwrap(), 1)
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_integer_gather_preserves_bits_all_axes_and_empty_outputs() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let shape = [2_i64, 3, 4];
    let values: Vec<_> = (0..24)
        .map(|i| {
            [i32::MIN, i32::MAX, 16_777_217, -16_777_217][i % 4].wrapping_add((i / 4 * 2) as i32)
        })
        .collect();
    for axis in 0..3 {
        for length in [0, 5] {
            let g = Graph::default();
            let x = g.input_i32(&shape).unwrap();
            let mut out_shape = shape;
            out_shape[axis] = length;
            let ids = g.input_i32(&out_shape).unwrap();
            let output = x.take_along_axis(&ids, axis).unwrap();
            let exe = g.compile_outputs(&client, &[output]).unwrap();
            let count = out_shape.iter().product::<i64>() as usize;
            let picks: Vec<_> = (0..count)
                .map(|i| [i32::MIN, 0, 1, 2, i32::MAX][i % 5])
                .collect();
            let input = client.buffer(&shape, &values).unwrap();
            let indices = client.buffer(&out_shape, &picks).unwrap();
            let out = exe.execute(&[&input, &indices]).unwrap();
            assert_eq!(out[0].dimensions().unwrap(), out_shape);
            let stride = shape[axis + 1..].iter().product::<i64>() as usize;
            let expected: Vec<_> = picks
                .iter()
                .enumerate()
                .map(|(i, &pick)| {
                    let outer = i / (length as usize * stride);
                    values[outer * shape[axis] as usize * stride
                        + pick.clamp(0, shape[axis] as i32 - 1) as usize * stride
                        + i % stride]
                })
                .collect();
            assert_eq!(out[0].to_vec::<i32>().unwrap(), expected);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_topk_sampling_maps_candidates_to_original_ids() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let logits = g.input(&[2, 5]).unwrap();
    let bits = g.input_i32(&[2, 2]).unwrap();
    let (values, candidates) = logits.topk(2, 1).unwrap();
    let sample = rxla_core::random::categorical_from_bits(&values, &bits, 1).unwrap();
    let tokens = candidates
        .take_along_axis(&sample.indices.reshape(&[2, 1]).unwrap(), 1)
        .unwrap();
    let exe = g.compile_outputs(&client, &[tokens, sample.valid]).unwrap();
    let input = client
        .buffer(&[2, 5], &[0., 10., 0., 9., 0., 3., 0., 0., 0., 4.])
        .unwrap();
    for (words, expected) in [([0, -1, 0, -1], [3, 0]), ([-1, 0, -1, 0], [1, 4])] {
        let bits = client.buffer(&[2, 2], &words).unwrap();
        let out = exe.execute(&[&input, &bits]).unwrap();
        assert_eq!(out[0].to_vec::<i32>().unwrap(), expected);
        assert_eq!(out[1].to_vec::<f32>().unwrap(), [1., 1.]);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_per_position_gather_every_axis() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let shape = [2i64, 3, 4];
    let values: Vec<_> = (0..24).map(|i| i as f32).collect();
    for axis in 0..3 {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let mut out_shape = shape;
        out_shape[axis] = 5;
        let indices = g.input_i32(&out_shape).unwrap();
        let output = x.take_along_axis(&indices, axis).unwrap();
        assert_eq!(output.shape(), out_shape);
        let exe = g.compile(&client, &output).unwrap();
        let data = client.buffer(&shape, &values).unwrap();
        let count = out_shape.iter().product::<i64>() as usize;
        let stride = shape[axis + 1..].iter().product::<i64>() as usize;
        for shift in [0, 2] {
            let ids: Vec<i32> = (0..count).map(|i| ((i + shift) % 7) as i32 - 2).collect();
            let buffer = client.buffer(&out_shape, &ids).unwrap();
            let actual = exe.execute(&[&data, &buffer]).unwrap()[0]
                .to_vec::<f32>()
                .unwrap();
            let expected: Vec<_> = ids
                .iter()
                .enumerate()
                .map(|(i, &id)| {
                    let outer = i / (5 * stride);
                    let inner = i % stride;
                    values[outer * shape[axis] as usize * stride
                        + id.clamp(0, shape[axis] as i32 - 1) as usize * stride
                        + inner]
                })
                .collect();
            assert_eq!(actual, expected, "axis {axis}");
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_token_log_probs_and_empty_gather() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let logits = g.input(&[2, 3]).unwrap();
    let ids = g.input_i32(&[2, 1]).unwrap();
    let selected = logits
        .log_softmax(1)
        .unwrap()
        .take_along_axis(&ids, 1)
        .unwrap();
    let exe = g.compile(&client, &selected).unwrap();
    let input = client
        .buffer(&[2, 3], &[1000., 0., -1000., 0., 0., 0.])
        .unwrap();
    let indices = client.buffer(&[2, 1], &[2, 1]).unwrap();
    let result = exe.execute(&[&input, &indices]).unwrap()[0]
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(result[0], -2000.);
    assert!((result[1] + 3f32.ln()).abs() < 2e-6);
    for (shape, ids_shape) in [([0, 3], [0, 1]), ([2, 3], [2, 0])] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let ids = g.input_i32(&ids_shape).unwrap();
        let output = x.take_along_axis(&ids, 1).unwrap();
        let exe = g.compile(&client, &output).unwrap();
        let data = vec![0.; shape.iter().product::<i64>() as usize];
        let x = client.buffer(&shape, &data).unwrap();
        let ids = client.buffer::<i32>(&ids_shape, &[]).unwrap();
        assert!(
            exe.execute(&[&x, &ids]).unwrap()[0]
                .to_vec::<f32>()
                .unwrap()
                .is_empty()
        );
    }
}
