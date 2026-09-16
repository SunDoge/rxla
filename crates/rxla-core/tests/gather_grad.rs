use rxla_core::{Client, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_embedding_nonlinear_loss_accumulates_repeated_rows() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let table = g.input(&[4, 2]).unwrap();
    let ids = g.input_i32(&[3]).unwrap();
    let embeddings = table.take(&ids, 0).unwrap();
    let loss = embeddings
        .mul(&embeddings)
        .unwrap()
        .sum(&[0, 1], false)
        .unwrap();
    let grad = loss.grad(&[table]).unwrap();
    let exe = g.compile_many(&client, &grad).unwrap();
    let table = client
        .buffer(&[4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.])
        .unwrap();
    let ids = client.buffer(&[3], &[2, 2, -1]).unwrap();
    assert_eq!(
        exe.execute(&[&table, &ids]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [2., 4., 0., 0., 20., 24., 0., 0.]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_indexed_log_probability_loss_gradient() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let logits = g.input(&[2, 3]).unwrap();
    let labels = g.input_i32(&[2, 1]).unwrap();
    let loss = logits
        .log_softmax(1)
        .unwrap()
        .take_along_axis(&labels, 1)
        .unwrap()
        .neg()
        .unwrap()
        .mean(&[0, 1], false)
        .unwrap();
    let grad = loss.grad(&[logits]).unwrap();
    let exe = g.compile_many(&client, &grad).unwrap();
    let values = [1.0f32, 2., 3., 3., 1., -1.];
    let input = client.buffer(&[2, 3], &values).unwrap();
    for targets in [[0, 2], [2, 0]] {
        let labels = client.buffer(&[2, 1], &targets).unwrap();
        let actual = exe.execute(&[&input, &labels]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap();
        for row in 0..2 {
            let total: f64 = values[row * 3..row * 3 + 3]
                .iter()
                .map(|v| (*v as f64).exp())
                .sum();
            for col in 0..3 {
                let p = (values[row * 3 + col] as f64).exp() / total;
                let expected = (p - if col as i32 == targets[row] { 1. } else { 0. }) / 2.;
                assert!((actual[row * 3 + col] as f64 - expected).abs() < 1e-6);
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_take_grad_all_axes_multidimensional_repeated_and_clamped_indices() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let shape = [2, 3, 4];
    for axis in 0..3 {
        for index_shape in [vec![], vec![2, 2], vec![0]] {
            let g = Tracer::default();
            let x = g.input(&shape).unwrap();
            let ids = g.input_i32(&index_shape).unwrap();
            let selected = x.take(&ids, axis).unwrap();
            let weights = g.input(selected.shape()).unwrap();
            let axes: Vec<_> = (0..selected.shape().len()).collect();
            let loss = selected.mul(&weights).unwrap().sum(&axes, false).unwrap();
            let grad = loss.grad(&[x]).unwrap();
            let exe = g.compile_many(&client, &grad).unwrap();
            let source = client.buffer(&shape, &[1.; 24]).unwrap();
            let outer = shape[..axis].iter().product::<i64>() as usize;
            let inner = shape[axis + 1..].iter().product::<i64>() as usize;
            let count = index_shape.iter().product::<i64>() as usize;
            for indices in [[-1, 1, 1, i32::MAX], [i32::MIN, 0, 0, 0]] {
                let values: Vec<_> = (0..outer * count * inner)
                    .map(|i| (i as f32 % 7. - 3.) * 0.25)
                    .collect();
                let wb = client.buffer(selected.shape(), &values).unwrap();
                let ib = client.buffer(&index_shape, &indices[..count]).unwrap();
                let mut expected = vec![0.; 24];
                for o in 0..outer {
                    for (i, &index) in indices[..count].iter().enumerate() {
                        let index = index.clamp(0, shape[axis] as i32 - 1) as usize;
                        for j in 0..inner {
                            expected[(o * shape[axis] as usize + index) * inner + j] +=
                                values[(o * count + i) * inner + j];
                        }
                    }
                }
                assert_eq!(
                    exe.execute(&[&source, &ib, &wb]).unwrap()[0]
                        .to_vec::<f32>()
                        .unwrap(),
                    expected
                );
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_take_along_axis_grad_all_axes_and_empty_batches() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axis in 0..3 {
        let shape = [2, 3, 4];
        let mut index_shape = shape;
        index_shape[axis] = 5;
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let ids = g.input_i32(&index_shape).unwrap();
        let weights = g.input(&index_shape).unwrap();
        let y = x.take_along_axis(&ids, axis).unwrap();
        let grad = y
            .mul(&weights)
            .unwrap()
            .sum(&[0, 1, 2], false)
            .unwrap()
            .grad(&[x])
            .unwrap();
        let exe = g.compile_many(&client, &grad).unwrap();
        let count = index_shape.iter().product::<i64>() as usize;
        let indices: Vec<_> = (0..count).map(|i| [-2, 0, 1, 1, 99][i % 5]).collect();
        let values: Vec<_> = (0..count).map(|i| (i % 11) as f32 * 0.25).collect();
        let mut expected = vec![0.; 24];
        for i in 0..count {
            let mut coordinates = [0; 3];
            let mut offset = i;
            for d in (0..3).rev() {
                coordinates[d] = offset % index_shape[d] as usize;
                offset /= index_shape[d] as usize;
            }
            coordinates[axis] = indices[i].clamp(0, shape[axis] as i32 - 1) as usize;
            let index = (coordinates[0] * 3 + coordinates[1]) * 4 + coordinates[2];
            expected[index] += values[i];
        }
        let source = client.buffer(&shape, &[1.; 24]).unwrap();
        let ib = client.buffer(&index_shape, &indices).unwrap();
        let wb = client.buffer(&index_shape, &values).unwrap();
        assert_eq!(
            exe.execute(&[&source, &ib, &wb]).unwrap()[0]
                .to_vec::<f32>()
                .unwrap(),
            expected
        );
    }
    let g = Tracer::default();
    let x = g.input(&[0, 3]).unwrap();
    let ids = g.input_i32(&[0, 2]).unwrap();
    let grad = x
        .take_along_axis(&ids, 1)
        .unwrap()
        .sum(&[0, 1], false)
        .unwrap()
        .grad(&[x])
        .unwrap();
    let exe = g.compile_many(&client, &grad).unwrap();
    let source = client.buffer::<f32>(&[0, 3], &[]).unwrap();
    let ib = client.buffer::<i32>(&[0, 2], &[]).unwrap();
    assert!(
        exe.execute(&[&source, &ib]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap()
            .is_empty()
    );
}
