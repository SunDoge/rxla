use rxla_core::{Client, Tracer};

#[test]
fn index_tensor_validation() {
    let g = Tracer::default();
    assert!(g.input_i32(&[-1]).is_err());
    assert!(g.constant_i32(&[2], &[1]).is_err());
    let x = g.input(&[4, 3]).unwrap();
    let ids = g.input_i32(&[2, 5]).unwrap();
    assert_eq!(ids.shape(), [2, 5]);
    assert_eq!(x.take(&ids, 0).unwrap().shape(), [2, 5, 3]);
    assert!(x.take(&ids, 2).is_err());
    assert!(g.input(&[0, 3]).unwrap().take(&ids, 0).is_err());
    assert!(
        x.take(&Tracer::default().input_i32_scalar().unwrap(), 0)
            .is_err()
    );
    assert!(
        x.dynamic_slice(&[ids, g.scalar_i32(0).unwrap()], &[1, 3])
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_gather_shapes_constants_and_clamping() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, axis, index_shape, indices) in [
        (vec![2, 3, 4], 1, vec![2, 2], vec![-4, 2, 1, 99]),
        (vec![3, 4], 0, vec![], vec![2]),
        (vec![2, 3], 1, vec![2], vec![2, 0]),
        (vec![4], 0, vec![3], vec![3, 3, 0]),
        (vec![4, 2], 0, vec![0], vec![]),
    ] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let ids = g.input_i32(&index_shape).unwrap();
        let constants = g.constant_i32(&index_shape, &indices).unwrap();
        let y = x.take(&ids, axis).unwrap();
        let c = x.take(&constants, axis).unwrap();
        let values: Vec<f32> = (0..shape.iter().product::<i64>())
            .map(|i| i as f32 / 2. - 3.)
            .collect();
        let inner = shape[axis + 1..].iter().product::<i64>() as usize;
        let outer = shape[..axis].iter().product::<i64>() as usize;
        let mut expected = Vec::new();
        for o in 0..outer {
            for &index in &indices {
                let start = (o * shape[axis] as usize
                    + index.clamp(0, shape[axis] as i32 - 1) as usize)
                    * inner;
                expected.extend_from_slice(&values[start..start + inner]);
            }
        }
        let exe = g.compile_many(&client, &[y.clone(), c]).unwrap();
        let input = client.buffer(&shape, &values).unwrap();
        let index = client.buffer(&index_shape, &indices).unwrap();
        let outputs = exe.execute(&[&input, &index]).unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            assert_eq!(output.dimensions().unwrap(), y.shape());
            assert_eq!(output.to_vec::<f32>().unwrap(), expected);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_batched_embedding_projection() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let table = g.input(&[4, 3]).unwrap();
    let ids = g.input_i32(&[2, 2]).unwrap();
    let weights = g.constant(&[3, 2], &[1., -1., 0.5, 0., 0., 1.]).unwrap();
    let output = table
        .take(&ids, 0)
        .unwrap()
        .matmul(&weights)
        .unwrap()
        .softmax(2)
        .unwrap();
    let executable = g.compile(&client, &output).unwrap();
    let table = client
        .buffer(&[4, 3], &[1., 0., 0., 0., 2., 0., 0., 0., 3., 1., 2., 3.])
        .unwrap();
    for tokens in [[0, 1, 2, 3], [3, 2, 1, 0]] {
        let ids = client.buffer(&[2, 2], &tokens).unwrap();
        let result = executable.execute(&[&table, &ids]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap();
        for (row, token) in result.chunks(2).zip(tokens) {
            let logit_difference = [2f64, 1., -3., 0.][token as usize];
            let p = 1. / (1. + (-logit_difference).exp());
            assert!((row[0] as f64 - p).abs() < 1e-6);
            assert!((row[1] as f64 - (1. - p)).abs() < 1e-6);
        }
    }
}
