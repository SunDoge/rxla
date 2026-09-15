use rxla_core::{Client, Graph, Tensor};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_unequal_integer_fragments_with_empty_part() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axis in 0..3 {
        let g = Graph::default();
        let mut shapes = Vec::new();
        let mut inputs = Vec::new();
        let mut host = Vec::new();
        for (part, length) in [0, 1, 3, 2].into_iter().enumerate() {
            let mut shape = vec![2, 2, 3];
            shape[axis] = length;
            let count = shape.iter().product::<i64>() as usize;
            host.push(
                (0..count)
                    .map(|i| i32::MAX - (part * 100 + i) as i32)
                    .collect::<Vec<_>>(),
            );
            inputs.push(g.input_i32(&shape).unwrap());
            shapes.push(shape);
        }
        let joined = Tensor::concatenate(&inputs, axis).unwrap();
        let mut expected_shape = vec![2, 2, 3];
        expected_shape[axis] = 6;
        assert_eq!(joined.shape(), expected_shape);
        let mut outputs = vec![joined];
        outputs.extend(inputs.iter().cloned());
        let exe = g.compile_outputs(&client, &outputs).unwrap();
        let buffers: Vec<_> = shapes
            .iter()
            .zip(&host)
            .map(|(shape, values)| client.buffer(shape, values).unwrap())
            .collect();
        let actual = exe.execute(&buffers.iter().collect::<Vec<_>>()).unwrap();
        let outer = expected_shape[..axis].iter().product::<i64>() as usize;
        let mut expected = Vec::new();
        for row in 0..outer {
            for values in &host {
                let chunk = values.len() / outer;
                expected.extend_from_slice(&values[row * chunk..(row + 1) * chunk]);
            }
        }
        assert_eq!(actual[0].dimensions().unwrap(), expected_shape);
        assert_eq!(actual[0].to_vec::<i32>().unwrap(), expected);
        for (output, values) in actual[1..].iter().zip(&host) {
            assert_eq!(output.to_vec::<i32>().unwrap(), *values);
        }
    }
}

#[test]
fn integer_join_validation() {
    let g = Graph::default();
    let a = g.input_i32(&[2, 3]).unwrap();
    assert!(Tensor::stack(&[], 0).is_err());
    assert!(Tensor::concatenate(&[], 0).is_err());
    assert!(Tensor::stack(std::slice::from_ref(&a), 3).is_err());
    assert!(Tensor::concatenate(std::slice::from_ref(&a), 2).is_err());
    for b in [
        g.input_i32(&[2, 4]).unwrap(),
        g.input_i32(&[2]).unwrap(),
        Graph::default().input_i32(&[2, 3]).unwrap(),
    ] {
        assert!(Tensor::stack(&[a.clone(), b.clone()], 0).is_err());
        assert!(Tensor::concatenate(&[a.clone(), b], 0).is_err());
    }
    let scalar = g.input_i32(&[]).unwrap();
    assert!(Tensor::concatenate(&[scalar], 0).is_err());
    let huge = g.input_i32(&[i64::MAX, 0]).unwrap();
    let small = g.input_i32(&[1, 0]).unwrap();
    assert!(Tensor::concatenate(&[huge, small], 0).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_integer_stack_and_concatenate_preserve_bits() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![], vec![3], vec![2, 3], vec![2, 0, 3]] {
        let count = shape.iter().product::<i64>() as usize;
        let a: Vec<_> = (0..count)
            .map(|i| [i32::MIN, i32::MAX, 16_777_217][i % 3])
            .collect();
        let b: Vec<_> = a.iter().map(|v| !v).collect();
        for axis in 0..=shape.len() {
            let g = Graph::default();
            let x = g.input_i32(&shape).unwrap();
            let y = g.input_i32(&shape).unwrap();
            let stack = Tensor::stack(&[x.clone(), y.clone()], axis).unwrap();
            let parts = stack.unbind(axis).unwrap();
            let singleton = Tensor::stack(std::slice::from_ref(&x), axis).unwrap();
            let mut outputs = vec![stack, parts[0].clone(), parts[1].clone(), singleton];
            if axis < shape.len() {
                outputs.push(Tensor::concatenate(&[x.clone(), y], axis).unwrap());
                outputs.push(Tensor::concatenate(&[x], axis).unwrap());
            }
            let exe = g.compile_outputs(&client, &outputs).unwrap();
            let x = client.buffer(&shape, &a).unwrap();
            let y = client.buffer(&shape, &b).unwrap();
            let actual = exe.execute(&[&x, &y]).unwrap();
            let interleave = |chunk: usize| {
                let mut expected = Vec::new();
                if chunk != 0 {
                    for (a, b) in a.chunks(chunk).zip(b.chunks(chunk)) {
                        expected.extend_from_slice(a);
                        expected.extend_from_slice(b);
                    }
                }
                expected
            };
            assert_eq!(
                actual[0].to_vec::<i32>().unwrap(),
                interleave(shape[axis..].iter().product::<i64>() as usize)
            );
            assert_eq!(actual[1].to_vec::<i32>().unwrap(), a);
            assert_eq!(actual[2].to_vec::<i32>().unwrap(), b);
            assert_eq!(actual[3].to_vec::<i32>().unwrap(), a);
            if axis < shape.len() {
                assert_eq!(
                    actual[4].to_vec::<i32>().unwrap(),
                    interleave(shape[axis..].iter().product::<i64>() as usize)
                );
                assert_eq!(actual[5].to_vec::<i32>().unwrap(), a);
            }
        }
    }
}
