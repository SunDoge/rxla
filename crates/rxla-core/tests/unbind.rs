use rxla_core::{Client, Graph, Tensor};

#[test]
fn unbind_validates_axes_and_empty_shapes() {
    let g = Graph::default();
    for shape in [vec![], vec![0], vec![2, 0, 3], vec![1, 2]] {
        let x = g.input(&shape).unwrap();
        let i = g.input_i32(&shape).unwrap();
        assert!(x.unbind(shape.len()).is_err());
        assert!(i.unbind(shape.len()).is_err());
        assert!(x.unbind(usize::MAX).is_err());
        for axis in 0..shape.len() {
            let xs = x.unbind(axis).unwrap();
            let indices = i.unbind(axis).unwrap();
            assert_eq!(xs.len(), shape[axis] as usize);
            assert_eq!(indices.len(), xs.len());
            let mut expected = shape.clone();
            expected.remove(axis);
            for (x, i) in xs.iter().zip(&indices) {
                assert_eq!(x.shape(), expected);
                assert_eq!(i.shape(), expected);
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_unbind_values_and_exact_indices() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![3], vec![2, 3, 2], vec![2, 0, 3]] {
        let count = shape.iter().product::<i64>() as usize;
        let values: Vec<_> = (0..count).map(|i| i as f32 - 4.).collect();
        let words: Vec<_> = (0..count)
            .map(|i| [i32::MIN, i32::MAX, 16_777_217, -16_777_217][i % 4])
            .collect();
        for axis in 0..shape.len() {
            if shape[axis] == 0 {
                continue;
            } // no executable outputs to check
            let g = Graph::default();
            let x = g.input(&shape).unwrap();
            let i = g.input_i32(&shape).unwrap();
            let xs = x.unbind(axis).unwrap();
            let indices = i.unbind(axis).unwrap();
            let mut outputs: Vec<_> = xs.to_vec();
            outputs.extend(indices.iter().cloned());
            outputs.push(Tensor::stack(&xs, axis).unwrap());
            let exe = g.compile_outputs(&client, &outputs).unwrap();
            let x = client.buffer(&shape, &values).unwrap();
            let i = client.buffer(&shape, &words).unwrap();
            let actual = exe.execute(&[&x, &i]).unwrap();
            let stride = shape[axis + 1..].iter().product::<i64>() as usize;
            for part in 0..xs.len() {
                let positions: Vec<_> = (0..count)
                    .filter(|&n| n / stride % xs.len() == part)
                    .collect();
                assert_eq!(
                    actual[part].to_vec::<f32>().unwrap(),
                    positions.iter().map(|&n| values[n]).collect::<Vec<_>>()
                );
                assert_eq!(
                    actual[part + xs.len()].to_vec::<i32>().unwrap(),
                    positions.iter().map(|&n| words[n]).collect::<Vec<_>>()
                );
            }
            assert_eq!(actual.last().unwrap().to_vec::<f32>().unwrap(), values);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_unbind_gradient_and_second_derivative() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axis in 0..2 {
        let g = Graph::default();
        let x = g.input(&[2, 3]).unwrap();
        let parts = x.unbind(axis).unwrap();
        // Use only one slice, twice, so reverse mode must both accumulate its
        // contributions and fill unused input slices with zeros.
        let part = &parts[1];
        let square = part.mul(part).unwrap();
        let loss = square.add(&square).unwrap().sum(&[0], false).unwrap();
        let grad = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
        let second = grad
            .sum(&[0, 1], false)
            .unwrap()
            .grad(std::slice::from_ref(&x))
            .unwrap()
            .remove(0);
        let exe = g.compile_many(&client, &[grad, second]).unwrap();
        let values = [1., -2., 3., -4., 5., -6.];
        let actual = exe.run_many(&[&values]).unwrap();
        for (n, &value) in values.iter().enumerate() {
            let used = if axis == 0 { n / 3 == 1 } else { n % 3 == 1 };
            assert_eq!(actual[0][n], if used { 4. * value } else { 0. });
            assert_eq!(actual[1][n], if used { 4. } else { 0. });
        }
    }
}
