use rxla_core::{Client, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_dynamic_index_dependencies_are_not_differentiated_and_scalars_work() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let x = graph.input(&[2]).unwrap();
    let index = x.argmax(0, false).unwrap();
    let y = x.dynamic_slice(&[index], &[1]).unwrap();
    let loss = y.square().unwrap().sum(&[0], false).unwrap();
    let grad = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let second = grad
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let exe = graph.compile_many(&client, &[grad, second]).unwrap();
    assert_eq!(
        exe.run_many(&[&[2., -3.]]).unwrap(),
        [vec![4., 0.], vec![2., 0.]]
    );
    assert_eq!(
        exe.run_many(&[&[1., 4.]]).unwrap(),
        [vec![0., 8.], vec![0., 2.]]
    );
    let graph = Tracer::default();
    let x = graph.input(&[]).unwrap();
    let u = graph.input(&[]).unwrap();
    let y = x
        .dynamic_update_slice(&u, &[])
        .unwrap()
        .dynamic_slice(&[], &[])
        .unwrap();
    let gradients = y.grad(&[x, u]).unwrap();
    let exe = graph.compile_many(&client, &gradients).unwrap();
    assert_eq!(exe.run_many(&[&[2.], &[3.]]).unwrap(), [vec![0.], vec![1.]]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_dynamic_slice_and_update_derivatives_follow_clamped_runtime_positions() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for updating in [false, true] {
        for empty in [false, true] {
            let graph = Tracer::default();
            let x = graph.input(&[3, 4]).unwrap();
            let shape = [if empty { 0 } else { 2 }, 2];
            let u = graph.input(&shape).unwrap();
            let starts = [
                graph.input_i32_scalar().unwrap(),
                graph.input_i32_scalar().unwrap(),
            ];
            let y = if updating {
                x.dynamic_update_slice(&u, &starts)
            } else {
                x.dynamic_slice(&starts, &shape)
            }
            .unwrap();
            let mut loss = y
                .square()
                .unwrap()
                .mul(&y)
                .unwrap()
                .sum(&[0, 1], false)
                .unwrap();
            let mut roots = Vec::new();
            for _ in 0..4 {
                let gradients = loss.grad(&[x.clone(), u.clone()]).unwrap();
                loss = gradients[0]
                    .sum(&[0, 1], false)
                    .unwrap()
                    .add(&gradients[1].sum(&[0, 1], false).unwrap())
                    .unwrap();
                roots.extend(gradients);
            }
            let executable = graph.compile_many(&client, &roots).unwrap();
            for (position, scale) in [
                ([-9_i32, i32::MAX], 1_f32),
                ([1, 1], -0.5),
                ([i32::MAX, i32::MIN], 2.),
            ] {
                let values: Vec<_> = (0..12).map(|i| (i as f32 - 4.) * scale).collect();
                let updates: Vec<_> = (0..if empty { 0 } else { 4 })
                    .map(|i| (i as f32 + 2.) * scale)
                    .collect();
                let xb = client.buffer(&[3, 4], &values).unwrap();
                let ub = client.buffer(&shape, &updates).unwrap();
                let row = client.buffer(&[], &[position[0]]).unwrap();
                let col = client.buffer(&[], &[position[1]]).unwrap();
                let results = executable.execute(&[&xb, &ub, &row, &col]).unwrap();
                let row = position[0].clamp(0, 3 - shape[0] as i32) as usize;
                let col = position[1].clamp(0, 2) as usize;
                for order in 0..4 {
                    let actual_x = results[2 * order].to_vec::<f32>().unwrap();
                    let actual_u = results[2 * order + 1].to_vec::<f32>().unwrap();
                    for (i, &x) in values.iter().enumerate() {
                        let selected = !empty
                            && i / 4 >= row
                            && i / 4 < row + 2
                            && i % 4 >= col
                            && i % 4 < col + 2;
                        let active = if updating { !selected } else { selected };
                        let expected = if active {
                            [3. * x * x, 6. * x, 6., 0.][order]
                        } else {
                            0.
                        };
                        assert_eq!(
                            actual_x[i], expected,
                            "update={updating} empty={empty} pos={position:?} order={order} i={i}"
                        );
                    }
                    for (i, &u) in updates.iter().enumerate() {
                        let expected = if updating {
                            [3. * u * u, 6. * u, 6., 0.][order]
                        } else {
                            0.
                        };
                        assert_eq!(actual_u[i], expected);
                    }
                }
            }
        }
    }
}
