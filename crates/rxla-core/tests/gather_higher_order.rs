use rxla_core::{Client, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_gather_polynomial_derivatives_with_runtime_repeated_clamped_indices() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let shape = [2, 3, 4];
    for axis in 0..3 {
        for batched in [false, true] {
            for empty in [false, true] {
                let g = Tracer::default();
                let x = g.input(&shape).unwrap();
                let mut index_shape = if batched { shape.to_vec() } else { vec![2, 2] };
                if batched {
                    index_shape[axis] = if empty { 0 } else { 4 };
                } else if empty {
                    index_shape[0] = 0;
                }
                let indices = g.input_i32(&index_shape).unwrap();
                let selected = if batched {
                    x.take_along_axis(&indices, axis)
                } else {
                    x.take(&indices, axis)
                }
                .unwrap();
                let mut loss = selected.square().unwrap().mul(&selected).unwrap();
                let mut roots = Vec::new();
                for _ in 0..4 {
                    let axes: Vec<_> = (0..loss.shape().len()).collect();
                    loss = loss
                        .sum(&axes, false)
                        .unwrap()
                        .grad(std::slice::from_ref(&x))
                        .unwrap()
                        .remove(0);
                    roots.push(loss.clone());
                }
                let exe = g.compile_many(&client, &roots).unwrap();
                for phase in [0, 1] {
                    let count = index_shape.iter().product::<i64>() as usize;
                    let ids: Vec<_> = (0..count)
                        .map(|i| [i32::MIN, 1, 1, i32::MAX][(i + phase) % 4])
                        .collect();
                    let values: Vec<_> = (0..24)
                        .map(|i| (i as f32 - 12.) * if phase == 0 { 1. } else { -0.5 })
                        .collect();
                    let mut multiplicity = [0usize; 24];
                    let outer = shape[..axis].iter().product::<i64>() as usize;
                    let inner = shape[axis + 1..].iter().product::<i64>() as usize;
                    let width = shape[axis] as usize;
                    let selections = if empty { 0 } else { 4 };
                    for o in 0..outer {
                        for s in 0..selections {
                            for j in 0..inner {
                                let id = ids[if batched {
                                    (o * selections + s) * inner + j
                                } else {
                                    s
                                }];
                                let id = id.clamp(0, width as i32 - 1) as usize;
                                multiplicity[(o * width + id) * inner + j] += 1;
                            }
                        }
                    }
                    let xb = client.buffer(&shape, &values).unwrap();
                    let ib = client.buffer(&index_shape, &ids).unwrap();
                    let actual = exe.execute(&[&xb, &ib]).unwrap();
                    for (order, output) in actual.iter().enumerate() {
                        let output = output.to_vec::<f32>().unwrap();
                        for i in 0..24 {
                            let x = values[i];
                            let expected =
                                multiplicity[i] as f32 * [3. * x * x, 6. * x, 6., 0.][order];
                            assert_eq!(
                                output[i], expected,
                                "axis={axis} batched={batched} empty={empty} phase={phase} order={order} i={i}"
                            );
                        }
                    }
                }
            }
        }
    }
}
