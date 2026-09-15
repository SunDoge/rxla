use rxla_core::{Client, Graph};

#[test]
fn attention_head_layout_helpers_preserve_exact_hlo() {
    // TinyLlama decode and chunk-prefill head shapes, without loading weights
    // or a native plugin. Equal protos retain the same compiler input/cache key.
    for chunk in [1, 8] {
        for heads in [4, 32] {
            let build = |helpers: bool| {
                let g = Graph::default();
                let x = g.input(&[chunk, heads * 64]).unwrap();
                let split = if helpers {
                    x.unflatten(1, &[heads, 64])
                        .unwrap()
                        .move_axis(1, 0)
                        .unwrap()
                } else {
                    x.reshape(&[chunk, heads, 64])
                        .unwrap()
                        .transpose(&[1, 0, 2])
                        .unwrap()
                };
                let merged = if helpers {
                    split.move_axis(0, 1).unwrap().flatten(1, 2).unwrap()
                } else {
                    split
                        .transpose(&[1, 0, 2])
                        .unwrap()
                        .reshape(&[chunk, heads * 64])
                        .unwrap()
                };
                g.stablehlo_many(&[split, merged]).unwrap()
            };
            assert_eq!(build(true), build(false));
        }
    }
}

#[test]
fn axis_reorder_validates_axes_and_preserves_other_dimensions() {
    let g = Graph::default();
    let x = g.input(&[2, 3, 4, 5]).unwrap();
    assert_eq!(x.move_axis(1, 3).unwrap().shape(), [2, 4, 5, 3]);
    assert_eq!(x.move_axis(3, 1).unwrap().shape(), [2, 5, 3, 4]);
    assert_eq!(x.swap_axes(1, 3).unwrap().shape(), [2, 5, 4, 3]);
    for shape in [vec![], vec![2, 0, 3]] {
        let x = g.input(&shape).unwrap();
        let ids = g.input_i32(&shape).unwrap();
        for (a, b) in [
            (0, shape.len()),
            (usize::MAX, 0),
            (shape.len(), shape.len()),
        ] {
            assert!(x.move_axis(a, b).is_err());
            assert!(ids.move_axis(a, b).is_err());
            assert!(x.swap_axes(a, b).is_err());
            assert!(ids.swap_axes(a, b).is_err());
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_axis_reorder_values_and_nonuniform_vjp() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![2, 3, 4], vec![2, 0, 3], vec![3]] {
        let n = shape.iter().product::<i64>() as usize;
        let values: Vec<_> = (0..n).map(|i| i as f32 - 5.).collect();
        let integers: Vec<_> = (0..n).map(|i| i32::MAX - i as i32).collect();
        for a in 0..shape.len() {
            for b in 0..shape.len() {
                for swap in [false, true] {
                    let g = Graph::default();
                    let x = g.input(&shape).unwrap();
                    let ids = g.input_i32(&shape).unwrap();
                    let y = if swap {
                        x.swap_axes(a, b)
                    } else {
                        x.move_axis(a, b)
                    }
                    .unwrap();
                    let iy = if swap {
                        ids.swap_axes(a, b)
                    } else {
                        ids.move_axis(a, b)
                    }
                    .unwrap();
                    // Independent coordinate mapping: map each original axis
                    // to its new position rather than reuse the implementation's permutation.
                    let destination: Vec<_> = (0..shape.len())
                        .map(|axis| {
                            if swap {
                                if axis == a {
                                    b
                                } else if axis == b {
                                    a
                                } else {
                                    axis
                                }
                            } else if axis == a {
                                b
                            } else if a < b && axis > a && axis <= b {
                                axis - 1
                            } else if b < a && axis >= b && axis < a {
                                axis + 1
                            } else {
                                axis
                            }
                        })
                        .collect();
                    let weights: Vec<_> = (0..n).map(|i| i as f32 * 0.5 - 3.).collect();
                    let seed = g.constant(y.shape(), &weights).unwrap();
                    let axes: Vec<_> = (0..shape.len()).collect();
                    let grad = y
                        .mul(&seed)
                        .unwrap()
                        .sum(&axes, false)
                        .unwrap()
                        .grad(&[x])
                        .unwrap()
                        .remove(0);
                    let mut expected = vec![0.; n];
                    let mut expected_ids = vec![0; n];
                    let mut expected_grad = vec![0.; n];
                    for input in 0..n {
                        let mut remainder = input;
                        let mut coordinates = vec![0; shape.len()];
                        for axis in (0..shape.len()).rev() {
                            coordinates[destination[axis]] = remainder % shape[axis] as usize;
                            remainder /= shape[axis] as usize;
                        }
                        let output = coordinates
                            .iter()
                            .zip(y.shape())
                            .fold(0, |index, (&coordinate, &size)| {
                                index * size as usize + coordinate
                            });
                        expected[output] = values[input];
                        expected_ids[output] = integers[input];
                        expected_grad[input] = weights[output];
                    }
                    let exe = g.compile_outputs(&client, &[y.clone(), iy, grad]).unwrap();
                    let input = client.buffer(&shape, &values).unwrap();
                    let ids = client.buffer(&shape, &integers).unwrap();
                    let outputs = exe.execute(&[&input, &ids]).unwrap();
                    assert_eq!(outputs[0].dimensions().unwrap(), y.shape());
                    assert_eq!(outputs[1].dimensions().unwrap(), y.shape());
                    assert_eq!(outputs[2].dimensions().unwrap(), shape);
                    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), expected);
                    assert_eq!(outputs[1].to_vec::<i32>().unwrap(), expected_ids);
                    assert_eq!(outputs[2].to_vec::<f32>().unwrap(), expected_grad);
                }
            }
        }
    }
}
