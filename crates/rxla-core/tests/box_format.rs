use rxla_core::{Client, Tracer};

#[test]
fn box_formats_validate_coordinate_axis() {
    let g = Tracer::default();
    for shape in [vec![], vec![3], vec![4, 1]] {
        let x = g.input(&shape).unwrap();
        assert!(x.box_cxcywh_to_xyxy().is_err());
        assert!(x.box_xyxy_to_cxcywh().is_err());
    }
    for shape in [vec![4], vec![2, 3, 4], vec![2, 0, 4]] {
        let x = g.input(&shape).unwrap();
        assert_eq!(x.box_cxcywh_to_xyxy().unwrap().shape(), shape);
        assert_eq!(x.box_xyxy_to_cxcywh().unwrap().shape(), shape);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_box_formats_match_linear_values_and_vjps() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![4], vec![2, 3, 4], vec![2, 0, 4]] {
        let n = shape.iter().product::<i64>() as usize;
        let values: Vec<_> = (0..n).map(|i| (i as f32 - 10.) * 0.25).collect();
        for forward in [true, false] {
            let g = Tracer::default();
            let x = g.input(&shape).unwrap();
            let y = if forward {
                x.box_cxcywh_to_xyxy()
            } else {
                x.box_xyxy_to_cxcywh()
            }
            .unwrap();
            let back = if forward {
                y.box_xyxy_to_cxcywh()
            } else {
                y.box_cxcywh_to_xyxy()
            }
            .unwrap();
            let seed = g
                .constant(&[4], &[1., 2., 3., 4.])
                .unwrap()
                .broadcast_to(&shape)
                .unwrap();
            let axes: Vec<_> = (0..shape.len()).collect();
            let grad = y
                .mul(&seed)
                .unwrap()
                .sum(&axes, false)
                .unwrap()
                .grad(&[x])
                .unwrap()
                .remove(0);
            let exe = g.compile_many(&client, &[y, back, grad]).unwrap();
            let outputs = exe.run_many(&[&values]).unwrap();
            let expected: Vec<_> = values
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|&[a, b, c, d]| {
                    if forward {
                        [a - c * 0.5, b - d * 0.5, a + c * 0.5, b + d * 0.5]
                    } else {
                        [a * 0.5 + c * 0.5, b * 0.5 + d * 0.5, c - a, d - b]
                    }
                })
                .collect();
            let partials = if forward {
                [4., 6., 1., 1.]
            } else {
                [-2.5, -3., 3.5, 5.]
            };
            let expected_grad: Vec<_> = (0..n).map(|i| partials[i % 4]).collect();
            assert_eq!(outputs[0], expected);
            // These dyadic fixtures are exactly reversible; arbitrary floats are not.
            assert_eq!(outputs[1], values);
            assert_eq!(outputs[2], expected_grad);
        }
    }
}
