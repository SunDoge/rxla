use rxla_core::{Client, Conv2dOptions, Graph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_ocr_stem_layout_matches_f64_with_scale_aware_roundoff_budget() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Graph::default();
    let x = graph.input(&[1, 3, 9, 11]).unwrap();
    let w = graph.input(&[16, 3, 3, 3]).unwrap();
    let b = graph.input(&[16]).unwrap();
    let y = x
        .transpose(&[0, 2, 3, 1])
        .unwrap()
        .conv2d(
            &w.transpose(&[2, 3, 1, 0]).unwrap(),
            Conv2dOptions {
                strides: [2, 2],
                padding: [[1, 1], [1, 1]],
                ..Default::default()
            },
        )
        .unwrap()
        .add(&b.broadcast_to(&[1, 5, 6, 16]).unwrap())
        .unwrap()
        .transpose(&[0, 3, 1, 2])
        .unwrap();
    assert_eq!(y.shape(), [1, 16, 5, 6]);
    let executable = graph.compile(&client, &y).unwrap();
    let weights: Vec<f32> = (0..16 * 3 * 3 * 3)
        .map(|i| ((i * 17 % 43) as f32 - 21.) * 0.137)
        .collect();
    let wb = client.buffer(&[16, 3, 3, 3], &weights).unwrap();
    // 27 products plus bias: conservative 56-rounding FP32 dot-product budget.
    // Applies to these finite, non-subnormal fixtures, not every XLA algorithm.
    let u = f64::from(f32::EPSILON) / 2.;
    let gamma = 56. * u / (1. - 56. * u);
    for scale in [0.001f32, 1., 1000.] {
        let values: Vec<f32> = (0..3 * 9 * 11)
            .map(|i| ((i * 13 % 37) as f32 - 18.) * 0.173 * scale)
            .collect();
        let bias: Vec<f32> = (0..16).map(|i| (i as f32 - 8.) * 0.019 * scale).collect();
        let xb = client.buffer(&[1, 3, 9, 11], &values).unwrap();
        let bb = client.buffer(&[16], &bias).unwrap();
        let actual = executable.execute(&[&xb, &wb, &bb]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap();
        let mut rejects_zero = 0;
        for oc in 0..16 {
            for oh in 0..5 {
                for ow in 0..6 {
                    let mut expected = f64::from(bias[oc]);
                    let mut magnitude = expected.abs();
                    for ic in 0..3 {
                        for kh in 0..3 {
                            for kw in 0..3 {
                                let ih = oh as isize * 2 + kh as isize - 1;
                                let iw = ow as isize * 2 + kw as isize - 1;
                                if !(0..9).contains(&ih) || !(0..11).contains(&iw) {
                                    continue;
                                }
                                let term =
                                    f64::from(values[(ic * 9 + ih as usize) * 11 + iw as usize])
                                        * f64::from(weights[((oc * 3 + ic) * 3 + kh) * 3 + kw]);
                                expected += term;
                                magnitude += term.abs();
                            }
                        }
                    }
                    let value = f64::from(actual[(oc * 5 + oh) * 6 + ow]);
                    let tolerance = gamma * magnitude + 1e-12;
                    assert!(
                        value.is_finite() && (value - expected).abs() <= tolerance,
                        "scale={scale}, ({oc},{oh},{ow}): {value} vs {expected}, budget={tolerance}"
                    );
                    rejects_zero += usize::from(expected.abs() > tolerance);
                }
            }
        }
        assert!(rejects_zero > actual.len() / 2);
    }
}
