use rxla_core::{Client, Graph};

#[test]
fn rounding_stops_unsupported_reverse_paths() {
    let g = Graph::default();
    let x = g.input(&[3]).unwrap();
    let mask = x.gt_mask(&x).unwrap();
    for y in [
        mask.floor().unwrap(),
        mask.ceil().unwrap(),
        mask.round().unwrap(),
        mask.round_ties_even().unwrap(),
    ] {
        assert!(
            y.sum(&[0], false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .is_ok()
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_rounding_matches_rust_and_explicit_zero_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let data = [
        0.5f32,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        0.,
        -0.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        -f32::MAX,
        f32::from_bits(0x3effffff),
        f32::from_bits(0x3f000001),
        -0.1,
        0.1,
        16_777_216.,
    ];
    for shape in [vec![], vec![3, 6], vec![0, 6]] {
        let n = shape.iter().product::<i64>() as usize;
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let axes: Vec<_> = (0..shape.len()).collect();
        let rounded = [
            x.floor().unwrap(),
            x.ceil().unwrap(),
            x.round().unwrap(),
            x.round_ties_even().unwrap(),
        ];
        let seed = g.constant(&shape, &vec![f32::INFINITY; n]).unwrap();
        let mut outputs = rounded.to_vec();
        for y in &rounded {
            let dx = y
                .sum(&axes, false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            let ddx = dx
                .sum(&axes, false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            let vjp = y.vjp(std::slice::from_ref(&x), &seed).unwrap().remove(0);
            outputs.extend([dx, ddx, vjp]);
        }
        let executable = g.compile_many(&client, &outputs).unwrap();
        let input = client.buffer(&shape, &data[..n]).unwrap();
        let actual = executable.execute(&[&input]).unwrap();
        for (k, output) in actual.iter().enumerate() {
            assert_eq!(output.dimensions().unwrap(), shape);
            for (i, a) in output.to_vec::<f32>().unwrap().into_iter().enumerate() {
                let expected = match k {
                    0 => data[i].floor(),
                    1 => data[i].ceil(),
                    2 => data[i].round(),
                    3 => data[i].round_ties_even(),
                    _ => 0.,
                };
                if expected.is_nan() {
                    assert!(a.is_nan());
                } else {
                    assert_eq!(
                        a.to_bits(),
                        expected.to_bits(),
                        "op={k} input={:?}",
                        data[i]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_fake_quantization_uses_explicit_clipped_surrogate() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[9]).unwrap();
    let clipped = x.clamp(-1., 1.).unwrap();
    let forward = clipped
        .mul_scalar(2.)
        .unwrap()
        .round_ties_even()
        .unwrap()
        .mul_scalar(0.5)
        .unwrap();
    let quantized = forward.with_gradient_of(&clipped).unwrap();
    let gradient = quantized
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let executable = g.compile_many(&client, &[quantized, gradient]).unwrap();
    let input = client
        .buffer(&[9], &[-2., -1., -0.75, -0.25, 0., 0.25, 0.75, 1., 2.])
        .unwrap();
    let outputs = executable.execute(&[&input]).unwrap();
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        [-1., -1., -1., 0., 0., 0., 1., 1., 1.]
    );
    assert_eq!(
        outputs[1].to_vec::<f32>().unwrap(),
        [0., 0.5, 1., 1., 1., 1., 1., 0.5, 0.]
    );
}
