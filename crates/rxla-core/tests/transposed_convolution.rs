use rxla_core::{Client, ConvTranspose2dOptions as Options, Tracer};

#[test]
fn transposed_convolution_validation() {
    let g = Tracer::default();
    let x = g.input(&[1, 3, 4, 2]).unwrap();
    let k = g.input(&[2, 2, 5, 2]).unwrap();
    assert_eq!(
        x.conv_transpose2d(
            &k,
            Options {
                strides: [2, 2],
                ..Default::default()
            }
        )
        .unwrap()
        .shape(),
        [1, 6, 8, 5]
    );
    for options in [
        Options {
            strides: [0, 1],
            ..Default::default()
        },
        Options {
            dilation: [-1, 1],
            ..Default::default()
        },
        Options {
            padding: [[-1, 0], [0, 0]],
            ..Default::default()
        },
        Options {
            output_padding: [1, 0],
            ..Default::default()
        },
        Options {
            padding: [[99, 0], [0, 0]],
            ..Default::default()
        },
        Options {
            strides: [i64::MAX, 1],
            ..Default::default()
        },
        Options {
            dilation: [i64::MAX, 1],
            ..Default::default()
        },
    ] {
        assert!(x.conv_transpose2d(&k, options).is_err());
    }
    assert!(
        x.conv_transpose2d(&g.input(&[2, 2, 2, 5]).unwrap(), Options::default())
            .is_err()
    );
    assert!(
        x.conv_transpose2d(
            &Tracer::default().input(&[2, 2, 5, 2]).unwrap(),
            Options::default()
        )
        .is_err()
    );
    assert!(
        g.input(&[1, 0, 4, 2])
            .unwrap()
            .conv_transpose2d(&k, Options::default())
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_transposed_convolution_matches_scatter_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let xs = [2, 2, 3, 2];
    let ks = [2, 3, 3, 2];
    let values: Vec<f32> = (0..24).map(|i| (i % 7) as f32 / 4. - 1.).collect();
    let weights: Vec<f32> = (0..36).map(|i| (i % 11) as f32 / 8. - 0.5).collect();
    for options in [
        Options::default(),
        Options {
            strides: [2, 2],
            ..Default::default()
        },
        Options {
            strides: [2, 3],
            padding: [[1, 0], [0, 2]],
            output_padding: [1, 2],
            ..Default::default()
        },
        Options {
            dilation: [2, 2],
            padding: [[0, 1], [2, 0]],
            ..Default::default()
        },
        // Cropping exceeds kernel extent on one side (negative HLO padding).
        Options {
            strides: [3, 2],
            padding: [[2, 0], [0, 0]],
            ..Default::default()
        },
    ] {
        let g = Tracer::default();
        let x = g.input(&xs).unwrap();
        let k = g.input(&ks).unwrap();
        let y = x.conv_transpose2d(&k, options).unwrap();
        let ys = y.shape();
        let mut expected = vec![0f64; ys.iter().product::<i64>() as usize];
        for n in 0..xs[0] {
            for h in 0..xs[1] {
                for w in 0..xs[2] {
                    for ic in 0..xs[3] {
                        let value =
                            values[(((n * xs[1] + h) * xs[2] + w) * xs[3] + ic) as usize] as f64;
                        for kh in 0..ks[0] {
                            for kw in 0..ks[1] {
                                let oh = h * options.strides[0] - options.padding[0][0]
                                    + kh * options.dilation[0];
                                let ow = w * options.strides[1] - options.padding[1][0]
                                    + kw * options.dilation[1];
                                if !(0..ys[1]).contains(&oh) || !(0..ys[2]).contains(&ow) {
                                    continue;
                                }
                                for oc in 0..ks[2] {
                                    let wi =
                                        (((kh * ks[1] + kw) * ks[2] + oc) * ks[3] + ic) as usize;
                                    let yi =
                                        (((n * ys[1] + oh) * ys[2] + ow) * ys[3] + oc) as usize;
                                    expected[yi] += value * weights[wi] as f64;
                                }
                            }
                        }
                    }
                }
            }
        }
        let actual = g
            .compile(&client, &y)
            .unwrap()
            .run(&[&values, &weights])
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(&expected) {
            assert!((a as f64 - e).abs() < 1e-5, "{a} != {e}, {options:?}");
        }
    }
}
