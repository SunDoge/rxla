use rxla_core::{Client, Conv2dOptions, Graph, Pool2dOptions, Tensor};

#[test]
fn upsample_shape_validation() {
    let g = Graph::default();
    let x = g.input(&[2, 3, 4, 5]).unwrap();
    assert_eq!(x.upsample_nearest2d([2, 3]).unwrap().shape(), [2, 6, 12, 5]);
    for scales in [[0, 1], [1, -1], [i64::MAX, 1]] {
        assert!(x.upsample_nearest2d(scales).is_err());
    }
    assert!(
        g.input(&[3, 4, 5])
            .unwrap()
            .upsample_nearest2d([2, 2])
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_integer_upsampling_matches_pixel_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [[2, 2, 3, 2], [1, 1, 1, 3], [1, 0, 2, 1]] {
        let values: Vec<f32> = (0..shape.iter().product::<i64>())
            .map(|i| i as f32 - 5.)
            .collect();
        for scales in [[1, 1], [2, 3], [3, 1]] {
            let g = Graph::default();
            let input = g.input(&shape).unwrap();
            let output = input.upsample_nearest2d(scales).unwrap();
            let mut expected = Vec::new();
            for n in 0..shape[0] {
                for h in 0..shape[1] * scales[0] {
                    for w in 0..shape[2] * scales[1] {
                        for c in 0..shape[3] {
                            let i = (((n * shape[1] + h / scales[0]) * shape[2] + w / scales[1])
                                * shape[3]
                                + c) as usize;
                            expected.push(values[i]);
                        }
                    }
                }
            }
            assert_eq!(
                g.compile(&client, &output)
                    .unwrap()
                    .run(&[&values])
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_convolution_pool_upsample_channel_merge() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[1, 4, 4, 1]).unwrap();
    let kernel = g.constant(&[1, 1, 1, 2], &[1., -1.]).unwrap();
    let fine = x
        .conv2d(&kernel, Conv2dOptions::default())
        .unwrap()
        .relu()
        .unwrap();
    let coarse = fine.max_pool2d(Pool2dOptions::default()).unwrap();
    let up = coarse.upsample_nearest2d([2, 2]).unwrap();
    let merged = Tensor::concatenate(&[fine, up], 3).unwrap();
    assert_eq!(merged.shape(), [1, 4, 4, 4]);
    let values: Vec<f32> = (0..16).map(|i| i as f32 - 8.).collect();
    let mut expected = Vec::new();
    for h in 0..4 {
        for w in 0..4 {
            let pixel = values[h * 4 + w];
            expected.extend([pixel.max(0.), (-pixel).max(0.)]);
            let mut pooled = [0f32; 2];
            for dh in 0..2 {
                for dw in 0..2 {
                    let value = values[(h / 2 * 2 + dh) * 4 + w / 2 * 2 + dw];
                    pooled[0] = pooled[0].max(value);
                    pooled[1] = pooled[1].max(-value);
                }
            }
            expected.extend(pooled);
        }
    }
    let executable = g.compile(&client, &merged).unwrap();
    assert_eq!(executable.run(&[&values]).unwrap(), expected);
    assert_eq!(executable.run(&[&[0.; 16]]).unwrap(), vec![0.; 64]);
}
