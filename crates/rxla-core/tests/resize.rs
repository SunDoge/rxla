use rxla_core::{Client, Conv2dOptions, DType, Pool2dOptions, Runtime, Tensor, Tracer};

#[test]
fn upsample_shape_validation() {
    let g = Tracer::default();
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
fn image_resize_and_normalization_validate_shapes_and_dtypes()
-> Result<(), Box<dyn std::error::Error>> {
    let x = Tensor::from_slice([2, 3, 5, 3], DType::F32, vec![0.0; 2 * 3 * 5 * 3])?;
    assert_eq!(x.resize_nearest2d([7, 2])?.shape(), [2, 7, 2, 3]);
    assert_eq!(x.resize_bilinear2d([7, 2])?.shape(), [2, 7, 2, 3]);
    assert_eq!(x.normalize_nhwc(&[0.5; 3], &[0.25; 3])?.shape(), x.shape());
    assert!(x.resize_nearest2d([0, 2]).is_err());
    assert!(x.resize_bilinear2d([2, -1]).is_err());
    let hwc = Tensor::from_slice([3, 5, 3], DType::F32, [0.0; 45])?;
    assert!(hwc.resize_bilinear2d([2, 2]).is_err());
    assert!(x.normalize_nhwc(&[0.; 2], &[1.; 2]).is_err());
    assert!(x.normalize_nhwc(&[0.; 3], &[1., 0., 1.]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_arbitrary_resize_and_normalization_match_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    let shape = [1, 2, 3, 2];
    let values = [0., 10., 2., 12., 4., 14., 6., 16., 8., 18., 10., 20.];
    let input = Tensor::from_slice(shape, DType::F32, values)?;
    let nearest = input.resize_nearest2d([3, 2])?;
    let mut runtime = Runtime::new(client);
    assert_eq!(
        nearest.eval(&mut runtime)?.to_vec::<f32>()?,
        vec![0., 10., 2., 12., 0., 10., 2., 12., 6., 16., 8., 18.]
    );

    let bilinear = input
        .resize_bilinear2d([3, 2])?
        .normalize_nhwc(&[1., 10.], &[2., 5.])?;
    let actual = bilinear.eval(&mut runtime)?.to_vec::<f32>()?;
    let expected = [
        -0.25, 0.1, 1.25, 0.7, 1.25, 0.7, 2.75, 1.3, 2.75, 1.3, 4.25, 1.9,
    ];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
    Ok(())
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
            let g = Tracer::default();
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
    let g = Tracer::default();
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
