use rxla_core::{Client, DType, Runtime, Tensor};

#[test]
fn image_augmentation_shapes_and_validation() -> Result<(), Box<dyn std::error::Error>> {
    let image = Tensor::from_slice([2, 5, 7, 3], DType::F32, vec![0.0; 2 * 5 * 7 * 3])?;
    assert_eq!(image.center_crop([3, 4])?.shape(), [2, 3, 4, 3]);
    assert_eq!(image.resize_with_crop_or_pad([8, 4])?.shape(), [2, 8, 4, 3]);
    assert_eq!(image.rgb_to_grayscale("rec601")?.shape(), [2, 5, 7, 1]);
    assert!(image.center_crop([0, 2]).is_err());
    assert!(image.rgb_to_grayscale("unknown").is_err());
    let rgba = Tensor::from_slice([1, 2, 2, 4], DType::F32, [0.0; 16])?;
    assert!(rgba.rgb_to_grayscale("rec601").is_err());
    assert!(image.adjust_brightness(f32::NAN).is_err());
    assert!(image.adjust_contrast(f32::INFINITY).is_err());
    Ok(())
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_chained_augmentations_match_scalar_reference() -> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    let values = [
        0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8,
    ];
    let input = Tensor::from_slice([1, 2, 3, 3], DType::F32, values)?;
    let output = input
        .resize_with_crop_or_pad([3, 3])?
        .flip_left_right()?
        .adjust_brightness(0.1)?
        .adjust_contrast(1.5)?
        .solarize(0.7)?
        .rgb_to_grayscale("rec601")?;
    let mut runtime = Runtime::new(client)?;
    let actual = output.eval(&mut runtime)?.to_vec::<f32>()?;

    let mut transformed = [0.0f32; 3 * 3 * 3];
    for y in 0..3 {
        for x in 0..3 {
            for channel in 0..3 {
                // Centering a two-row image in three rows adds the odd padding
                // pixel at the bottom, then the horizontal flip reverses width.
                transformed[(y * 3 + x) * 3 + channel] = if y < 2 {
                    values[(y * 3 + (2 - x)) * 3 + channel] + 0.1
                } else {
                    0.1
                };
            }
        }
    }
    for channel in 0..3 {
        let mean = (0..9)
            .map(|pixel| transformed[pixel * 3 + channel])
            .sum::<f32>()
            / 9.0;
        for pixel in 0..9 {
            let value = (transformed[pixel * 3 + channel] - mean) * 1.5 + mean;
            transformed[pixel * 3 + channel] = if value < 0.7 { value } else { 1.0 - value };
        }
    }
    let expected = (0..9)
        .map(|pixel| {
            transformed[pixel * 3] * 0.2989
                + transformed[pixel * 3 + 1] * 0.5870
                + transformed[pixel * 3 + 2] * 0.1140
        })
        .collect::<Vec<_>>();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
    }
    Ok(())
}
