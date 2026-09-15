use image::{RgbImage, imageops::FilterType};
use rayon::prelude::*;

use super::{
    Result,
    error::{EmptyImageSnafu, InvalidTargetSnafu},
};

const MEAN: f32 = 0.5;
const SCALE: f32 = 2.0 / 255.0;

/// NCHW detector input and the scale needed to map boxes back to the source.
#[derive(Debug)]
pub struct DetectorImage {
    pub nchw: Vec<f32>,
    pub shape: [usize; 4],
    pub scale_y: f32,
    pub scale_x: f32,
}

impl DetectorImage {
    /// Resize the page to a multiple of 32 and normalize directly into NCHW.
    pub fn prepare(source: &RgbImage, limit: u32) -> Result<Self> {
        snafu::ensure!(source.width() > 0 && source.height() > 0, EmptyImageSnafu);
        snafu::ensure!(limit >= 32, InvalidTargetSnafu);
        let ratio = (limit as f32 / source.width().max(source.height()) as f32).min(1.0);
        let aligned = |size: u32| ((size as f32 * ratio).round() as u32).max(1).div_ceil(32) * 32;
        let width = aligned(source.width());
        let height = aligned(source.height());
        let resized = image::imageops::resize(source, width, height, FilterType::Triangle);
        let nchw = rgb_to_nchw(&resized, width as usize);
        Ok(Self {
            nchw,
            shape: [1, 3, height as usize, width as usize],
            scale_y: height as f32 / source.height() as f32,
            scale_x: width as f32 / source.width() as f32,
        })
    }
}

/// Contiguous, zero-padded `[N, 3, 48, width]` recognition input.
#[derive(Debug)]
pub struct RecognitionBatch {
    pub nchw: Vec<f32>,
    pub shape: [usize; 4],
    pub valid_ratios: Vec<f32>,
}

/// PP-OCRv6 recognition resize/normalize policy.
#[derive(Clone, Copy, Debug)]
pub struct RecognitionPreprocessor {
    width: u32,
}

impl Default for RecognitionPreprocessor {
    fn default() -> Self {
        Self { width: 320 }
    }
}

impl RecognitionPreprocessor {
    pub fn new(width: u32) -> Result<Self> {
        snafu::ensure!(width > 0, InvalidTargetSnafu);
        Ok(Self { width })
    }

    /// Prepare independent crops in parallel. Each worker writes one disjoint
    /// batch slot, so there is no lock or intermediate HWC float allocation.
    pub fn prepare(&self, crops: &[RgbImage]) -> Result<RecognitionBatch> {
        snafu::ensure!(
            crops
                .iter()
                .all(|crop| crop.width() > 0 && crop.height() > 0),
            EmptyImageSnafu
        );
        let slot = 3 * 48 * self.width as usize;
        let prepared: Vec<_> = crops
            .par_iter()
            .map(|crop| self.prepare_one(crop))
            .collect();
        let mut nchw = vec![0.0; slot * crops.len()];
        let mut valid_ratios = Vec::with_capacity(crops.len());
        for (index, (values, ratio)) in prepared.into_iter().enumerate() {
            nchw[index * slot..index * slot + values.len()].copy_from_slice(&values);
            valid_ratios.push(ratio);
        }
        Ok(RecognitionBatch {
            nchw,
            shape: [crops.len(), 3, 48, self.width as usize],
            valid_ratios,
        })
    }

    fn prepare_one(&self, crop: &RgbImage) -> (Vec<f32>, f32) {
        let resized_width = ((48.0 * crop.width() as f32 / crop.height() as f32).ceil() as u32)
            .clamp(1, self.width);
        let resized = image::imageops::resize(crop, resized_width, 48, FilterType::Triangle);
        (
            rgb_to_nchw(&resized, self.width as usize),
            resized_width as f32 / self.width as f32,
        )
    }
}

fn rgb_to_nchw(image: &RgbImage, row_width: usize) -> Vec<f32> {
    let plane = image.height() as usize * row_width;
    let mut output = vec![0.0; plane * 3];
    for (row, pixels) in image
        .as_raw()
        .chunks_exact(image.width() as usize * 3)
        .enumerate()
    {
        let base = row * row_width;
        for (column, rgb) in pixels.as_chunks::<3>().0.iter().enumerate() {
            let index = base + column;
            output[index] = rgb[0] as f32 * SCALE - MEAN * 2.0;
            output[plane + index] = rgb[1] as f32 * SCALE - MEAN * 2.0;
            output[2 * plane + index] = rgb[2] as f32 * SCALE - MEAN * 2.0;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognition_is_planar_normalized_and_padded() {
        let crop = RgbImage::from_pixel(2, 1, image::Rgb([0, 128, 255]));
        let batch = RecognitionPreprocessor::new(128)
            .unwrap()
            .prepare(&[crop])
            .unwrap();
        assert_eq!(batch.shape, [1, 3, 48, 128]);
        assert_eq!(batch.valid_ratios, vec![0.75]);
        let plane = 48 * 128;
        assert_eq!(batch.nchw[0], -1.0);
        assert!((batch.nchw[plane] - 1.0 / 255.0).abs() < 1e-6);
        assert_eq!(batch.nchw[2 * plane], 1.0);
        assert_eq!(batch.nchw[127], 0.0);
    }
}
