//! Graph-native NHWC image transformations inspired by dm-pix. These methods
//! compose ordinary Tensor operations so a backend can fuse a complete
//! preprocessing/augmentation chain with adjacent model computation.
use super::*;

fn require_f32_nhwc(image: &Tensor, operation: &str) -> Result<[i64; 4]> {
    if image.dtype() != DType::F32 || image.shape().len() != 4 {
        return Err(err(format!("{operation} requires F32 NHWC input")));
    }
    Ok([
        image.shape()[0],
        image.shape()[1],
        image.shape()[2],
        image.shape()[3],
    ])
}

impl Tensor {
    /// Add a finite value to every pixel without clipping. This follows dm-pix
    /// chaining semantics: callers decide whether and when values are clamped.
    pub fn adjust_brightness(&self, delta: f32) -> Result<Self> {
        require_f32_nhwc(self, "adjust_brightness")?;
        if !delta.is_finite() {
            return Err(err("brightness delta must be finite"));
        }
        self.add_scalar(delta)
    }

    /// Scale every channel around its per-image spatial mean. Batch members and
    /// channels retain independent means; output values are not clipped.
    pub fn adjust_contrast(&self, factor: f32) -> Result<Self> {
        require_f32_nhwc(self, "adjust_contrast")?;
        if !factor.is_finite() {
            return Err(err("contrast factor must be finite"));
        }
        let mean = self.mean(&[1, 2], true)?;
        self.sub(&mean.broadcast_to(self.shape())?)?
            .mul_scalar(factor)?
            .add(&mean.broadcast_to(self.shape())?)
    }

    /// Flip NHWC images horizontally. This is an IR reverse operation and does
    /// not materialize a host-side copy.
    pub fn flip_left_right(&self) -> Result<Self> {
        require_f32_nhwc(self, "flip_left_right")?;
        self.flip(&[2])
    }

    /// Flip NHWC images vertically without materializing a host-side copy.
    pub fn flip_up_down(&self) -> Result<Self> {
        require_f32_nhwc(self, "flip_up_down")?;
        self.flip(&[1])
    }

    /// Center-crop spatial dimensions, treating a target larger than the input
    /// as a no-op for that dimension. For an odd discarded extent, the
    /// bottom/right side loses the extra pixel, matching dm-pix `center_crop`.
    pub fn center_crop(&self, size: [i64; 2]) -> Result<Self> {
        let [batch, height, width, channels] = require_f32_nhwc(self, "center_crop")?;
        if size.iter().any(|&dimension| dimension <= 0) {
            return Err(err("center crop size must be positive"));
        }
        let output_height = size[0].min(height);
        let output_width = size[1].min(width);
        let top = height / 2 - output_height / 2;
        let left = width / 2 - output_width / 2;
        self.slice(
            &[0, top, left, 0],
            &[batch, top + output_height, left + output_width, channels],
            &[1; 4],
        )
    }

    /// Center-crop and/or zero-pad to an exact positive spatial size. Odd
    /// padding puts the extra pixel on the bottom/right, as dm-pix does.
    pub fn resize_with_crop_or_pad(&self, size: [i64; 2]) -> Result<Self> {
        require_f32_nhwc(self, "resize_with_crop_or_pad")?;
        let cropped = self.center_crop(size)?;
        let height = cropped.shape()[1];
        let width = cropped.shape()[2];
        let vertical = size[0]
            .checked_sub(height)
            .ok_or_else(|| err("invalid crop/pad height"))?;
        let horizontal = size[1]
            .checked_sub(width)
            .ok_or_else(|| err("invalid crop/pad width"))?;
        let top = vertical / 2;
        let left = horizontal / 2;
        cropped.pad(
            &[
                [0, 0],
                [top, vertical - top],
                [left, horizontal - left],
                [0, 0],
            ],
            0.0,
        )
    }

    /// Convert three-channel NHWC RGB to one-channel grayscale using a selected
    /// luma standard. Supported names and coefficients match dm-pix.
    pub fn rgb_to_grayscale(&self, luma_standard: &str) -> Result<Self> {
        let [batch, height, width, channels] = require_f32_nhwc(self, "rgb_to_grayscale")?;
        if channels != 3 {
            return Err(err("rgb_to_grayscale requires exactly three channels"));
        }
        let weights = match luma_standard {
            "rec601" => [0.2989, 0.5870, 0.1140],
            "rec709" => [0.2126, 0.7152, 0.0722],
            "bt2001" => [0.2627, 0.6780, 0.0593],
            _ => return Err(err("unsupported grayscale luma standard")),
        };
        let weights = self
            .graph()
            .constant(&[1, 1, 1, 3], &weights)?
            .broadcast_to(&[batch, height, width, 3])?;
        self.mul(&weights)?.sum(&[3], true)
    }

    /// Invert values greater than or equal to `threshold`, leaving lower values
    /// unchanged. Inputs and outputs are not otherwise clipped to `[0, 1]`.
    pub fn solarize(&self, threshold: f32) -> Result<Self> {
        require_f32_nhwc(self, "solarize")?;
        if !threshold.is_finite() {
            return Err(err("solarize threshold must be finite"));
        }
        let threshold = self.scalar(threshold)?.broadcast_to(self.shape())?;
        self.lt_mask(&threshold)?
            .select(self, &self.neg()?.add_scalar(1.0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chained_augmentation_lowers_without_a_runtime()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let image = Tensor::from_slice([1, 8, 10, 3], DType::U8, vec![0u8; 8 * 10 * 3])?;
        let output = image
            .cast(DType::F32)?
            .mul_scalar(1.0 / 255.0)?
            .resize_bilinear2d([7, 9])?
            .center_crop([6, 6])?
            .flip_left_right()?
            .adjust_brightness(0.05)?
            .adjust_contrast(1.1)?
            .normalize_nhwc(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225])?;
        let program = output.program()?;
        let lowered = program.lowered_program();
        assert_eq!(lowered.format(), "mlir");
        assert_eq!(lowered.input_count(), 1);
        assert_eq!(
            lowered.input_spec(0).ok_or("missing input spec")?.dtype,
            DType::U8
        );
        assert_eq!(
            lowered.output_spec(0).ok_or("missing output spec")?.shape,
            [1, 6, 6, 3]
        );
        let stablehlo = std::str::from_utf8(lowered.code())?;
        assert!(stablehlo.contains("stablehlo.gather"));
        assert!(stablehlo.contains("stablehlo.reduce"));
        assert!(stablehlo.contains("stablehlo.reverse"));
        Ok(())
    }
}
