use super::*;

impl Tensor {
    /// Repeat each NHWC pixel by positive integer factors `[height, width]`.
    /// Output `(n, h, w, c)` reads input `(n, h / scale_h, w / scale_w, c)`.
    /// This is integer nearest-neighbor upsampling, not general image resize:
    /// no half-pixel/align-corners convention, antialiasing or interpolation.
    pub fn upsample_nearest2d(&self, scales: [i64; 2]) -> Result<Self> {
        if self.shape.len() != 4 || scales.iter().any(|&scale| scale <= 0) {
            return Err(err(
                "upsample_nearest2d requires NHWC input and positive integer scales",
            ));
        }
        let [n, h, w, c] = <[i64; 4]>::try_from(self.shape.as_ref()).unwrap();
        let oh = h
            .checked_mul(scales[0])
            .ok_or_else(|| err("upsampled height overflow"))?;
        let ow = w
            .checked_mul(scales[1])
            .ok_or_else(|| err("upsampled width overflow"))?;
        let shape = [n, oh, ow, c];
        elements(&shape)?;
        if scales == [1, 1] {
            return Ok(self.clone());
        }
        self.reshape(&[n, h, 1, w, 1, c])?
            .broadcast_to(&[n, h, scales[0], w, scales[1], c])?
            .reshape(&shape)
    }
}
