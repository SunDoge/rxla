use super::*;

fn validate_resize(input: &Tensor, size: [i64; 2], operation: &str) -> Result<[i64; 4]> {
    if input.shape.len() != 4
        || input.shape[1] <= 0
        || input.shape[2] <= 0
        || size.iter().any(|&dimension| dimension <= 0)
    {
        return Err(err(format!(
            "{operation} requires nonempty NHWC spatial dimensions and a positive output size"
        )));
    }
    Ok([
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    ])
}

fn nearest_indices(input: i64, output: i64) -> Result<Vec<i32>> {
    (0..output)
        .map(|coordinate| {
            // `coordinate * input` cannot overflow i128 for valid i64 shapes.
            i32::try_from((i128::from(coordinate) * i128::from(input) / i128::from(output)) as i64)
                .map_err(|_| err("resize index exceeds I32 range"))
        })
        .collect()
}

fn linear_samples(input: i64, output: i64) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>)> {
    if input > i64::from(i32::MAX) {
        return Err(err("resize input dimension exceeds I32 range"));
    }
    let scale = input as f64 / output as f64;
    let mut lower = Vec::with_capacity(output as usize);
    let mut upper = Vec::with_capacity(output as usize);
    let mut weight = Vec::with_capacity(output as usize);
    for coordinate in 0..output {
        // Half-pixel centers, with edge-value padding. Clamping before deriving
        // the fraction makes a one-pixel input an exact broadcast.
        let source = ((coordinate as f64 + 0.5) * scale - 0.5).clamp(0.0, input as f64 - 1.0);
        let low = source.floor() as i64;
        let high = (low + 1).min(input - 1);
        lower.push(low as i32);
        upper.push(high as i32);
        weight.push((source - low as f64) as f32);
    }
    Ok((lower, upper, weight))
}

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

    /// Resize an NHWC tensor to an arbitrary positive spatial size using the
    /// asymmetric nearest-neighbor convention: output `(y, x)` reads input
    /// `(floor(y * H / OH), floor(x * W / OW))`. Batch and channel dimensions
    /// are preserved. The operation is differentiable with respect to selected
    /// input pixels, but not with respect to the static output size.
    pub fn resize_nearest2d(&self, size: [i64; 2]) -> Result<Self> {
        let [_, height, width, _] = validate_resize(self, size, "resize_nearest2d")?;
        if size == [height, width] {
            return Ok(self.clone());
        }
        let rows = self
            .graph()
            .constant_i32(&[size[0]], &nearest_indices(height, size[0])?)?;
        let columns = self
            .graph()
            .constant_i32(&[size[1]], &nearest_indices(width, size[1])?)?;
        self.take(&rows, 1)?.take(&columns, 2)
    }

    /// Resize an F32 NHWC tensor using separable bilinear interpolation and
    /// half-pixel centers. Coordinates outside the source extent use the nearest
    /// edge value. This matches the common ML preprocessing convention with
    /// `align_corners=false`; antialias filtering is not applied when downsampling.
    /// All gathers and arithmetic remain in the graph for backend fusion.
    pub fn resize_bilinear2d(&self, size: [i64; 2]) -> Result<Self> {
        let [batch, height, width, channels] = validate_resize(self, size, "resize_bilinear2d")?;
        if self.dtype() != DType::F32 {
            return Err(err("resize_bilinear2d requires F32 input; cast explicitly"));
        }
        if size == [height, width] {
            return Ok(self.clone());
        }
        let (row0, row1, row_weight) = linear_samples(height, size[0])?;
        let (column0, column1, column_weight) = linear_samples(width, size[1])?;
        let graph = self.graph();
        let row0 = graph.constant_i32(&[size[0]], &row0)?;
        let row1 = graph.constant_i32(&[size[0]], &row1)?;
        let row_weight = graph
            .constant(&[size[0]], &row_weight)?
            .reshape(&[1, size[0], 1, 1])?
            .broadcast_to(&[batch, size[0], width, channels])?;
        let top = self.take(&row0, 1)?;
        let bottom = self.take(&row1, 1)?;
        let vertical = top.add(&bottom.sub(&top)?.mul(&row_weight)?)?;

        let column0 = graph.constant_i32(&[size[1]], &column0)?;
        let column1 = graph.constant_i32(&[size[1]], &column1)?;
        let column_weight = graph
            .constant(&[size[1]], &column_weight)?
            .reshape(&[1, 1, size[1], 1])?
            .broadcast_to(&[batch, size[0], size[1], channels])?;
        let left = vertical.take(&column0, 2)?;
        let right = vertical.take(&column1, 2)?;
        left.add(&right.sub(&left)?.mul(&column_weight)?)
    }

    /// Apply per-channel `(x - mean) / stddev` normalization to an F32 NHWC
    /// tensor. Statistics are graph constants and may therefore fuse with cast,
    /// resize and the model's first operation. Every standard deviation must be
    /// finite and strictly positive; means must be finite.
    pub fn normalize_nhwc(&self, mean: &[f32], stddev: &[f32]) -> Result<Self> {
        if self.dtype() != DType::F32 || self.shape.len() != 4 {
            return Err(err("normalize_nhwc requires F32 NHWC input"));
        }
        let channels = usize::try_from(self.shape[3]).map_err(|_| err("invalid channel count"))?;
        if mean.len() != channels
            || stddev.len() != channels
            || mean.iter().any(|value| !value.is_finite())
            || stddev
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(err(
                "normalize_nhwc requires one finite mean and positive finite stddev per channel",
            ));
        }
        let mean = self
            .graph()
            .constant(&[self.shape[3]], mean)?
            .broadcast_to(self.shape())?;
        let inverse_stddev = stddev.iter().map(|value| value.recip()).collect::<Vec<_>>();
        let inverse_stddev = self
            .graph()
            .constant(&[self.shape[3]], &inverse_stddev)?
            .broadcast_to(self.shape())?;
        self.sub(&mean)?.mul(&inverse_stddev)
    }
}
