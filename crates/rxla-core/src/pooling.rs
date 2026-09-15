use super::*;

impl Tensor {
    /// Max pooling over H/W, independently for every batch/channel. Padding is
    /// negative infinity, including windows entirely outside the input. Output
    /// sizes use floor division; no ceil mode, dilation or argmax indices.
    /// Max-pool training is outside the current Pliron training surface.
    pub fn max_pool2d(&self, options: Pool2dOptions) -> Result<Self> {
        let output = self.pool2d_shape(options)?;
        self.graph()
            .node(Op::MaxPool2d(options), vec![self.node_id()], &output)
    }
    /// NHWC average pooling with zero-valued padding and floor output sizes.
    /// With `count_include_pad`, divide by the full window area. Otherwise divide
    /// by the number of input elements in each spatial window; entirely padded
    /// windows produce NaN (0/0). No ceil mode, dilation or divisor override.
    /// Sums and divisors are F32, including rounding of large window counts.
    /// Reverse-mode accumulates overlapping windows and excludes padding from
    /// input gradients. The convolution adjoints support higher-order composition.
    /// Gradients of
    /// entirely padded, NaN-producing windows are undefined.
    pub fn avg_pool2d(&self, options: Pool2dOptions, count_include_pad: bool) -> Result<Self> {
        let output = self.pool2d_shape(options)?;
        let sum = self
            .graph()
            .node(Op::SumPool2d(options), vec![self.node_id()], &output)?;
        if count_include_pad {
            return sum.mul_scalar(1. / (options.window[0] as f32 * options.window[1] as f32));
        }
        // Counts depend only on spatial shape. Do not materialize one mask per
        // batch/channel: XLA can fold this broadcast + reduce-window subgraph.
        let ones = self.graph().constant(&[], &[1.])?.broadcast_to(&[
            1,
            self.shape[1],
            self.shape[2],
            1,
        ])?;
        let counts = self.graph().node(
            Op::SumPool2d(options),
            vec![ones.node_id()],
            &[1, output[1], output[2], 1],
        )?;
        sum.div(&counts.broadcast_to(&output)?)
    }
    pub(super) fn sum_pool2d_gradient(
        &self,
        input_shape: &[i64],
        options: Pool2dOptions,
    ) -> Result<Self> {
        if input_shape.contains(&0) || self.shape().contains(&0) {
            return self.graph().constant(&[], &[0.])?.broadcast_to(input_shape);
        }
        // Treat channels as independent batches, not a dense C-by-C identity
        // kernel. A scalar broadcast creates the all-ones spatial kernel.
        let batch = input_shape[0]
            .checked_mul(input_shape[3])
            .ok_or_else(|| err("pool gradient batch/channel size overflow"))?;
        let channels_first =
            self.transpose(&[0, 3, 1, 2])?
                .reshape(&[batch, self.shape[1], self.shape[2], 1])?;
        let kernel = self.graph().constant(&[], &[1.])?.broadcast_to(&[
            options.window[0],
            options.window[1],
            1,
            1,
        ])?;
        let mut output_padding = [0; 2];
        for (axis, extra) in output_padding.iter_mut().enumerate() {
            let padded = i128::from(input_shape[axis + 1])
                + i128::from(options.padding[axis][0])
                + i128::from(options.padding[axis][1]);
            *extra = ((padded - i128::from(options.window[axis]))
                % i128::from(options.strides[axis])) as i64;
        }
        channels_first
            .conv_transpose2d(
                &kernel,
                ConvTranspose2dOptions {
                    strides: options.strides,
                    padding: options.padding,
                    output_padding,
                    ..Default::default()
                },
            )?
            .reshape(&[
                input_shape[0],
                input_shape[3],
                input_shape[1],
                input_shape[2],
            ])?
            .transpose(&[0, 2, 3, 1])
    }

    fn pool2d_shape(&self, options: Pool2dOptions) -> Result<Vec<i64>> {
        if self.shape.len() != 4 {
            return Err(err("pool2d requires rank-four NHWC input"));
        }
        let mut output = self.shape.to_vec();
        for axis in 0..2 {
            let window = options.window[axis];
            let stride = options.strides[axis];
            let [low, high] = options.padding[axis];
            if window <= 0 || stride <= 0 || low < 0 || high < 0 {
                return Err(err(
                    "pool2d requires positive windows/strides and nonnegative padding",
                ));
            }
            let padded = self.shape[axis + 1]
                .checked_add(low)
                .and_then(|v| v.checked_add(high))
                .ok_or_else(|| err("pool2d padded size overflow"))?;
            output[axis + 1] = if padded < window {
                0
            } else {
                (padded - window) / stride + 1
            };
        }
        Ok(output)
    }
}
