use super::*;

impl Tensor {
    /// Max pooling over H/W, independently for every batch/channel. Padding is
    /// negative infinity, including windows entirely outside the input. Output
    /// sizes use floor division; no ceil mode, dilation or argmax indices.
    /// Reverse mode uses an explicit offset-wise scatter and divides ties evenly.
    pub fn max_pool2d(&self, options: Pool2dOptions) -> Result<Self> {
        let output = self.pool2d_shape(options)?;
        self.graph()
            .node_typed(Op::MaxPool2d(options), vec![self.node_id()], output)
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
        let sum = self.graph().node_typed(
            Op::SumPool2d(options),
            vec![self.node_id()],
            output.clone(),
        )?;
        if count_include_pad {
            return sum.mul_scalar(1. / (options.window[0] as f32 * options.window[1] as f32));
        }
        // Counts depend only on spatial shape. Do not materialize one mask per
        // batch/channel: XLA can fold this broadcast + reduce-window subgraph.
        let ones = self.graph().constant(&[], &[1.])?.broadcast_to([
            1,
            self.shape[1],
            self.shape[2],
            1,
        ])?;
        let counts = self.graph().node(
            Op::SumPool2d(options),
            vec![ones.node_id()],
            &[1, output.dims[1], output.dims[2], 1],
        )?;
        sum.div(&counts.broadcast_as(&sum)?)
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
        let channels_first = Rearrange::new("n h w c -> (n c) h w 1").apply(self)?;
        let kernel = self.graph().constant(&[], &[1.])?.broadcast_to([
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
        let scattered = channels_first.conv_transpose2d(
            &kernel,
            ConvTranspose2dOptions {
                strides: options.strides,
                padding: options.padding,
                output_padding,
                ..Default::default()
            },
        )?;
        Rearrange::new("(n c) h w 1 -> n h w c")
            .axis("n", input_shape[0])
            .apply(&scattered)
    }

    pub(super) fn max_pool2d_gradient(
        &self,
        input: &Tensor,
        options: Pool2dOptions,
    ) -> Result<Self> {
        if input.shape().contains(&0) || self.shape().contains(&0) {
            return input.scalar(0.0)?.broadcast_to(input.shape());
        }
        let pooled = input.max_pool2d(options)?;
        let padded = input.pad(
            &[[0, 0], options.padding[0], options.padding[1], [0, 0]],
            f32::NEG_INFINITY,
        )?;
        let mut masks = Vec::with_capacity((options.window[0] * options.window[1]) as usize);
        let limits = [input.shape()[0], 0, 0, input.shape()[3]];
        for row in 0..options.window[0] {
            for column in 0..options.window[1] {
                let mut offset_limits = limits;
                offset_limits[1] = row + (pooled.shape()[1] - 1) * options.strides[0] + 1;
                offset_limits[2] = column + (pooled.shape()[2] - 1) * options.strides[1] + 1;
                let values = padded.slice(
                    &[0, row, column, 0],
                    &offset_limits,
                    &[1, options.strides[0], options.strides[1], 1],
                )?;
                masks.push(values.eq_mask(&pooled)?);
            }
        }
        let mut ties = pooled.scalar(0.0)?.broadcast_to(pooled.shape())?;
        for mask in &masks {
            ties = ties.add(mask)?;
        }
        let shared = self.div(&ties)?;
        let mut result = input.scalar(0.0)?.broadcast_to(input.shape())?;
        for (offset, mask) in masks.into_iter().enumerate() {
            let row = offset as i64 / options.window[1];
            let column = offset as i64 % options.window[1];
            let contribution = shared.mul(&mask)?;
            result = result.add(&contribution.scatter_pool_offset(
                input.shape(),
                options,
                [row, column],
            )?)?;
        }
        Ok(result)
    }

    fn scatter_pool_offset(
        &self,
        input_shape: &[i64],
        options: Pool2dOptions,
        offset: [i64; 2],
    ) -> Result<Self> {
        let channels_first = Rearrange::new("n h w c -> (n c) h w 1").apply(self)?;
        let rows = self.iota_i32(&[options.window[0], options.window[1], 1, 1], 0)?;
        let columns = self.iota_i32(&[options.window[0], options.window[1], 1, 1], 1)?;
        let row = rows
            .scalar_i32(offset[0] as i32)?
            .broadcast_to(rows.shape())?;
        let column = columns
            .scalar_i32(offset[1] as i32)?
            .broadcast_to(columns.shape())?;
        let kernel = rows.eq_mask(&row)?.mul(&columns.eq_mask(&column)?)?;
        let mut output_padding = [0; 2];
        for (axis, extra) in output_padding.iter_mut().enumerate() {
            let padded = i128::from(input_shape[axis + 1])
                + i128::from(options.padding[axis][0])
                + i128::from(options.padding[axis][1]);
            *extra = ((padded - i128::from(options.window[axis]))
                % i128::from(options.strides[axis])) as i64;
        }
        let scattered = channels_first.conv_transpose2d(
            &kernel,
            ConvTranspose2dOptions {
                strides: options.strides,
                padding: options.padding,
                output_padding,
                ..Default::default()
            },
        )?;
        Rearrange::new("(n c) h w 1 -> n h w c")
            .axis("n", input_shape[0])
            .apply(&scattered)
    }

    fn pool2d_shape(&self, options: Pool2dOptions) -> Result<TensorType> {
        if self.shape.len() != 4 {
            return Err(err("pool2d requires rank-four NHWC input"));
        }
        if self.shape[1..].iter().any(|&dimension| dimension < 0) {
            return Err(err(
                "pool2d currently supports dynamic batch dimensions only",
            ));
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
        let mut output_type = self.ty();
        output_type.dims = output;
        Ok(output_type)
    }
}
