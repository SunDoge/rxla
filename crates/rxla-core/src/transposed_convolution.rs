use super::*;

impl Tensor {
    /// Scatter each input pixel through a spatial kernel: input `[N,H,W,Cin]`,
    /// weight `[Kh,Kw,Cout,Cin]`, result `[N,Oh,Ow,Cout]`. Bias is separate.
    /// Output size is `(input-1)*stride + (kernel-1)*dilation + 1 - padding_low - padding_high + output_padding`.
    /// Spatial input/output sizes must be positive. Groups and explicit
    /// output-shape overrides are unsupported.
    /// Reverse-mode supports input and kernel gradients, including output_padding.
    /// Generated convolution adjoints also support reverse-mode composition.
    pub fn conv_transpose2d(&self, kernel: &Self, options: ConvTranspose2dOptions) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &kernel.graph().0) {
            return Err(err("cross-graph transposed convolution operands"));
        }
        if self.shape.len() != 4 || kernel.shape.len() != 4 {
            return Err(err("conv_transpose2d requires NHWC input and HWOI kernel"));
        }
        let x = &self.shape;
        let k = &kernel.shape;
        if x[1..].iter().any(|&v| v <= 0) || k.iter().any(|&v| v <= 0) || x[3] != k[3] {
            return Err(err("invalid transposed convolution channels or dimensions"));
        }
        let mut output = vec![x[0]];
        for axis in 0..2 {
            let stride = options.strides[axis];
            let dilation = options.dilation[axis];
            let [low, high] = options.padding[axis];
            let extra = options.output_padding[axis];
            if stride <= 0 || dilation <= 0 || low < 0 || high < 0 || extra < 0 || extra >= stride {
                return Err(err(
                    "invalid transposed convolution stride/dilation/padding",
                ));
            }
            (x[axis + 1] - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| err("transposed convolution dilated input overflow"))?;
            let effective = (k[axis] - 1)
                .checked_mul(dilation)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| err("transposed convolution effective kernel overflow"))?;
            // Check the high-side HLO padding as well as the public output shape.
            (effective - 1)
                .checked_sub(high)
                .and_then(|v| v.checked_add(extra))
                .ok_or_else(|| err("transposed convolution HLO padding overflow"))?;
            let size = (x[axis + 1] as i128 - 1) * stride as i128 + effective as i128
                - low as i128
                - high as i128
                + extra as i128;
            let size =
                i64::try_from(size).map_err(|_| err("transposed convolution output overflow"))?;
            if size <= 0 {
                return Err(err(
                    "transposed convolution output must have positive spatial dimensions",
                ));
            }
            output.push(size);
        }
        output.push(k[2]);
        let mut output_type = self.ty();
        output_type.dims = output;
        self.graph().node_typed(
            Op::ConvTranspose2d(options),
            vec![self.node_id(), kernel.node_id()],
            output_type,
        )
    }
}
