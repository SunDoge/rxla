use super::*;

impl Tensor {
    /// Input `[N,H,W,Cin]`, kernel `[Kh,Kw,Cin/groups,Cout]`, output `[N,Oh,Ow,Cout]`.
    /// Output channels are contiguous by group. Depthwise convolution uses
    /// groups=Cin and Cout=Cin*multiplier. Bias is a separate broadcast/add.
    /// First-order reverse-mode supports both inputs and kernels, including
    /// groups, stride, dilation and asymmetric padding. Generated adjoints also
    /// have reverse rules; this does not guarantee high-order numerical stability.
    pub fn conv2d(&self, kernel: &Self, options: Conv2dOptions) -> Result<Self> {
        self.conv2d_layout(kernel, options, false)
    }

    /// Input `[N,H,W,Cin]`, checkpoint-native kernel
    /// `[Cout,Cin/groups,Kh,Kw]`, output `[N,Oh,Ow,Cout]`.
    ///
    /// This records OIHW directly in StableHLO convolution dimension numbers;
    /// it does not transpose weights at runtime. Prefer this for PyTorch and
    /// safetensors checkpoints. Reverse mode converts only its symbolic gradient
    /// path to the existing HWIO adjoint representation.
    pub fn conv2d_oihw(&self, kernel: &Self, options: Conv2dOptions) -> Result<Self> {
        self.conv2d_layout(kernel, options, true)
    }

    fn conv2d_layout(&self, kernel: &Self, options: Conv2dOptions, oihw: bool) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &kernel.graph().0) {
            return Err(err("cross-graph convolution operands"));
        }
        if self.shape.len() != 4 || kernel.shape.len() != 4 {
            return Err(err(
                "conv2d requires rank-four NHWC input and rank-four kernel",
            ));
        }
        let x = &self.shape;
        let k = &kernel.shape;
        let (kh, kw, kernel_input, kernel_output) = if oihw {
            (k[2], k[3], k[1], k[0])
        } else {
            (k[0], k[1], k[2], k[3])
        };
        if options.groups <= 0
            || x[1..].iter().any(|&d| d < 0)
            || x[3] <= 0
            || k.iter().any(|&d| d <= 0)
            || x[3] % options.groups != 0
            || x[3] / options.groups != kernel_input
            || kernel_output % options.groups != 0
        {
            return Err(err(
                "conv2d channels/kernel dimensions/groups are incompatible",
            ));
        }
        let mut output = vec![x[0]];
        for axis in 0..2 {
            let stride = options.strides[axis];
            let dilation = options.dilation[axis];
            let [low, high] = options.padding[axis];
            if stride <= 0 || dilation <= 0 || low < 0 || high < 0 {
                return Err(err(
                    "conv2d requires positive stride/dilation and nonnegative padding",
                ));
            }
            let kernel_size = [kh, kw][axis];
            let effective = (kernel_size - 1)
                .checked_mul(dilation)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| err("conv2d effective kernel size overflow"))?;
            let padded = x[axis + 1]
                .checked_add(low)
                .and_then(|v| v.checked_add(high))
                .ok_or_else(|| err("conv2d padded input size overflow"))?;
            output.push(if padded < effective {
                0
            } else {
                (padded - effective) / stride + 1
            });
        }
        output.push(kernel_output);
        let mut output_type = self.ty();
        output_type.dims = output;
        self.graph().node_typed(
            if oihw {
                Op::Conv2dOihw(options)
            } else {
                Op::Conv2d(options)
            },
            vec![self.node_id(), kernel.node_id()],
            output_type,
        )
    }
}
