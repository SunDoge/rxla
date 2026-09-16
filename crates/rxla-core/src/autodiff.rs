//! Experimental reverse-mode graph construction; no native execution or tracing
//! of arbitrary Rust code. Unsupported operations fail before gradient recording.
use super::*;

impl Tensor {
    /// Build F32 gradients of this scalar loss with respect to graph input
    /// tensors, in the requested order. Disconnected inputs receive zeros and
    /// duplicate requests return the same gradient. Parameters from StateGraph
    /// can be supplied through `Parameter::tensor`.
    ///
    /// Experimental: supports add/sub/mul/div, elementwise maximum/minimum,
    /// abs (zero gradient at zero), softplus and smooth unary ops,
    /// ReLU (zero gradient at zero), matmul, reshape, transpose, broadcast, sum/max
    /// slice/concatenate/constant padding, conv2d/conv_transpose2d, pooling and gather data gradients.
    /// Select routes gradients to value branches, not its numeric mask.
    /// Integer indices are nondifferentiable. Gather gradients accumulate repeated
    /// clamped indices; the scatter-add reverse rule gathers with those indices.
    /// Slice gradients insert zeros through native padding; their reverse rule
    /// extracts the original strided slice for higher-order composition.
    /// Dynamic slice/update differentiate data values at the clamped runtime
    /// positions; index construction paths remain nondifferentiable.
    /// Average pooling uses a differentiable transposed-convolution adjoint.
    /// Max pooling uses native select-and-scatter, also first-order only.
    /// Conv2d input/kernel adjoints support groups/stride/dilation and reverse rules
    /// for higher-order composition. Numerical stability is not universal.
    /// Any other operation on a backward path
    /// returns an error (even if independent of a requested input), unless behind
    /// an explicit `detach` boundary. Requires
    /// input/parameter leaves, not intermediate tensors. No implicit state,
    /// optimizer, RNG, nonfinite handling, or general training support is added.
    /// The derivatives are mathematical F32 expressions; singularities and
    /// roundoff are not regularized. Original graph handles remain valid.
    pub fn grad(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if !self.shape().is_empty() {
            return Err(err("grad requires a scalar loss"));
        }
        self.reverse_gradients(inputs, None)
    }

    /// Build a vector-Jacobian product for this arbitrarily shaped F32 output.
    /// The cotangent must have the same shape and graph. Returns one gradient
    /// per requested input/parameter leaf, with the same supported operation and
    /// disconnected-input rules as `grad`; it never materializes a Jacobian.
    ///
    /// The cotangent seeds this reverse sweep: its own construction is not
    /// differentiated as an extra loss factor. It may be a runtime input. The
    /// returned gradient graph still references that construction, so a later
    /// higher-order differentiation can see its dependencies; use `detach` if
    /// those dependencies should be stopped. No native execution occurs here.
    pub fn vjp(&self, inputs: &[Tensor], cotangent: &Tensor) -> Result<Vec<Tensor>> {
        if !Arc::ptr_eq(&self.graph().0, &cotangent.graph().0) {
            return Err(err("cross-graph cotangent"));
        }
        if self.shape != cotangent.shape {
            return Err(err(
                "cotangent must match output shape; broadcast explicitly",
            ));
        }
        self.reverse_gradients(inputs, Some(cotangent))
    }

    fn reverse_gradients(
        &self,
        inputs: &[Tensor],
        cotangent: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        if self.is_implicit_lazy()
            || inputs.iter().any(Tensor::is_implicit_lazy)
            || cotangent.is_some_and(Tensor::is_implicit_lazy)
        {
            return Err(err(
                "autodiff requires tensors from an active tracing session",
            ));
        }
        if self.dtype() != DType::F32
            || inputs.iter().any(|t| t.dtype() != DType::F32)
            || cotangent.is_some_and(|t| t.dtype() != DType::F32)
        {
            return Err(err("autodiff requires F32 outputs, inputs and cotangents"));
        }
        let nodes = self
            .graph()
            .0
            .lock()
            .map_err(|_| err("graph lock poisoned"))?
            .semantic_nodes()?;
        for input in inputs {
            if !Arc::ptr_eq(&self.graph().0, &input.graph().0) {
                return Err(err("cross-graph gradient input"));
            }
            if !matches!(
                nodes[input.node_id().index()].op,
                Op::Parameter(_) | Op::StateRead { .. }
            ) {
                return Err(err("grad inputs must be graph input/parameter leaves"));
            }
        }
        let mut live = vec![false; nodes.len()];
        live[self.node_id().index()] = true;
        for (i, node) in nodes.iter().enumerate().rev() {
            if !live[i] {
                continue;
            }
            match node.op {
                Op::StopGradient | Op::IndexToFloat | Op::Bf16ToFloat => continue,
                Op::Unary(Unary::Floor | Unary::Ceil | Unary::Round | Unary::RoundTiesEven) => {
                    continue;
                }
                Op::WithGradient => {
                    live[node.operands[1].index()] = true;
                    continue;
                }
                Op::WithElementwiseDerivative => {
                    for &input in node.operands[1..].iter().step_by(2) {
                        live[input.index()] = true;
                    }
                    continue;
                }
                Op::Select => {
                    // A numeric mask chooses a branch; it is not differentiated.
                    for &operand in &node.operands[1..] {
                        live[operand.index()] = true;
                    }
                    continue;
                }
                Op::Take { .. }
                | Op::TakeAlongAxis { .. }
                | Op::GatherGradient { .. }
                | Op::DynamicSlice => {
                    // Integer selection is nondifferentiable. Its forward
                    // dependencies remain live in the ordinary lowering pass.
                    live[node.operands[0].index()] = true;
                    continue;
                }
                Op::DynamicUpdateSlice => {
                    for &operand in &node.operands[..2] {
                        live[operand.index()] = true;
                    }
                    continue;
                }
                Op::Attention { .. } => {
                    for &operand in &node.operands {
                        live[operand.index()] = true;
                    }
                    continue;
                }
                Op::Parameter(_)
                | Op::StateInput { .. }
                | Op::StateRead { .. }
                | Op::StateWrite { .. }
                | Op::Relu
                | Op::Softplus
                | Op::Sigmoid
                | Op::SumPool2d(_)
                | Op::Cumsum { .. }
                | Op::MaxPool2d(_)
                | Op::Conv2d(_)
                | Op::Conv2dOihw(_)
                | Op::ConvTranspose2d(_)
                | Op::Conv2dInputGradient(_)
                | Op::Conv2dKernelGradient(_)
                | Op::ConstantF32(_)
                | Op::Reshape
                | Op::OptimizationBarrier
                | Op::Slice(_)
                | Op::SliceGradient(_)
                | Op::Pad(_)
                | Op::Concatenate { .. }
                | Op::Transpose { .. }
                | Op::Reverse { .. }
                | Op::Broadcast { .. }
                | Op::Matmul { .. }
                | Op::Reduce {
                    kind: Reduction::Sum | Reduction::Maximum,
                    ..
                }
                | Op::Binary(_)
                | Op::Unary(
                    Unary::Erf
                    | Unary::Sin
                    | Unary::Cos
                    | Unary::Abs
                    | Unary::Exp
                    | Unary::Log
                    | Unary::Log1p
                    | Unary::Expm1
                    | Unary::Neg
                    | Unary::Sqrt
                    | Unary::Rsqrt
                    | Unary::Tanh,
                ) => {}
                Op::Convert { .. } => {}
                _ => {
                    return Err(err(format!(
                        "grad unsupported operation at node {}: {:?}",
                        i + 1,
                        node.op
                    )));
                }
            }
            for &operand in &node.operands {
                live[operand.index()] = true;
            }
        }
        let value = |id: rxla_ir::SsaId| {
            Tensor::symbolic(
                self.graph().clone(),
                id,
                &nodes[id.index()].ty.dims,
                nodes[id.index()].ty.dtype,
            )
        };
        let mut gradients: Vec<Option<Tensor>> = vec![None; nodes.len()];
        gradients[self.node_id().index()] = Some(match cotangent {
            Some(seed) => seed.clone(),
            None => self.graph().constant(&[], &[1.])?,
        });
        for (i, node) in nodes.iter().enumerate().rev() {
            let Some(dy) = gradients[i].clone() else {
                continue;
            };
            let mut contributions = Vec::new();
            let mut add = |operand: usize, gradient: Tensor| {
                contributions.push((node.operands[operand].index(), gradient));
            };
            match &node.op {
                Op::Parameter(_)
                | Op::StateInput { .. }
                | Op::StateRead { .. }
                | Op::ConstantF32(_)
                | Op::StopGradient
                | Op::IndexToFloat
                | Op::Bf16ToFloat => {}
                Op::StateWrite { .. } => add(0, dy),
                Op::Convert { .. } => {
                    let source = value(node.operands[0]);
                    // Integer/byte inputs are data boundaries, not differentiable
                    // floating-point leaves. Keep floating casts differentiable,
                    // but do not manufacture an integer cotangent while sweeping
                    // past image/token preprocessing.
                    if matches!(source.dtype(), DType::F16 | DType::BF16 | DType::F32) {
                        add(0, dy.cast(source.dtype())?);
                    }
                }
                Op::WithGradient => add(1, dy),
                Op::WithElementwiseDerivative => {
                    for input in (1..node.operands.len()).step_by(2) {
                        add(input, dy.mul(&value(node.operands[input + 1]))?);
                    }
                }
                Op::Attention { scale } => {
                    let query = value(node.operands[0]);
                    let key = value(node.operands[1]);
                    let value_tensor = value(node.operands[2]);
                    let rank = query.shape().len();
                    let mut scores = query
                        .matmul(&key.swap_axes(rank - 2, rank - 1)?)?
                        .mul_scalar(*scale)?;
                    if node.operands.len() == 4 {
                        scores = scores.add(&value(node.operands[3]))?;
                    }
                    let probabilities = scores.softmax(rank - 1)?;
                    let d_probabilities =
                        dy.matmul(&value_tensor.swap_axes(rank - 2, rank - 1)?)?;
                    let centered = d_probabilities.sub(
                        &d_probabilities
                            .mul(&probabilities)?
                            .sum(&[rank - 1], true)?
                            .broadcast_to(probabilities.shape())?,
                    )?;
                    let d_scores = probabilities.mul(&centered)?;
                    add(0, d_scores.matmul(&key)?.mul_scalar(*scale)?);
                    add(
                        1,
                        d_scores
                            .swap_axes(rank - 2, rank - 1)?
                            .matmul(&query)?
                            .mul_scalar(*scale)?,
                    );
                    add(2, probabilities.swap_axes(rank - 2, rank - 1)?.matmul(&dy)?);
                    if node.operands.len() == 4 {
                        add(3, d_scores);
                    }
                }
                Op::Relu => {
                    let x = value(node.operands[0]);
                    let zero = self.graph().constant(&[], &[0.])?.broadcast_to(x.shape())?;
                    add(0, x.gt_mask(&zero)?.select(&dy, &zero)?);
                }
                Op::Softplus => add(0, dy.mul(&value(node.operands[0]).sigmoid()?)?),
                Op::Sigmoid => {
                    let y = value(rxla_ir::SsaId::from_index(i));
                    let complement = value(node.operands[0]).neg()?.sigmoid()?;
                    add(0, dy.mul(&y)?.mul(&complement)?);
                }
                Op::Select => {
                    let mask = value(node.operands[0]);
                    let zero = self
                        .graph()
                        .constant(&[], &[0.])?
                        .broadcast_to(dy.shape())?;
                    add(1, mask.select(&dy, &zero)?);
                    add(2, mask.select(&zero, &dy)?);
                }
                Op::Binary(op) => {
                    let a = value(node.operands[0]);
                    let b = value(node.operands[1]);
                    match op {
                        Binary::Add => {
                            add(0, dy.clone());
                            add(1, dy);
                        }
                        Binary::Sub => {
                            add(0, dy.clone());
                            add(1, dy.neg()?);
                        }
                        Binary::Mul => {
                            add(0, dy.mul(&b)?);
                            add(1, dy.mul(&a)?);
                        }
                        Binary::Div => {
                            add(0, dy.div(&b)?);
                            add(1, dy.mul(&a.div(&b)?)?.div(&b)?.neg()?);
                        }
                        Binary::Maximum | Binary::Minimum => {
                            let zero =
                                self.graph().constant(&[], &[0.])?.broadcast_to(a.shape())?;
                            let nan = self
                                .graph()
                                .constant(&[], &[f32::NAN])?
                                .broadcast_to(a.shape())?;
                            let tie = a.eq_mask(&b)?.select(&dy.mul_scalar(0.5)?, &nan)?;
                            let (left, right) = match op {
                                Binary::Maximum => (a.gt_mask(&b)?, a.lt_mask(&b)?),
                                _ => (a.lt_mask(&b)?, a.gt_mask(&b)?),
                            };
                            add(0, left.select(&dy, &right.select(&zero, &tie)?)?);
                            add(1, right.select(&dy, &left.select(&zero, &tie)?)?);
                        }
                    }
                }
                Op::Unary(op) => {
                    let x = value(node.operands[0]);
                    let y = value(rxla_ir::SsaId::from_index(i));
                    let dx = match op {
                        // Explicit zero-derivative convention: stop the reverse
                        // path, rather than multiplying potentially nonfinite dy by zero.
                        Unary::Floor | Unary::Ceil | Unary::Round | Unary::RoundTiesEven => {
                            continue;
                        }
                        Unary::Neg => dy.neg()?,
                        Unary::Sin => dy.mul(&x.cos()?)?,
                        Unary::Cos => dy.mul(&x.sin()?)?.neg()?,
                        Unary::Exp => dy.mul(&y)?,
                        Unary::Expm1 => dy.mul(&x.exp()?)?,
                        Unary::Log => dy.div(&x)?,
                        Unary::Log1p => dy.div(&x.add_scalar(1.)?)?,
                        Unary::Sqrt => dy.div(&y)?.mul_scalar(0.5)?,
                        Unary::Rsqrt => dy.mul(&y.mul(&y)?.mul(&y)?)?.mul_scalar(-0.5)?,
                        Unary::Tanh => dy.mul(&y.mul(&y)?.neg()?.add_scalar(1.)?)?,
                        Unary::Erf => dy
                            .mul(&x.mul(&x)?.neg()?.exp()?)?
                            .mul_scalar(2. / std::f32::consts::PI.sqrt())?,
                        Unary::Abs => {
                            let zero =
                                self.graph().constant(&[], &[0.])?.broadcast_to(x.shape())?;
                            let nan = self
                                .graph()
                                .constant(&[], &[f32::NAN])?
                                .broadcast_to(x.shape())?;
                            let at_zero = x.eq_mask(&zero)?.select(&zero, &nan)?;
                            x.gt_mask(&zero)?
                                .select(&dy, &x.lt_mask(&zero)?.select(&dy.neg()?, &at_zero)?)?
                        }
                    };
                    add(0, dx);
                }
                Op::Reshape => add(0, dy.reshape(value(node.operands[0]).shape())?),
                Op::OptimizationBarrier => add(0, dy.optimization_barrier()?),
                Op::Slice(axes) => {
                    let input = value(node.operands[0]);
                    add(
                        0,
                        self.graph().node(
                            Op::SliceGradient(axes.clone()),
                            vec![dy.node_id()],
                            input.shape(),
                        )?,
                    );
                }
                Op::SliceGradient(axes) => {
                    let starts: Vec<_> = axes.iter().map(|a| a.start).collect();
                    let limits: Vec<_> = axes.iter().map(|a| a.limit).collect();
                    let strides: Vec<_> = axes.iter().map(|a| a.stride).collect();
                    add(0, dy.slice(&starts, &limits, &strides)?);
                }
                Op::Pad(padding) => {
                    let input = value(node.operands[0]);
                    let starts: Vec<_> = padding.iter().map(|p| p[0]).collect();
                    let limits: Vec<_> = starts
                        .iter()
                        .zip(input.shape())
                        .map(|(low, size)| low + size)
                        .collect();
                    add(0, dy.slice(&starts, &limits, &vec![1; starts.len()])?);
                    // Select only border cotangents. Subtracting the interior
                    // sum from the full sum loses small boundary contributions;
                    // multiplying by zero would propagate interior NaNs.
                    if live[node.operands[1].index()] {
                        let zero = self.graph().constant(&[], &[0.])?;
                        let one = self.graph().constant(&[], &[1.])?;
                        let interior = one.broadcast_to(input.shape())?.pad(padding, 0.)?;
                        let border = interior.select(&zero.broadcast_to(dy.shape())?, &dy)?;
                        add(
                            1,
                            border.sum(&(0..dy.shape().len()).collect::<Vec<_>>(), false)?,
                        );
                    }
                }
                Op::Concatenate { axis } => {
                    let mut start = 0;
                    for (operand, &id) in node.operands.iter().enumerate() {
                        let size = value(id).shape()[*axis];
                        add(operand, dy.narrow(*axis, start, size)?);
                        start += size;
                    }
                }
                Op::Reverse { axes } => add(0, dy.flip(axes)?),
                Op::Transpose { permutation } => {
                    let mut inverse = vec![0; permutation.len()];
                    for (axis, &p) in permutation.iter().enumerate() {
                        inverse[p] = axis;
                    }
                    add(0, dy.transpose(&inverse)?);
                }
                Op::Broadcast { axes } => {
                    let input = value(node.operands[0]);
                    let reduce: Vec<_> = (0..node.ty.dims.len())
                        .filter(|axis| match axes.iter().position(|a| a == axis) {
                            None => true,
                            Some(i) => input.shape()[i] == 1 && node.ty.dims[*axis] != 1,
                        })
                        .collect();
                    add(0, dy.sum(&reduce, true)?.reshape(input.shape())?);
                }
                Op::Cumsum { axis } => add(0, dy.flip(&[*axis])?.cumsum(*axis)?.flip(&[*axis])?),
                Op::Reduce { kind, axes } => {
                    let input = value(node.operands[0]);
                    let dims: Vec<_> = input
                        .shape()
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
                        .collect();
                    let dy = dy.reshape(&dims)?.broadcast_to(input.shape())?;
                    let dx = match kind {
                        Reduction::Sum => dy,
                        Reduction::Maximum => {
                            let maximum = value(rxla_ir::SsaId::from_index(i))
                                .reshape(&dims)?
                                .broadcast_to(input.shape())?;
                            let mask = input.eq_mask(&maximum)?;
                            // Winner selection and its multiplicity are discrete;
                            // subsequent differentiation must not follow these.
                            let count = mask
                                .sum(axes, true)?
                                .detach()?
                                .broadcast_to(input.shape())?;
                            let zero = self
                                .graph()
                                .constant(&[], &[0.])?
                                .broadcast_to(input.shape())?;
                            let nan = self
                                .graph()
                                .constant(&[], &[f32::NAN])?
                                .broadcast_to(input.shape())?;
                            let selected = mask.select(&dy.div(&count)?, &zero)?;
                            // Native max reductions may ignore NaN depending on
                            // backend/reduction order. Define the backward policy
                            // from the inputs rather than relying on that output.
                            let nan_count = input
                                .ne_mask(&input)?
                                .sum(axes, true)?
                                .broadcast_to(input.shape())?;
                            let selected = count.gt_mask(&zero)?.select(&selected, &nan)?;
                            nan_count.gt_mask(&zero)?.select(&nan, &selected)?
                        }
                    };
                    add(0, dx);
                }
                Op::Matmul { .. } => {
                    let a = value(node.operands[0]);
                    let b = value(node.operands[1]);
                    let mut perm: Vec<_> = (0..a.shape().len()).collect();
                    let rank = perm.len();
                    perm.swap(rank - 2, rank - 1);
                    add(0, dy.matmul(&b.transpose(&perm)?)?);
                    add(1, a.transpose(&perm)?.matmul(&dy)?);
                }
                Op::SumPool2d(options) => {
                    add(
                        0,
                        dy.sum_pool2d_gradient(value(node.operands[0]).shape(), *options)?,
                    );
                }
                Op::MaxPool2d(_) => {
                    return Err(err(
                        "max-pool gradients are not supported by the Pliron frontend",
                    ));
                }
                Op::Conv2dInputGradient(options) => {
                    let source = value(node.operands[0]);
                    let kernel = value(node.operands[1]);
                    add(0, dy.conv2d(&kernel, *options)?);
                    add(
                        1,
                        self.graph().node(
                            Op::Conv2dKernelGradient(*options),
                            vec![dy.node_id(), source.node_id()],
                            kernel.shape(),
                        )?,
                    );
                }
                Op::Conv2dKernelGradient(options) => {
                    let input = value(node.operands[0]);
                    let source = value(node.operands[1]);
                    add(
                        0,
                        self.graph().node(
                            Op::Conv2dInputGradient(*options),
                            vec![source.node_id(), dy.node_id()],
                            input.shape(),
                        )?,
                    );
                    add(1, input.conv2d(&dy, *options)?);
                }
                Op::ConvTranspose2d(options) => {
                    let input = value(node.operands[0]);
                    let kernel = value(node.operands[1]);
                    if input.shape().contains(&0) {
                        add(
                            0,
                            self.graph()
                                .constant(&[], &[0.])?
                                .broadcast_to(input.shape())?,
                        );
                        add(
                            1,
                            self.graph()
                                .constant(&[], &[0.])?
                                .broadcast_to(kernel.shape())?,
                        );
                    } else {
                        // <transpose_conv(x,k),dy> = <x,conv(dy,k)>.
                        // HWOI for transpose_conv is HWIO for this paired conv.
                        let options = Conv2dOptions {
                            strides: options.strides,
                            padding: options.padding,
                            dilation: options.dilation,
                            groups: 1,
                        };
                        add(0, dy.conv2d(&kernel, options)?);
                        add(
                            1,
                            self.graph().node(
                                Op::Conv2dKernelGradient(options),
                                vec![dy.node_id(), input.node_id()],
                                kernel.shape(),
                            )?,
                        );
                    }
                }
                Op::Conv2d(options) => {
                    let input = value(node.operands[0]);
                    let kernel = value(node.operands[1]);
                    if input.shape().contains(&0) || dy.shape().contains(&0) {
                        add(
                            0,
                            self.graph()
                                .constant(&[], &[0.])?
                                .broadcast_to(input.shape())?,
                        );
                        add(
                            1,
                            self.graph()
                                .constant(&[], &[0.])?
                                .broadcast_to(kernel.shape())?,
                        );
                    } else {
                        if options.groups == 1 {
                            add(
                                0,
                                conv2d_input_gradient(&dy, &kernel, input.shape(), *options)?,
                            );
                            add(
                                1,
                                self.graph().node(
                                    Op::Conv2dKernelGradient(*options),
                                    vec![input.node_id(), dy.node_id()],
                                    kernel.shape(),
                                )?,
                            );
                        } else {
                            add(
                                0,
                                self.graph().node(
                                    Op::Conv2dInputGradient(*options),
                                    vec![dy.node_id(), kernel.node_id()],
                                    input.shape(),
                                )?,
                            );
                            add(
                                1,
                                self.graph().node(
                                    Op::Conv2dKernelGradient(*options),
                                    vec![input.node_id(), dy.node_id()],
                                    kernel.shape(),
                                )?,
                            );
                        }
                    }
                }
                Op::Conv2dOihw(options) => {
                    let input = value(node.operands[0]);
                    let kernel = value(node.operands[1]);
                    if input.shape().contains(&0) || dy.shape().contains(&0) {
                        add(
                            0,
                            self.graph()
                                .constant(&[], &[0.])?
                                .broadcast_to(input.shape())?,
                        );
                        add(
                            1,
                            self.graph()
                                .constant(&[], &[0.])?
                                .broadcast_to(kernel.shape())?,
                        );
                    } else {
                        let hwio = kernel.transpose(&[2, 3, 1, 0])?;
                        let hwio_shape = [
                            kernel.shape()[2],
                            kernel.shape()[3],
                            kernel.shape()[1],
                            kernel.shape()[0],
                        ];
                        let (input_gradient, gradient) = if options.groups == 1 {
                            (
                                conv2d_input_gradient(&dy, &hwio, input.shape(), *options)?,
                                self.graph().node(
                                    Op::Conv2dKernelGradient(*options),
                                    vec![input.node_id(), dy.node_id()],
                                    &hwio_shape,
                                )?,
                            )
                        } else {
                            (
                                self.graph().node(
                                    Op::Conv2dInputGradient(*options),
                                    vec![dy.node_id(), hwio.node_id()],
                                    input.shape(),
                                )?,
                                self.graph().node(
                                    Op::Conv2dKernelGradient(*options),
                                    vec![input.node_id(), dy.node_id()],
                                    &hwio_shape,
                                )?,
                            )
                        };
                        add(0, input_gradient);
                        add(1, gradient.transpose(&[3, 2, 0, 1])?);
                    }
                }
                Op::DynamicSlice | Op::DynamicUpdateSlice => {
                    let updating = matches!(node.op, Op::DynamicUpdateSlice);
                    let starts = node.operands[if updating { 2 } else { 1 }..]
                        .iter()
                        .map(|&id| value(id))
                        .collect::<Vec<_>>();
                    let zero = self.graph().constant(&[], &[0.])?;
                    if updating {
                        let update = value(node.operands[1]);
                        add(
                            0,
                            dy.dynamic_update_slice(&zero.broadcast_to(update.shape())?, &starts)?,
                        );
                        add(1, dy.dynamic_slice(&starts, update.shape())?);
                    } else {
                        let source = value(node.operands[0]);
                        add(
                            0,
                            zero.broadcast_to(source.shape())?
                                .dynamic_update_slice(&dy, &starts)?,
                        );
                    }
                }
                Op::GatherGradient { axis, batched } => {
                    let id = node.operands[1];
                    let indices = value(id);
                    add(
                        0,
                        if *batched {
                            dy.take_along_axis(&indices, *axis)?
                        } else {
                            dy.take(&indices, *axis)?
                        },
                    );
                }
                Op::Take { axis } | Op::TakeAlongAxis { axis } => {
                    let source = value(node.operands[0]);
                    let gradient = self.graph().node(
                        Op::GatherGradient {
                            axis: *axis,
                            batched: matches!(node.op, Op::TakeAlongAxis { .. }),
                        },
                        vec![dy.node_id(), node.operands[1]],
                        source.shape(),
                    )?;
                    add(0, gradient);
                }
                _ => unreachable!("validated operation"),
            }
            for (index, contribution) in contributions {
                gradients[index] = Some(match gradients[index].take() {
                    Some(previous) => previous.add(&contribution)?,
                    None => contribution,
                });
            }
        }
        inputs
            .iter()
            .map(|input| match &gradients[input.node_id().index()] {
                Some(gradient) => Ok(gradient.clone()),
                None => self
                    .graph()
                    .constant(&[], &[0.])?
                    .broadcast_to(input.shape()),
            })
            .collect()
    }
}

fn conv2d_input_gradient(
    output_gradient: &Tensor,
    kernel: &Tensor,
    input_shape: &[i64],
    options: Conv2dOptions,
) -> Result<Tensor> {
    let mut output_padding = [0; 2];
    for axis in 0..2 {
        let effective = (kernel.shape()[axis] - 1) * options.dilation[axis] + 1;
        let base = (output_gradient.shape()[axis + 1] - 1) * options.strides[axis] + effective
            - options.padding[axis][0]
            - options.padding[axis][1];
        output_padding[axis] = input_shape[axis + 1] - base;
    }
    output_gradient.conv_transpose2d(
        kernel,
        ConvTranspose2dOptions {
            strides: options.strides,
            padding: options.padding,
            dilation: options.dilation,
            output_padding,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_rule_does_not_append_partial_gradients() {
        let graph = Graph::default();
        let x = graph.input(&[2]).unwrap();
        let loss = x
            .is_finite_mask()
            .unwrap()
            .sum(&[0], false)
            .unwrap()
            .exp()
            .unwrap();
        let count = graph.0.lock().unwrap().len();
        assert!(loss.grad(&[x]).is_err());
        assert_eq!(graph.0.lock().unwrap().len(), count);
    }

    #[test]
    fn byte_preprocessing_is_a_nondifferentiable_data_boundary() {
        let graph = Graph::default();
        let bytes = graph.input_dtype(&[2], DType::U8).unwrap();
        let weight = graph.input(&[2]).unwrap();
        let loss = bytes
            .cast(DType::F32)
            .unwrap()
            .mul(&weight)
            .unwrap()
            .sum(&[0], false)
            .unwrap();
        let gradient = loss.grad(&[weight]).unwrap().remove(0);
        assert_eq!(gradient.dtype(), DType::F32);
        let lowered = graph.prepare(&gradient).unwrap();
        assert_eq!(lowered.input_spec(0).unwrap().dtype, DType::U8);
        assert_eq!(lowered.output_spec(0).unwrap().dtype, DType::F32);
    }
}
