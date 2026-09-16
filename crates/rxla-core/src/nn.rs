//! Composite model operations: no opaque kernels or separate compilation boundary.
use super::*;

/// Graph values from BatchNorm training. Statistics have shape `[channels]` and
/// remain differentiable; detach them when recording nondifferentiable running
/// statistics. This does not own or implicitly update runtime state.
pub struct BatchNormTraining {
    pub output: Tensor,
    pub mean: Tensor,
    /// Population variance (correction=0), before epsilon is added.
    pub variance: Tensor,
}

impl Tensor {
    /// Elementwise Gaussian log density at `self`, parameterized by mean and
    /// log standard deviation (not log variance). All shapes/graphs must match;
    /// broadcast parameters and sum event dimensions explicitly. Computes
    /// `-0.5 * ((self - mean) * exp(-log_std))² - log_std - 0.5*ln(2*pi)`.
    ///
    /// All three operands differentiate; detach sampled actions explicitly for
    /// score-function losses, or retain their graph for pathwise derivatives.
    /// No RNG, reduction, parameter clamp, action transform/Jacobian correction,
    /// or validation of runtime values is implicit. Intended for finite inputs
    /// within F32 range; overflow/underflow and nonfinite results are not repaired.
    /// This evaluates the continuous Gaussian density, not the probability mass
    /// of the finite-grid `normal_f32` sampler.
    pub fn normal_log_prob(&self, mean: &Tensor, log_std: &Tensor) -> Result<Tensor> {
        for parameter in [mean, log_std] {
            if !Arc::ptr_eq(&self.graph().0, &parameter.graph().0) || self.shape != parameter.shape
            {
                return Err(err(
                    "normal log-prob parameters must match value graph and shape",
                ));
            }
        }
        let normalized = self.sub(mean)?.mul(&log_std.neg()?.exp()?)?;
        normalized
            .mul(&normalized)?
            .mul_scalar(-0.5)?
            .sub(log_std)?
            .add_scalar(-0.5 * std::f32::consts::TAU.ln())
    }

    /// Inverted dropout with an explicit same-shaped F32 keep mask. Zero drops
    /// an element; nonzero (including NaN) keeps it, following `select` semantics.
    /// The caller supplies Bernoulli(keep_probability) masks for ordinary
    /// dropout. No RNG, implicit broadcasting or training/evaluation mode exists
    /// here; reuse the input directly for inference.
    ///
    /// keep_probability is a finite construction-time value in (0, 1]. Kept
    /// values are divided by it; dropped values are selected to zero before
    /// scaling, including dropped NaN/Inf inputs. Kept nonfinite values and true
    /// F32 overflow are not repaired. The mask has no derivative, while value
    /// derivatives follow the same mask. Recomputations must reuse the mask.
    pub fn dropout_with_mask(&self, keep_mask: &Tensor, keep_probability: f32) -> Result<Tensor> {
        if !keep_probability.is_finite() || keep_probability <= 0. || keep_probability > 1. {
            return Err(err("dropout keep probability must be finite and in (0, 1]"));
        }
        if !Arc::ptr_eq(&self.graph().0, &keep_mask.graph().0) || self.shape != keep_mask.shape {
            return Err(err("dropout mask must match input graph and shape"));
        }
        let zero = self
            .graph()
            .constant(&[], &[0.])?
            .broadcast_to(self.shape())?;
        let divisor = self
            .graph()
            .constant(&[], &[keep_probability])?
            .broadcast_to(self.shape())?;
        keep_mask.select(self, &zero)?.div(&divisor)
    }

    /// Elementwise Huber regression loss for same-shaped targets. With residual
    /// r = self - targets, returns 0.5*r² for |r| <= delta, otherwise
    /// delta*(|r| - 0.5*delta). Delta is a finite positive construction-time value.
    /// There is no implicit reduction, broadcasting or target detach.
    ///
    /// Both inputs differentiate. The first residual derivative is r in the
    /// quadratic region and +/-delta outside. At +/-delta the first derivative
    /// agrees on both branches; the second derivative is undefined and this
    /// implementation selects the quadratic-side convention (1). At r=0 the
    /// second derivative is 1. The inactive quadratic residual is masked to zero
    /// before squaring to avoid overflow contaminating large-residual gradients.
    /// Nonfinite residuals/results are not repaired; ordinary F32 limits apply.
    pub fn huber_loss(&self, targets: &Tensor, delta: f32) -> Result<Tensor> {
        if !delta.is_finite() || delta <= 0. {
            return Err(err("Huber delta must be finite and positive"));
        }
        if !Arc::ptr_eq(&self.graph().0, &targets.graph().0) || self.shape != targets.shape {
            return Err(err("Huber targets must match prediction graph and shape"));
        }
        let residual = self.sub(targets)?;
        let magnitude = residual.abs()?;
        let limit = self
            .graph()
            .constant(&[], &[delta])?
            .broadcast_to(self.shape())?;
        let quadratic = magnitude.le_mask(&limit)?;
        let zero = self
            .graph()
            .constant(&[], &[0.])?
            .broadcast_to(self.shape())?;
        let safe_residual = quadratic.select(&residual, &zero)?;
        let squared = safe_residual.mul_scalar(0.5)?.mul(&safe_residual)?;
        let linear = magnitude.sub(&limit.mul_scalar(0.5)?)?.mul_scalar(delta)?;
        quadratic.select(&squared, &linear)
    }

    /// Elementwise binary cross entropy from F32 logits and same-shape targets.
    /// Computes `targets*softplus(-logits) + (1-targets)*softplus(logits)` without
    /// materializing probabilities or subtracting nearly equal large losses.
    /// There is no reduction: use mean/sum explicitly. Broadcasting is explicit.
    ///
    /// The caller supplies finite targets in `[0,1]`; runtime values are neither
    /// checked nor clamped. Other values compute the same weighted expression.
    /// Logits and soft targets are both differentiable; detach targets if needed.
    /// Intended for finite inputs: zero times infinity is not masked away, and
    /// nonfinite inputs/gradients are not repaired. Class weighting, label
    /// smoothing and ignore masks are explicit caller-side graph operations.
    pub fn binary_cross_entropy_with_logits(&self, targets: &Tensor) -> Result<Tensor> {
        if !Arc::ptr_eq(&self.graph().0, &targets.graph().0) {
            return Err(err("cross-graph binary cross entropy targets"));
        }
        if self.shape != targets.shape {
            return Err(err("binary cross entropy targets must match logits shape"));
        }
        targets
            .mul(&self.neg()?.softplus()?)?
            .add(&targets.neg()?.add_scalar(1.)?.mul(&self.softplus()?)?)
    }

    /// Per-example cross entropy from logits and I32 class IDs. Target shape
    /// must equal the logits shape with `axis` removed; no batch reduction is
    /// implicit. Uses log-softmax and indexed selection, not dense one-hot labels.
    ///
    /// An ID outside [0, class_count) produces NaN loss at that position, rather
    /// than silently using gather's clamped class. This is a runtime data result,
    /// not a Rust error: validate input labels or gate updates on a finite loss.
    /// Invalid-label gradients are not meaningful. Shape/owner/axis errors are
    /// returned while building the graph. Indices are nondifferentiable; there
    /// is no ignore-index, class weighting or label smoothing policy.
    pub fn cross_entropy_with_indices(&self, targets: &Tensor, axis: usize) -> Result<Tensor> {
        if !Arc::ptr_eq(&self.graph().0, &targets.graph().0) {
            return Err(err("cross-graph cross entropy indices"));
        }
        if axis >= self.shape.len() || self.shape[axis] == 0 {
            return Err(err("cross entropy requires a nonempty class axis"));
        }
        let mut expected = self.shape.to_vec();
        expected.remove(axis);
        if targets.shape() != expected {
            return Err(err("cross entropy index shape must omit the class axis"));
        }
        let mut expanded = expected.clone();
        expanded.insert(axis, 1);
        let loss = self
            .log_softmax(axis)?
            .take_along_axis(&targets.reshape(&expanded)?, axis)?
            .reshape(&expected)?
            .neg()?;
        let zero = self.graph().scalar_i32(0)?.broadcast_to(&expected)?;
        let last = self
            .graph()
            .scalar_i32((self.shape[axis] - 1).min(i32::MAX as i64) as i32)?
            .broadcast_to(&expected)?;
        let valid = zero.le_mask(targets)?.mul(&targets.le_mask(&last)?)?;
        let invalid = self
            .graph()
            .constant(&[], &[f32::NAN])?
            .broadcast_to(&expected)?;
        valid.select(&loss, &invalid)
    }

    /// Cross entropy from F32 logits and dense F32 target probabilities:
    /// `-sum(targets * log_softmax(logits), axis)`. Removes the class axis but
    /// performs no batch reduction; use mean/sum explicitly on the result.
    /// Shapes and graph ownership must match, and the class axis must be nonempty.
    ///
    /// Target values are runtime data: the caller ensures nonnegative values
    /// summing to one per distribution. They are not normalized or validated
    /// here; arbitrary coefficients compute the same weighted log loss. Both
    /// logits and targets remain differentiable (detach targets if appropriate).
    /// Intended for finite logits/targets: in particular zero times -infinity
    /// is not masked away. No sparse labels, ignore-index, weighting or label
    /// smoothing policy is implicit. This is a composite graph, not a fused
    /// kernel or a separate compilation boundary.
    pub fn cross_entropy_with_probs(&self, targets: &Tensor, axis: usize) -> Result<Tensor> {
        if !Arc::ptr_eq(&self.graph().0, &targets.graph().0) {
            return Err(err("cross-graph cross entropy targets"));
        }
        if self.shape != targets.shape {
            return Err(err("cross entropy targets must match logits shape"));
        }
        self.log_softmax(axis)?
            .mul(targets)?
            .sum(&[axis], false)?
            .neg()
    }

    /// Training BatchNorm over all dimensions except `axis`, with F32 affine
    /// weight and bias of shape `[channels]`. Uses two-pass population variance and
    /// computes `(x - mean) * rsqrt(variance + epsilon) * weight + bias`.
    /// Returns batch statistics alongside the same-shape output; no running
    /// average, momentum, counter or mutable session update is implicit.
    ///
    /// A single observation is allowed (variance zero); empty observation sets
    /// are rejected. Epsilon must be finite and positive. Finite data is intended:
    /// invalid/nonfinite statistics are not repaired. Unbiased running-variance
    /// conversion, if desired, is an explicit caller policy, not the normalization
    /// divisor. Input, affine parameters and returned statistics are differentiable.
    pub fn batch_norm_training(
        &self,
        axis: usize,
        weight: &Tensor,
        bias: &Tensor,
        epsilon: f32,
    ) -> Result<BatchNormTraining> {
        if axis >= self.shape.len()
            || self.shape[axis] <= 0
            || !epsilon.is_finite()
            || epsilon <= 0.
        {
            return Err(err(
                "BatchNorm requires a nonempty channel axis and finite positive epsilon",
            ));
        }
        for parameter in [weight, bias] {
            if parameter.shape() != [self.shape[axis]] {
                return Err(err(
                    "BatchNorm affine parameters must have shape [channels]",
                ));
            }
            if !Arc::ptr_eq(&self.graph().0, &parameter.graph().0) {
                return Err(err("cross-graph BatchNorm affine parameter"));
            }
        }
        let axes: Vec<_> = (0..self.shape.len()).filter(|&a| a != axis).collect();
        let (mean, variance) = self.moments(&axes, false, 0)?;
        let output = self.batch_norm_inference(axis, &mean, &variance, weight, bias, epsilon)?;
        Ok(BatchNormTraining {
            output,
            mean,
            variance,
        })
    }

    /// Inference BatchNorm with supplied running statistics, without statistics
    /// updates. All four parameter tensors have shape `[channels]` for `axis`.
    /// Computes `(x - mean) * rsqrt(variance + epsilon) * weight + bias` in F32.
    /// Variance values are runtime data; callers must supply valid statistics.
    pub fn batch_norm_inference(
        &self,
        axis: usize,
        mean: &Tensor,
        variance: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        epsilon: f32,
    ) -> Result<Self> {
        if axis >= self.shape.len()
            || self.shape[axis] <= 0
            || !epsilon.is_finite()
            || epsilon <= 0.
        {
            return Err(err(
                "BatchNorm requires a nonempty channel axis and finite positive epsilon",
            ));
        }
        for parameter in [mean, variance, weight, bias] {
            if parameter.shape.as_ref() != [self.shape[axis]] {
                return Err(err("BatchNorm parameters must have shape [channels]"));
            }
            if !Arc::ptr_eq(&self.graph().0, &parameter.graph().0) {
                return Err(err("cross-graph BatchNorm parameter"));
            }
        }
        let scale = variance.add_scalar(epsilon)?.rsqrt()?.mul(weight)?;
        self.sub(&mean.broadcast_in_dim(self.shape(), &[axis])?)?
            .mul(&scale.broadcast_in_dim(self.shape(), &[axis])?)?
            .add(&bias.broadcast_in_dim(self.shape(), &[axis])?)
    }

    /// Apply `x @ weight.T + bias` to the last input axis. Weight layout is
    /// `[out_features, in_features]`, matching common checkpoint conventions.
    /// Leading input dimensions are preserved; bias, if present, is `[out_features]`.
    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<Self> {
        if self.shape.is_empty()
            || weight.shape.len() != 2
            || self.shape.last() != weight.shape.get(1)
        {
            return Err(err("Linear requires input [..., in] and weight [out, in]"));
        }
        if !Arc::ptr_eq(&self.graph().0, &weight.graph().0) {
            return Err(err("cross-graph Linear weight"));
        }
        if let Some(bias) = bias {
            if bias.shape.as_ref() != [weight.shape[0]] {
                return Err(err("Linear bias must have shape [out]"));
            }
            if !Arc::ptr_eq(&self.graph().0, &bias.graph().0) {
                return Err(err("cross-graph Linear bias"));
            }
        }
        let output = self.matmul(&weight.transpose(&[1, 0])?)?;
        match bias {
            Some(bias) => output.add(&bias.broadcast_to(output.shape())?),
            None => Ok(output),
        }
    }

    /// Group normalization for `[N, C, ...]` data, independently per sample and
    /// channel group. `groups` must be positive and divide C. Channels and spatial
    /// dimensions must be nonempty; an empty batch is supported. Optional weight
    /// and bias have shape `[C]` and belong to this graph. No running statistics
    /// or training/evaluation distinction exists. Uses two-pass F32 population
    /// variance with finite positive epsilon inside the square root. Inputs and
    /// affine tensors remain differentiable, including higher derivatives.
    pub fn group_norm(
        &self,
        groups: i64,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        epsilon: f32,
    ) -> Result<Self> {
        if self.shape.len() < 2
            || groups <= 0
            || self.shape[1..].iter().any(|&size| size <= 0)
            || self.shape[1] % groups != 0
            || !epsilon.is_finite()
            || epsilon <= 0.
        {
            return Err(err(
                "GroupNorm requires [N,C,...], nonempty groups dividing C, nonempty spatial axes and finite positive epsilon",
            ));
        }
        let channels = self.shape[1];
        for affine in [weight, bias].into_iter().flatten() {
            if affine.shape.as_ref() != [channels]
                || !Arc::ptr_eq(&self.graph().0, &affine.graph().0)
            {
                return Err(err(
                    "GroupNorm affine tensors must have shape [C] in the input graph",
                ));
            }
        }
        let mut grouped = vec![self.shape[0], groups, channels / groups];
        grouped.extend_from_slice(&self.shape[2..]);
        let mut output = self
            .reshape(&grouped)?
            .layer_norm(&grouped[2..], None, None, epsilon)?
            .reshape(self.shape())?;
        if let Some(weight) = weight {
            output = output.mul(&weight.broadcast_in_dim(self.shape(), &[1])?)?;
        }
        if let Some(bias) = bias {
            output = output.add(&bias.broadcast_in_dim(self.shape(), &[1])?)?;
        }
        Ok(output)
    }

    /// Normalize over the specified trailing dimensions using population variance
    /// `mean((x - mean(x))²)`. Optional affine tensors must exactly match
    /// `normalized_shape`. Epsilon is added inside the square root.
    /// This uses F32 two-pass statistics, not `mean(x²) - mean(x)²`.
    pub fn layer_norm(
        &self,
        normalized_shape: &[i64],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        epsilon: f32,
    ) -> Result<Self> {
        if normalized_shape.is_empty()
            || normalized_shape.iter().any(|&dim| dim <= 0)
            || !self.shape.ends_with(normalized_shape)
            || !epsilon.is_finite()
            || epsilon <= 0.
        {
            return Err(err(
                "LayerNorm requires nonempty matching trailing dimensions and finite positive epsilon",
            ));
        }
        for affine in [weight, bias].into_iter().flatten() {
            if affine.shape.as_ref() != normalized_shape {
                return Err(err("LayerNorm affine shape must equal normalized_shape"));
            }
            if !Arc::ptr_eq(&self.graph().0, &affine.graph().0) {
                return Err(err("cross-graph LayerNorm affine tensor"));
            }
        }
        let axes: Vec<_> = (self.shape.len() - normalized_shape.len()..self.shape.len()).collect();
        let centered = self.sub(&self.mean(&axes, true)?.broadcast_to(self.shape())?)?;
        let inverse = centered
            .square()?
            .mean(&axes, true)?
            .add_scalar(epsilon)?
            .rsqrt()?;
        let mut output = centered.mul(&inverse.broadcast_to(self.shape())?)?;
        if let Some(weight) = weight {
            output = output.mul(&weight.broadcast_to(self.shape())?)?;
        }
        if let Some(bias) = bias {
            output = output.add(&bias.broadcast_to(self.shape())?)?;
        }
        Ok(output)
    }
}
