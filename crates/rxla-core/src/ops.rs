use super::*;
use rxla_ir::Comparison;

impl Tensor {
    /// Explicit floating-point element conversion. RXLA never inserts dtype
    /// promotion implicitly; shape and device placement are unchanged.
    pub fn cast(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        self.graph()
            .node(Op::Convert { dtype }, vec![self.node_id()], self.shape())
    }
    /// Preserve a value across an XLA optimization boundary. Unlike `detach`,
    /// this has an identity derivative. It does not copy, execute or synchronize
    /// a device buffer; backend lowering still controls final instruction choices.
    pub fn optimization_barrier(&self) -> Result<Self> {
        self.graph()
            .node(Op::OptimizationBarrier, vec![self.node_id()], self.shape())
    }
    /// Return the same symbolic F32 value with a reverse-mode gradient boundary.
    /// This is not a device copy, execution, snapshot, or compiler optimization
    /// barrier. Forward dependencies remain live; `grad` does not differentiate
    /// through this node. Other undetached uses retain their gradients.
    pub fn detach(&self) -> Result<Self> {
        self.graph()
            .node(Op::StopGradient, vec![self.node_id()], self.shape())
    }
    /// Preserve this forward value, but route reverse-mode cotangents through
    /// `surrogate` instead. Both tensors must have the same shape and graph.
    /// This explicitly substitutes a derivative, which need not be the true
    /// derivative of the forward expression. Other uses retain their own rules.
    ///
    /// Unlike `self.detach() + surrogate - surrogate.detach()`, forward does
    /// not perform cancellation or evaluate surrogate-only computations. Normal
    /// compilation preserves declared inputs; pruned compilation can remove
    /// surrogate-only inputs. Higher derivatives follow the constructed backward
    /// graph, including cotangent dependencies, not merely the surrogate Hessian.
    /// Unsupported backward operations in surrogate still fail normally.
    /// The surrogate may compose shape-changing operations such as matmul or
    /// reduction: only its final output must match this value. Its inputs may
    /// have different shapes, and its Jacobian need not be diagonal. This uses
    /// the existing derivative rules of that expression, not a supplied VJP body.
    /// No callback, arbitrary custom VJP, external kernel or state effect is added.
    pub fn with_gradient_of(&self, surrogate: &Self) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &surrogate.graph().0) || self.shape != surrogate.shape {
            return Err(err("gradient surrogate must match value graph and shape"));
        }
        self.graph().node(
            Op::WithGradient,
            vec![self.node_id(), surrogate.node_id()],
            self.shape(),
        )
    }
    /// Preserve this forward value while defining a diagonal reverse-mode rule
    /// with respect to `input`: its cotangent is multiplied by `derivative`.
    /// All three tensors must share a graph and shape. The forward expression's
    /// other dependencies receive no gradient through this wrapper; independent
    /// uses keep their ordinary rules. No Jacobian or native callback is created.
    ///
    /// This asserts an elementwise derivative, not a general VJP for reductions,
    /// broadcasts or matrix operations. The rule can intentionally differ from
    /// the true derivative. Higher derivatives differentiate the supplied
    /// derivative expression and cotangent dependencies; detach explicitly when
    /// a coefficient should be constant. Forward-only compilation excludes the
    /// input/derivative edges unless used by the forward expression itself.
    pub fn with_elementwise_derivative(&self, input: &Self, derivative: &Self) -> Result<Self> {
        self.with_elementwise_derivatives(&[(input, derivative)])
    }

    /// Multi-input diagonal rule: each pair is `(input, local_partial_derivative)`.
    /// The nonempty list must contain tensors with this value's graph and shape.
    /// Repeated inputs add contributions, as do shared ancestors of distinct
    /// inputs. Validation finishes before creating the wrapper node.
    ///
    /// In the first reverse sweep, derivative expressions are coefficients and
    /// cotangents flow only through listed inputs. Higher-order sweeps also
    /// differentiate those expressions, including cross-input dependencies.
    /// This does not declare arbitrary cross-element Jacobians. All forward,
    /// pruning and substitution semantics of the single-input method apply.
    pub fn with_elementwise_derivatives(&self, partials: &[(&Self, &Self)]) -> Result<Self> {
        if partials.is_empty() {
            return Err(err("elementwise derivatives require at least one input"));
        }
        for tensor in partials
            .iter()
            .flat_map(|&(input, derivative)| [input, derivative])
        {
            if !Arc::ptr_eq(&self.graph().0, &tensor.graph().0) || self.shape != tensor.shape {
                return Err(err(
                    "elementwise derivative must match value graph and shape",
                ));
            }
        }
        let mut operands = vec![self.node_id()];
        for &(input, derivative) in partials {
            operands.extend([input.node_id(), derivative.node_id()]);
        }
        self.graph()
            .node(Op::WithElementwiseDerivative, operands, self.shape())
    }

    fn compare_mask(&self, rhs: &Self, comparison: Comparison) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &rhs.graph().0) {
            return Err(err("cross-graph comparison operands"));
        }
        if self.shape != rhs.shape {
            return Err(err(
                "comparison shape mismatch; broadcasting must be explicit",
            ));
        }
        self.graph().node(
            Op::CompareMask(comparison),
            vec![self.node_id(), rhs.node_id()],
            self.shape(),
        )
    }

    /// Elementwise F32 equality mask (true=1, false=0), with explicit broadcasting.
    /// NaN compares unequal to every value, including itself; +0 equals -0.
    pub fn eq_mask(&self, rhs: &Self) -> Result<Self> {
        self.compare_mask(rhs, Comparison::Equal)
    }
    /// Elementwise F32 inequality mask. NaN compares unequal to every value.
    /// Shapes and graph ownership must match, as for `eq_mask`.
    pub fn ne_mask(&self, rhs: &Self) -> Result<Self> {
        self.compare_mask(rhs, Comparison::NotEqual)
    }
    /// Elementwise F32 less-than mask. NaN operands yield false.
    /// Shapes and graph ownership must match, as for `eq_mask`.
    pub fn lt_mask(&self, rhs: &Self) -> Result<Self> {
        self.compare_mask(rhs, Comparison::Less)
    }
    /// Elementwise F32 less-than-or-equal mask. NaN operands yield false.
    /// Shapes and graph ownership must match, as for `eq_mask`.
    pub fn le_mask(&self, rhs: &Self) -> Result<Self> {
        if self.dtype() == DType::I32 && rhs.dtype() == DType::I32 {
            if !Arc::ptr_eq(&self.graph().0, &rhs.graph().0) || self.shape != rhs.shape {
                return Err(err(
                    "integer comparison operands must match graph and shape",
                ));
            }
            return self.graph().node(
                Op::IndexLessEqualMask,
                vec![self.node_id(), rhs.node_id()],
                self.shape(),
            );
        }
        self.compare_mask(rhs, Comparison::LessEqual)
    }
    /// Elementwise F32 greater-than mask. NaN operands yield false.
    /// Shapes and graph ownership must match, as for `eq_mask`.
    pub fn gt_mask(&self, rhs: &Self) -> Result<Self> {
        self.compare_mask(rhs, Comparison::Greater)
    }
    /// Elementwise F32 greater-than-or-equal mask. NaN operands yield false.
    /// Shapes and graph ownership must match, as for `eq_mask`.
    pub fn ge_mask(&self, rhs: &Self) -> Result<Self> {
        self.compare_mask(rhs, Comparison::GreaterEqual)
    }

    /// Elementwise selection using this F32 mask: zero (including -0) selects
    /// `on_false`; every nonzero value, including NaN, selects `on_true`.
    /// All operands must belong to this graph and have identical shapes; use
    /// explicit broadcasting. This lowers to HLO select, not mask arithmetic,
    /// so an unselected NaN/Inf does not contaminate the selected value.
    /// Both branches are graph operands: this is not lazy control flow and does
    /// not suppress computation or effects in either branch.
    pub fn select(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        if self.dtype() != DType::F32 {
            return Err(err("select mask must be F32"));
        }
        if on_true.dtype() != on_false.dtype() {
            return Err(err("select branches must have the same dtype"));
        }
        for value in [on_true, on_false] {
            if !Arc::ptr_eq(&self.graph().0, &value.graph().0) {
                return Err(err("cross-graph select operands"));
            }
            if self.shape != value.shape {
                return Err(err("select shape mismatch; broadcasting must be explicit"));
            }
        }
        self.graph().node(
            Op::Select,
            vec![self.node_id(), on_true.node_id(), on_false.node_id()],
            self.shape(),
        )
    }

    /// Elementwise finite-value mask (F32 1 for finite, 0 for NaN or infinity).
    /// Uses HLO is-finite followed by conversion, not arithmetic cancellation.
    pub fn is_finite_mask(&self) -> Result<Self> {
        self.graph()
            .node(Op::IsFiniteMask, vec![self.node_id()], self.shape())
    }
    /// Gaussian error function lowered to the backend's native HLO `erf` op.
    pub fn erf(&self) -> Result<Self> {
        self.unary(Unary::Erf)
    }
    /// Round down toward negative infinity, retaining F32 shape/dtype.
    /// Rounding ops use an explicit zero reverse-derivative convention everywhere
    /// (including discontinuities and nonfinite inputs/cotangents), stopping the
    /// backward path. Use `with_gradient_of` for an explicitly chosen estimator.
    /// Forward NaNs remain NaN and infinities remain infinite; no integer cast.
    pub fn floor(&self) -> Result<Self> {
        self.unary(Unary::Floor)
    }
    /// Round up toward positive infinity; same zero-gradient policy as `floor`.
    pub fn ceil(&self) -> Result<Self> {
        self.unary(Unary::Ceil)
    }
    /// Round to nearest integer, halfway cases away from zero (Rust f32::round
    /// convention). Retains F32; same zero-gradient policy as `floor`.
    pub fn round(&self) -> Result<Self> {
        self.unary(Unary::Round)
    }
    /// Round to nearest integer, halfway cases to the nearest even integer.
    /// Retains F32; same zero-gradient policy as `floor`. No implicit quantization
    /// scale, clipping, zero point, integer storage or straight-through gradient.
    pub fn round_ties_even(&self) -> Result<Self> {
        self.unary(Unary::RoundTiesEven)
    }
    /// Elementwise sine of angles in radians. Uses native F32 backend math and
    /// supports reverse-mode/higher-order differentiation. Accuracy and argument
    /// reduction for very large angles are backend-dependent, not exact arithmetic.
    pub fn sin(&self) -> Result<Self> {
        self.unary(Unary::Sin)
    }
    /// Elementwise cosine of angles in radians, with the same native F32 and
    /// differentiation semantics as `sin`.
    pub fn cos(&self) -> Result<Self> {
        self.unary(Unary::Cos)
    }
    pub(super) fn unary(&self, op: Unary) -> Result<Self> {
        self.graph()
            .node(Op::Unary(op), vec![self.node_id()], &self.shape)
    }
    pub fn log(&self) -> Result<Self> {
        self.unary(Unary::Log)
    }
    /// Natural logarithm of `1+x`, using native HLO log-plus-one to retain small
    /// values near zero. Inputs below -1 produce NaN; -1 produces -infinity.
    pub fn log1p(&self) -> Result<Self> {
        self.unary(Unary::Log1p)
    }
    /// `exp(x)-1`, using native HLO exponential-minus-one near zero.
    pub fn expm1(&self) -> Result<Self> {
        self.unary(Unary::Expm1)
    }
    /// Absolute value; reverse-mode uses zero at either signed zero and NaN at NaN.
    pub fn abs(&self) -> Result<Self> {
        self.unary(Unary::Abs)
    }
    /// Stable `log(1+exp(x))` at beta=1, without a threshold approximation.
    /// Uses `max(x,0) + log1p(exp(-abs(x)))`; large positive finite values avoid
    /// exponential overflow and small negative tails avoid subtractive loss.
    pub fn softplus(&self) -> Result<Self> {
        self.graph()
            .node(Op::Softplus, vec![self.node_id()], self.shape())
    }
    pub fn neg(&self) -> Result<Self> {
        self.unary(Unary::Neg)
    }
    pub fn sqrt(&self) -> Result<Self> {
        self.unary(Unary::Sqrt)
    }
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary(Unary::Rsqrt)
    }
    pub fn tanh(&self) -> Result<Self> {
        self.unary(Unary::Tanh)
    }
    pub fn square(&self) -> Result<Self> {
        self.mul(self)
    }
    /// Elementwise maximum; backward splits ties equally between operands.
    /// An unordered (NaN) comparison yields NaN gradients for both operands.
    pub fn maximum(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, Binary::Maximum)
    }
    /// Elementwise minimum, with the same backward tie/NaN policy as maximum.
    pub fn minimum(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, Binary::Minimum)
    }
    /// Clamp elementwise to an ordered interval. Infinite bounds are allowed;
    /// NaN bounds and reversed intervals are rejected during graph construction.
    /// Backward gives half the incoming gradient at a finite boundary when
    /// low < high. Equal bounds stop gradients, preserving forward NaN behavior.
    pub fn clamp(&self, low: f32, high: f32) -> Result<Self> {
        if low.is_nan() || high.is_nan() || low > high {
            return Err(err("clamp requires ordered non-NaN bounds"));
        }
        let equal_bounds = low == high;
        let low = self
            .graph()
            .constant(&[], &[low])?
            .broadcast_to(self.shape())?;
        let high = self
            .graph()
            .constant(&[], &[high])?
            .broadcast_to(self.shape())?;
        let result = self.maximum(&low)?.minimum(&high)?;
        if equal_bounds {
            result.detach()
        } else {
            Ok(result)
        }
    }
    /// `clamp(alpha * x + beta, 0, 1)`, with explicit finite coefficients.
    /// Different model/export conventions use different alpha values.
    pub fn hard_sigmoid(&self, alpha: f32, beta: f32) -> Result<Self> {
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(err("hard_sigmoid requires finite coefficients"));
        }
        self.mul_scalar(alpha)?.add_scalar(beta)?.clamp(0., 1.)
    }
    pub fn add_scalar(&self, value: f32) -> Result<Self> {
        self.add(&self.graph().constant(&[], &[value])?.broadcast_as(self)?)
    }
    pub fn mul_scalar(&self, value: f32) -> Result<Self> {
        self.mul(&self.graph().constant(&[], &[value])?.broadcast_as(self)?)
    }
    /// Elementwise max(x, 0), with gradient 1 for x > 0 and 0 for x <= 0.
    /// The zero-point convention includes both signed zeros; gradients at NaN
    /// and infinity are not a numerical-recovery guarantee.
    pub fn relu(&self) -> Result<Self> {
        self.graph()
            .node(Op::Relu, vec![self.node_id()], self.shape())
    }
    /// Logistic sigmoid with a nonpositive exponential argument to avoid
    /// intermediate overflow. Backward uses sigmoid(x)*sigmoid(-x), retaining
    /// positive-tail derivatives even when sigmoid(x) rounds to one. Subnormal
    /// outputs/derivatives may be flushed to zero by the device.
    pub fn sigmoid(&self) -> Result<Self> {
        self.graph()
            .node(Op::Sigmoid, vec![self.node_id()], self.shape())
    }
    pub fn silu(&self) -> Result<Self> {
        self.mul(&self.sigmoid()?)
    }
    /// GELU using the erf formula `0.5*x*(1 + erf(x/sqrt(2)))`.
    /// "Exact" refers to the formula, not exact arithmetic: evaluation is F32,
    /// and small negative tails may round to zero. Use `gelu_tanh` only when the
    /// model requests that approximation. Nonfinite inputs have ordinary formula
    /// semantics (in particular, negative infinity can produce NaN).
    pub fn gelu(&self) -> Result<Self> {
        self.mul_scalar(0.5)?.mul(
            &self
                .mul_scalar(std::f32::consts::FRAC_1_SQRT_2)?
                .erf()?
                .add_scalar(1.)?,
        )
    }
    /// Tanh-approximate GELU:
    /// `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`.
    /// This is a different approximation from `gelu`, not a selectable backend
    /// optimization of it. All operations remain visible to XLA for fusion.
    pub fn gelu_tanh(&self) -> Result<Self> {
        let cubic = self.square()?.mul(self)?.mul_scalar(0.044715)?;
        let gate = self
            .add(&cubic)?
            .mul_scalar((2. / std::f32::consts::PI).sqrt())?
            .tanh()?
            .add_scalar(1.)?;
        self.mul_scalar(0.5)?.mul(&gate)
    }

    /// Broadcast using trailing-axis alignment. Singleton axes can expand.
    pub fn broadcast_to(&self, dims: &[i64]) -> Result<Self> {
        if dims.len() < self.shape.len() {
            return Err(err("broadcast target rank is too small"));
        }
        let offset = dims.len() - self.shape.len();
        self.broadcast_in_dim(dims, &(offset..dims.len()).collect::<Vec<_>>())
    }

    /// Broadcast to another tensor's logical shape, retaining any bounded
    /// dynamic axes in that tensor's signature.
    pub fn broadcast_as(&self, target: &Tensor) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &target.graph().0) {
            return Err(err("broadcast target belongs to another graph"));
        }
        if target.dynamic_bounds.is_empty() {
            return self.broadcast_to(target.shape());
        }
        if target.ndim() < self.ndim() {
            return Err(err("broadcast target rank is too small"));
        }
        let offset = target.ndim() - self.ndim();
        let axes = (offset..target.ndim()).collect::<Vec<_>>();
        for (&axis, &size) in axes.iter().zip(self.shape.iter()) {
            let target_size = target.shape[axis];
            if size != 1 && size != target_size {
                return Err(err("incompatible bounded broadcast dimension"));
            }
        }
        if self.ty() == target.ty() {
            return Ok(self.clone());
        }
        self.graph()
            .node_typed(Op::Broadcast { axes }, vec![self.node_id()], target.ty())
    }
    pub fn broadcast_in_dim(&self, dims: &[i64], axes: &[usize]) -> Result<Self> {
        elements(dims)?;
        if axes.len() != self.shape.len() || axes.windows(2).any(|w| w[0] >= w[1]) {
            return Err(err(
                "broadcast axes must be strictly increasing and match input rank",
            ));
        }
        for (&axis, &size) in axes.iter().zip(self.shape.iter()) {
            if axis >= dims.len() || (size != 1 && size != dims[axis]) {
                return Err(err("incompatible broadcast dimension"));
            }
        }
        if dims == self.shape.as_ref() && axes.iter().copied().eq(0..dims.len()) {
            return Ok(self.clone());
        }
        self.graph().node(
            Op::Broadcast {
                axes: axes.to_vec(),
            },
            vec![self.node_id()],
            dims,
        )
    }
    /// Reverse element order along explicit axes without changing shape.
    /// Axes must be distinct and in range. Empty axes are a no-op.
    pub fn flip(&self, axes: &[usize]) -> Result<Self> {
        let mut canonical = axes.to_vec();
        canonical.sort_unstable();
        if canonical.iter().any(|&axis| axis >= self.shape.len())
            || canonical.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(err("flip axes must be distinct and in range"));
        }
        if canonical.is_empty() {
            return Ok(self.clone());
        }
        self.graph().node(
            Op::Reverse { axes: canonical },
            vec![self.node_id()],
            self.shape(),
        )
    }

    pub fn transpose(&self, permutation: &[usize]) -> Result<Self> {
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        if sorted != (0..self.shape.len()).collect::<Vec<_>>() {
            return Err(err("invalid transpose permutation"));
        }
        let dims: Vec<_> = permutation.iter().map(|&i| self.shape[i]).collect();
        self.graph().node(
            Op::Transpose {
                permutation: permutation.to_vec(),
            },
            vec![self.node_id()],
            &dims,
        )
    }
    fn reduce(&self, axes: &[usize], keepdims: bool, maximum: bool) -> Result<Self> {
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        if sorted.iter().any(|&a| a >= self.shape.len()) || sorted.windows(2).any(|w| w[0] == w[1])
        {
            return Err(err("invalid reduction axes"));
        }
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let dims: Vec<_> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !sorted.contains(i))
            .map(|(_, &d)| d)
            .collect();
        let output = self.graph().node(
            Op::Reduce {
                kind: if maximum {
                    Reduction::Maximum
                } else {
                    Reduction::Sum
                },
                axes: sorted.clone(),
            },
            vec![self.node_id()],
            &dims,
        )?;
        if keepdims {
            output.reshape(
                &self
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| if sorted.contains(&i) { 1 } else { d })
                    .collect::<Vec<_>>(),
            )
        } else {
            Ok(output)
        }
    }
    pub fn sum(&self, axes: &[usize], keepdims: bool) -> Result<Self> {
        self.reduce(axes, keepdims, false)
    }
    /// Inclusive prefix sums along one axis, retaining the input shape.
    /// Empty tensors and length-one axes are identity. Uses one HLO prefix
    /// reduce-window; graph size does not scale with the axis length. Backend
    /// summation order and performance are not prescribed, so F32 results need
    /// not match a sequential host sum bitwise. Reverse-mode is a suffix sum
    /// and supports higher derivatives through composed differentiable ops.
    pub fn cumsum(&self, axis: usize) -> Result<Self> {
        if axis >= self.shape.len() {
            return Err(err("cumsum axis out of range"));
        }
        let length = self.shape[axis];
        if length <= 1 || self.shape.contains(&0) {
            return Ok(self.clone());
        }
        if length.checked_add(length - 1).is_none() {
            return Err(err("cumsum padded extent overflows I64"));
        }
        self.graph()
            .node(Op::Cumsum { axis }, vec![self.node_id()], self.shape())
    }
    /// Stable elementwise `log(exp(self) + exp(rhs))`, with explicit broadcasting.
    /// Finite results use `max + log1p(exp(-abs(self-rhs)))` to retain small
    /// increments. Explicit sigmoid partials give smooth derivatives at ties
    /// without differentiating max/abs tie selection. Negative infinity acts as a
    /// mask; positive infinity and NaN propagate. Gradients of nonfinite results
    /// are undefined and may be NaN, as for `logsumexp`.
    pub fn logaddexp(&self, rhs: &Self) -> Result<Self> {
        let maximum = self.maximum(rhs)?;
        let delta = self.sub(rhs)?;
        let finite = maximum.add(&delta.abs()?.neg()?.exp()?.log1p()?)?;
        // Handles equal infinities and opposite infinities without inf-inf in
        // the selected result. Nonfinite gradients remain intentionally undefined.
        let nonfinite = self.exp()?.add(&rhs.exp()?)?.log()?;
        let forward = maximum.is_finite_mask()?.select(&finite, &nonfinite)?;
        forward.with_elementwise_derivatives(&[
            (self, &delta.sigmoid()?),
            (rhs, &delta.neg()?.sigmoid()?),
        ])
    }

    /// Inclusive stable cumulative log-sum-exp along one axis, preserving shape.
    /// Uses a doubling scan with O(log N) stages and O(N log N) elementwise work.
    /// Empty and singleton axes are identity. Composed derivatives are supported;
    /// nonfinite-result derivatives are undefined. See `logcumsumexp_tree` for an
    /// opt-in work-efficient alternative with different compilation/runtime costs.
    pub fn logcumsumexp(&self, axis: usize) -> Result<Self> {
        let &length = self
            .shape
            .get(axis)
            .ok_or_else(|| err("logcumsumexp axis out of range"))?;
        if length <= 1 || self.shape.contains(&0) {
            return Ok(self.clone());
        }
        let mut result = self.clone();
        let mut offset = 1;
        while offset < length {
            let prefix = result.narrow(axis, 0, offset)?;
            let tail = result
                .narrow(axis, offset, length - offset)?
                .logaddexp(&result.narrow(axis, 0, length - offset)?)?;
            result = Self::concatenate(&[prefix, tail], axis)?;
            if offset >= length - offset {
                break;
            }
            offset *= 2;
        }
        Ok(result)
    }

    /// Opt-in work-efficient version of `logcumsumexp`, with identical semantics
    /// but potentially different F32 rounding and substantially higher compile
    /// cost. Local measurements favor its long CPU execution, not CUDA execution;
    /// benchmark your shapes before choosing it. No automatic placement occurs.
    /// Uses a tree prefix scan with O(log(axis length)) graph stages and O(N)
    /// total elementwise work, without a quadratic prefix matrix. Backend fusion
    /// and allocation determine actual performance. Empty and length-one axes are
    /// identity. Reverse mode and higher derivatives use the composed graph.
    /// Nonfinite-result derivatives are undefined, as for `logaddexp`.
    pub fn logcumsumexp_tree(&self, axis: usize) -> Result<Self> {
        self.prefix_tree(axis, Self::logaddexp)
    }

    /// Inclusive cumulative products along an axis, preserving shape. Empty
    /// tensors and singleton axes are identity. Uses a work-efficient tree scan;
    /// F32 association can differ from a sequential product. Reverse and higher
    /// derivatives follow multiplication, without dividing by input values, so
    /// finite zeros and negative values are supported. Nonfinite arithmetic may
    /// produce NaN. This does not promise a single native scan kernel.
    pub fn cumprod(&self, axis: usize) -> Result<Self> {
        self.prefix_tree(axis, Self::mul)
    }

    /// Inclusive elementwise affine recurrence along `axis`, with zero initial
    /// state: `s[t] = multiplier[t] * s[t-1] + self[t]`. Operands must share
    /// graph and shape; broadcasting is explicit. Other axes index independent
    /// sequences. Returns all states with the original shape.
    /// The first state is exactly `self[0]`; the first multiplier is ignored,
    /// rather than evaluating a multiplication by an explicit initial zero.
    ///
    /// Composes affine pairs in a work-efficient tree without division. Finite
    /// zero coefficients reset the recurrence and retain ordinary derivatives.
    /// Empty tensors are identity. F32 reassociation can differ from sequential
    /// evaluation; NaN/Inf follow ordinary arithmetic (zero does not mask NaN).
    /// This is a graph composition, not a new native scan or distributed primitive.
    pub fn affine_scan(&self, multiplier: &Self, axis: usize) -> Result<Self> {
        self.affine_prefix_pairs(multiplier, axis)?
            .narrow(self.shape.len(), 1, 1)?
            .reshape(self.shape())
    }

    /// Affine recurrence with an explicit initial state. Its shape must equal
    /// the input shape with `axis` removed; graph ownership must also match.
    /// All multipliers, including the first, participate: `s[0] = a[0]*initial+b[0]`.
    /// Returns every state, not a separately owned final-state handle. The last
    /// slice can seed another symbolic block, retaining gradients through the
    /// boundary; detach it explicitly for truncated backpropagation.
    ///
    /// Uses composed prefix affine pairs and has the same reassociation and
    /// nonfinite caveats as `affine_scan`. An empty scan returns an empty tensor,
    /// not the initial state. This does not implicitly update a Session slot.
    pub fn affine_scan_from(&self, multiplier: &Self, initial: &Self, axis: usize) -> Result<Self> {
        if axis >= self.shape.len() {
            return Err(err("affine scan axis out of range"));
        }
        let mut initial_shape = self.shape().to_vec();
        initial_shape.remove(axis);
        if !Arc::ptr_eq(&self.graph().0, &initial.graph().0) || initial.shape() != initial_shape {
            return Err(err(
                "affine scan initial state must match graph and non-scan dimensions",
            ));
        }
        let pairs = self.affine_prefix_pairs(multiplier, axis)?;
        initial_shape.insert(axis, 1);
        let initial = initial
            .reshape(&initial_shape)?
            .broadcast_to(self.shape())?;
        let a = pairs
            .narrow(self.shape.len(), 0, 1)?
            .reshape(self.shape())?;
        let b = pairs
            .narrow(self.shape.len(), 1, 1)?
            .reshape(self.shape())?;
        a.mul(&initial)?.add(&b)
    }

    fn affine_prefix_pairs(&self, multiplier: &Self, axis: usize) -> Result<Self> {
        if axis >= self.shape.len() {
            return Err(err("affine scan axis out of range"));
        }
        if !Arc::ptr_eq(&self.graph().0, &multiplier.graph().0) || self.shape != multiplier.shape {
            return Err(err("affine scan operands must match graph and shape"));
        }
        let pair_axis = self.shape.len();
        let pairs = Self::stack(&[multiplier.clone(), self.clone()], pair_axis)?;
        pairs.prefix_tree(axis, Self::compose_affine_pairs)
    }

    fn compose_affine_pairs(left: &Self, right: &Self) -> Result<Self> {
        let axis = left.shape.len() - 1;
        let a = right.narrow(axis, 0, 1)?;
        let b = a
            .mul(&left.narrow(axis, 1, 1)?)?
            .add(&right.narrow(axis, 1, 1)?)?;
        let a = a.mul(&left.narrow(axis, 0, 1)?)?;
        Self::concatenate(&[a, b], axis)
    }

    // A function pointer only dispatches while constructing the graph; no
    // callback or generic specialization enters the compiled executable.
    fn prefix_tree(&self, axis: usize, combine: fn(&Self, &Self) -> Result<Self>) -> Result<Self> {
        let &length = self
            .shape
            .get(axis)
            .ok_or_else(|| err("prefix scan axis out of range"))?;
        if length <= 1 || self.shape.contains(&0) {
            return Ok(self.clone());
        }
        let mut starts = vec![0; self.shape.len()];
        let mut strides = vec![1; self.shape.len()];
        strides[axis] = 2;
        let even = self.slice(&starts, self.shape(), &strides)?;
        starts[axis] = 1;
        let odd = self.slice(&starts, self.shape(), &strides)?;
        let pairs = length / 2;
        let odd_prefixes =
            combine(&even.narrow(axis, 0, pairs)?, &odd)?.prefix_tree(axis, combine)?;
        let remaining_even = length - pairs - 1;
        let first = self.narrow(axis, 0, 1)?;
        let even_prefixes = if remaining_even == 0 {
            first
        } else {
            let tail = combine(
                &odd_prefixes.narrow(axis, 0, remaining_even)?,
                &even.narrow(axis, 1, remaining_even)?,
            )?;
            Self::concatenate(&[first, tail], axis)?
        };
        let mut merged_shape = self.shape().to_vec();
        merged_shape[axis] = 2 * pairs;
        let merged = Self::stack(
            &[even_prefixes.narrow(axis, 0, pairs)?, odd_prefixes],
            axis + 1,
        )?
        .reshape(&merged_shape)?;
        if length % 2 == 0 {
            Ok(merged)
        } else {
            Self::concatenate(&[merged, even_prefixes.narrow(axis, pairs, 1)?], axis)
        }
    }

    /// Reduce maximum over explicit axes. Reverse-mode divides the incoming
    /// gradient equally among all tied winners (including equal infinities).
    /// Any NaN input gives NaN gradients throughout its reduced slice, independent
    /// of whether the backend's native forward max propagates that NaN. Winner
    /// selection/count is nondifferentiable; counts use F32 reduction arithmetic.
    /// Empty input gradients are empty; empty axes return the original tensor.
    pub fn max(&self, axes: &[usize], keepdims: bool) -> Result<Self> {
        self.reduce(axes, keepdims, true)
    }
    /// Reduce minimum using `-max(-x)`, with the same backward tie/NaN policy as
    /// max. Empty reductions produce positive infinity; empty axes are identity.
    pub fn min(&self, axes: &[usize], keepdims: bool) -> Result<Self> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        self.neg()?.max(axes, keepdims)?.neg()
    }
    /// Stable log of the sum of exponentials over explicit axes. Finite slices
    /// use a detached maximum shift, so backward is softmax without derivatives
    /// of the discrete centering choice. Empty axes return the original tensor.
    ///
    /// A slice with finite values and negative-infinity masks is supported;
    /// masked entries receive zero gradient. All-negative-infinity or empty
    /// reductions yield negative infinity, positive infinity yields positive
    /// infinity, and NaN inputs propagate NaN. Gradients for nonfinite results
    /// are undefined and may be NaN; empty input gradients remain empty.
    pub fn logsumexp(&self, axes: &[usize], keepdims: bool) -> Result<Self> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let maximum = self.max(axes, true)?;
        let zero = self
            .graph()
            .constant(&[], &[0.])?
            .broadcast_to(maximum.shape())?;
        // Avoid inf-inf in forward evaluation. Selecting the finite shift is
        // discrete and is not part of the derivative of logsumexp.
        let shift = maximum
            .is_finite_mask()?
            .select(&maximum, &zero)?
            .detach()?;
        let centered = self.sub(&shift.broadcast_to(self.shape())?)?;
        let result = centered.exp()?.sum(axes, true)?.log()?.add(&shift)?;
        if keepdims {
            Ok(result)
        } else {
            let shape: Vec<_> = self
                .shape()
                .iter()
                .enumerate()
                .filter(|(axis, _)| !axes.contains(axis))
                .map(|(_, &size)| size)
                .collect();
            result.reshape(&shape)
        }
    }
    pub fn mean(&self, axes: &[usize], keepdims: bool) -> Result<Self> {
        let result = self.sum(axes, keepdims)?;
        let count = elements(&axes.iter().map(|&a| self.shape[a]).collect::<Vec<_>>())?;
        if count == 0 {
            return Err(err("mean over empty dimensions is undefined"));
        }
        result.mul_scalar(1. / count as f32)
    }
    /// Mean and two-pass variance over explicit axes. Variance divides the sum
    /// of squared deviations by `count - correction`: pass 0 for population
    /// variance or 1 for sample variance. No default correction is implicit.
    /// Both outputs have the same shape; keepdims retains reduced singleton axes.
    ///
    /// Requires count > correction; empty reduced dimensions are errors even
    /// when another dimension is empty. Other empty axes in the output are valid.
    /// Empty axes mean one observation per input element. Uses ordinary F32
    /// summation/centering, not Welford or an exact/high-precision accumulator.
    /// Finite inputs are intended; NaNs/infinities are not ignored or repaired.
    pub fn moments(
        &self,
        axes: &[usize],
        keepdims: bool,
        correction: usize,
    ) -> Result<(Self, Self)> {
        // mean validates axes and nonzero reduction size before indexing shape.
        let mean = self.mean(axes, true)?;
        let count = elements(
            &axes
                .iter()
                .map(|&axis| self.shape[axis])
                .collect::<Vec<_>>(),
        )?;
        if correction >= count {
            return Err(err(
                "variance requires reduction count greater than correction",
            ));
        }
        let centered = self.sub(&mean.broadcast_to(self.shape())?)?;
        let variance = centered
            .square()?
            .sum(axes, keepdims)?
            .mul_scalar(1. / (count - correction) as f32)?;
        let mean = if keepdims {
            mean
        } else {
            mean.reshape(variance.shape())?
        };
        Ok((mean, variance))
    }

    /// Two-pass variance; see `moments` for correction, shape and numeric policy.
    /// For standard deviation, explicitly apply sqrt (whose derivative is singular
    /// at zero variance); add epsilon explicitly if regularization is desired.
    pub fn variance(&self, axes: &[usize], keepdims: bool, correction: usize) -> Result<Self> {
        Ok(self.moments(axes, keepdims, correction)?.1)
    }

    pub fn softmax(&self, axis: usize) -> Result<Self> {
        if axis >= self.shape.len() || self.shape[axis] == 0 {
            return Err(err("softmax requires nonempty valid axis"));
        }
        let centered = self.sub(
            &self
                .max(&[axis], true)?
                .detach()?
                .broadcast_to(&self.shape)?,
        )?;
        let e = centered.exp()?;
        e.div(&e.sum(&[axis], true)?.broadcast_to(&self.shape)?)
    }
    /// Log probabilities along a nonempty axis, computed as
    /// `(x - max(x)) - log(sum(exp(x - max(x))))`, not `log(softmax(x))`.
    /// This avoids probability underflow for widely separated finite logits.
    /// Negative-infinity masks are supported when a slice has a finite maximum;
    /// all-masked slices, positive infinity and NaN have no finite-result guarantee.
    /// Extremely large differences can still overflow F32. Other empty axes are
    /// allowed and yield an empty output of the same shape.
    pub fn log_softmax(&self, axis: usize) -> Result<Self> {
        if axis >= self.shape.len() || self.shape[axis] == 0 {
            return Err(err("log_softmax requires nonempty valid axis"));
        }
        let centered = self.sub(
            &self
                .max(&[axis], true)?
                .detach()?
                .broadcast_to(&self.shape)?,
        )?;
        let normalizer = centered.exp()?.sum(&[axis], true)?.log()?;
        centered.sub(&normalizer.broadcast_to(&self.shape)?)
    }
    pub fn rms_norm(&self, weight: &Tensor, epsilon: f32) -> Result<Self> {
        if self.shape.is_empty() || !epsilon.is_finite() || epsilon <= 0. {
            return Err(err("RMSNorm requires rank >=1 and finite positive epsilon"));
        }
        let axis = self.shape.len() - 1;
        if weight.shape.as_ref() != [self.shape[axis]] {
            return Err(err("RMSNorm weight shape mismatch"));
        }
        let scale = self
            .square()?
            .mean(&[axis], true)?
            .add_scalar(epsilon)?
            .rsqrt()?
            .broadcast_to(&self.shape)?;
        self.mul(&scale)?.mul(&weight.broadcast_to(&self.shape)?)
    }
}
