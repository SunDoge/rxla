//! Backend-independent typed tensor IR semantics.
//!
//! This crate owns the operation vocabulary shared by the tensor frontend and
//! IR implementations. It deliberately has no dependency on tracing, models,
//! Pliron, StableHLO, or PJRT clients.

use rxla_pjrt::DType;
use std::sync::Arc;

mod error;
pub use error::{IrError, Result};
mod pliron;
#[doc(hidden)]
pub use pliron::{LoweringTarget, ProgramIr, SemanticProgram, SsaId, StableHloProgram};
mod sharding;
pub use sharding::{Mesh, MeshAxis, PartitionSpec, Sharding, ShardingError};

/// One explicit placement attached to a compact semantic SSA value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardingConstraint {
    pub value: usize,
    pub sharding: Sharding,
}

/// Backend-neutral planning facts extracted from semantic SSA.
#[derive(Clone, Debug)]
pub struct PlanningFacts {
    input_count: usize,
    output_count: usize,
    value_count: usize,
    constraints: Vec<ShardingConstraint>,
}

impl PlanningFacts {
    pub fn new(
        input_count: usize,
        value_count: usize,
        output_count: usize,
        mut constraints: Vec<ShardingConstraint>,
    ) -> Self {
        constraints.sort_by_key(|constraint| constraint.value);
        Self {
            input_count,
            output_count,
            value_count,
            constraints,
        }
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    pub fn value_count(&self) -> usize {
        self.value_count
    }

    pub fn sharding_constraints(&self) -> &[ShardingConstraint] {
        &self.constraints
    }
}

/// Options for NHWC × HWIO convolution (cross-correlation, no kernel reversal).
#[derive(Clone, Copy, Debug)]
pub struct Conv2dOptions {
    pub strides: [i64; 2],
    /// `[[top, bottom], [left, right]]`, with nonnegative explicit padding.
    pub padding: [[i64; 2]; 2],
    pub dilation: [i64; 2],
    pub groups: i64,
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self {
            strides: [1, 1],
            padding: [[0, 0]; 2],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

/// Ungrouped NHWC transposed convolution with HWOI weights.
#[derive(Clone, Copy, Debug)]
pub struct ConvTranspose2dOptions {
    pub strides: [i64; 2],
    /// `[[top, bottom], [left, right]]` cropping of the full output.
    pub padding: [[i64; 2]; 2],
    pub dilation: [i64; 2],
    /// Additional high-end output extent; each value is below its stride.
    pub output_padding: [i64; 2],
}

impl Default for ConvTranspose2dOptions {
    fn default() -> Self {
        Self {
            strides: [1, 1],
            padding: [[0, 0]; 2],
            dilation: [1, 1],
            output_padding: [0, 0],
        }
    }
}

/// NHWC spatial pooling options.
#[derive(Clone, Copy, Debug)]
pub struct Pool2dOptions {
    pub window: [i64; 2],
    pub strides: [i64; 2],
    /// `[[top, bottom], [left, right]]`, with nonnegative explicit padding.
    pub padding: [[i64; 2]; 2],
}

impl Default for Pool2dOptions {
    fn default() -> Self {
        Self {
            window: [2, 2],
            strides: [2, 2],
            padding: [[0, 0]; 2],
        }
    }
}

/// Result type carried by every Pliron tensor SSA value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    pub dims: Vec<i64>,
    pub dtype: DType,
    /// Upper bounds for dynamic dimensions. An empty vector means a fully
    /// static shape; otherwise it is rank-sized, uses `-1` for static axes and
    /// a positive upper bound for every axis whose dimension is `-1`.
    pub dynamic_bounds: Vec<i64>,
}

impl TensorType {
    pub fn static_shape(dims: Vec<i64>, dtype: DType) -> Self {
        Self {
            dims,
            dtype,
            dynamic_bounds: Vec::new(),
        }
    }

    pub fn bounded(dims: Vec<i64>, dtype: DType, dynamic_bounds: Vec<i64>) -> Self {
        Self {
            dims,
            dtype,
            dynamic_bounds,
        }
    }

    pub fn bound(&self, axis: usize) -> Option<i64> {
        self.dynamic_bounds
            .get(axis)
            .copied()
            .filter(|&bound| bound >= 0)
    }

    fn normalized_bounds(mut bounds: Vec<i64>) -> Vec<i64> {
        if bounds.iter().all(|&bound| bound == -1) {
            bounds.clear();
        }
        bounds
    }

    /// Reorder axes while keeping every dynamic upper bound attached to its axis.
    pub fn permuted(&self, permutation: &[usize]) -> Option<Self> {
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        if sorted != (0..self.dims.len()).collect::<Vec<_>>() {
            return None;
        }
        let dims = permutation.iter().map(|&axis| self.dims[axis]).collect();
        let dynamic_bounds = if self.dynamic_bounds.is_empty() {
            Vec::new()
        } else {
            permutation
                .iter()
                .map(|&axis| self.dynamic_bounds[axis])
                .collect()
        };
        Some(Self {
            dims,
            dtype: self.dtype,
            dynamic_bounds,
        })
    }

    /// Remove reduction axes, or replace them with static singleton axes.
    pub fn reduced(&self, axes: &[usize], keepdims: bool) -> Option<Self> {
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        if sorted.iter().any(|&axis| axis >= self.dims.len())
            || sorted.windows(2).any(|pair| pair[0] == pair[1])
        {
            return None;
        }
        let mut dims = Vec::with_capacity(if keepdims {
            self.dims.len()
        } else {
            self.dims.len() - sorted.len()
        });
        let mut bounds = Vec::with_capacity(dims.capacity());
        for (axis, &dimension) in self.dims.iter().enumerate() {
            if sorted.binary_search(&axis).is_ok() {
                if keepdims {
                    dims.push(1);
                    bounds.push(-1);
                }
            } else {
                dims.push(dimension);
                bounds.push(self.bound(axis).unwrap_or(-1));
            }
        }
        Some(Self {
            dims,
            dtype: self.dtype,
            dynamic_bounds: Self::normalized_bounds(bounds),
        })
    }

    /// Insert one statically-sized axis without changing the other axes.
    pub fn inserted_axis(&self, axis: usize, size: i64) -> Option<Self> {
        if axis > self.dims.len() || size < 0 {
            return None;
        }
        let mut dims = self.dims.clone();
        dims.insert(axis, size);
        let mut bounds = if self.dynamic_bounds.is_empty() {
            vec![-1; self.dims.len()]
        } else {
            self.dynamic_bounds.clone()
        };
        bounds.insert(axis, -1);
        Some(Self {
            dims,
            dtype: self.dtype,
            dynamic_bounds: Self::normalized_bounds(bounds),
        })
    }

    /// Remove one statically-sized axis, preserving the remaining bounds.
    pub fn removed_axis(&self, axis: usize, expected_size: i64) -> Option<Self> {
        if self.dims.get(axis) != Some(&expected_size) {
            return None;
        }
        let mut dims = self.dims.clone();
        dims.remove(axis);
        let mut bounds = if self.dynamic_bounds.is_empty() {
            vec![-1; self.dims.len()]
        } else {
            self.dynamic_bounds.clone()
        };
        bounds.remove(axis);
        Some(Self {
            dims,
            dtype: self.dtype,
            dynamic_bounds: Self::normalized_bounds(bounds),
        })
    }

    /// Infer the result of concatenating compatible tensor types along one axis.
    /// A dynamic concatenation extent receives the sum of all operand maxima.
    pub fn concatenated(types: &[Self], axis: usize) -> Option<Self> {
        let first = types.first()?;
        if axis >= first.dims.len() {
            return None;
        }
        let mut axis_extent = 0_i64;
        let mut dynamic_axis = false;
        for ty in types {
            if ty.dtype != first.dtype || ty.dims.len() != first.dims.len() {
                return None;
            }
            for dimension in 0..first.dims.len() {
                if dimension != axis
                    && (ty.dims[dimension] != first.dims[dimension]
                        || ty.bound(dimension) != first.bound(dimension))
                {
                    return None;
                }
            }
            let extent = if ty.dims[axis] == -1 {
                dynamic_axis = true;
                ty.bound(axis)?
            } else {
                ty.dims[axis]
            };
            axis_extent = axis_extent.checked_add(extent)?;
        }

        let mut dims = first.dims.clone();
        dims[axis] = if dynamic_axis { -1 } else { axis_extent };
        let mut bounds = (0..dims.len())
            .map(|dimension| first.bound(dimension).unwrap_or(-1))
            .collect::<Vec<_>>();
        bounds[axis] = if dynamic_axis { axis_extent } else { -1 };
        Some(Self {
            dims,
            dtype: first.dtype,
            dynamic_bounds: Self::normalized_bounds(bounds),
        })
    }
}

/// Static half-open slice semantics retained as a typed Pliron attribute.
#[derive(Clone, Debug)]
pub struct SliceAxis {
    pub start: i64,
    pub limit: i64,
    pub stride: i64,
}

#[derive(Clone, Copy, Debug)]
pub enum Unary {
    Floor,
    Ceil,
    Round,
    RoundTiesEven,
    Sin,
    Cos,
    Erf,
    Exp,
    Log,
    Log1p,
    Expm1,
    Abs,
    Neg,
    Sqrt,
    Rsqrt,
    Tanh,
}

#[derive(Clone, Copy, Debug)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
}

#[derive(Clone, Copy, Debug)]
pub enum IntegerBinary {
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
}

#[derive(Clone, Copy, Debug)]
pub enum Comparison {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[derive(Clone, Copy, Debug)]
pub enum Reduction {
    Sum,
    Maximum,
}

/// Semantic operation descriptors accepted by the Pliron builder.
#[derive(Clone, Debug)]
pub enum Op {
    Parameter(usize),
    /// A named mutable value before state-effect discharge.
    StateInput {
        number: usize,
        state_id: usize,
        path: String,
    },
    /// Reads the current SSA version of a state value.
    StateRead {
        state_id: usize,
    },
    /// Produces the next SSA version of a state value.
    StateWrite {
        state_id: usize,
    },
    ConstantF32(Arc<[f32]>),
    ConstantI32(Arc<[i32]>),
    IndexToFloat,
    Bf16ToFloat,
    Convert {
        dtype: DType,
    },
    Iota {
        axis: usize,
    },
    GetDimensionSize {
        axis: usize,
    },
    Unary(Unary),
    Relu,
    Softplus,
    Sigmoid,
    Binary(Binary),
    IntegerBinary(IntegerBinary),
    Matmul {
        batch_rank: usize,
    },
    Attention {
        scale: f32,
    },
    Reshape,
    StopGradient,
    WithGradient,
    WithElementwiseDerivative,
    OptimizationBarrier,
    Broadcast {
        axes: Vec<usize>,
    },
    Transpose {
        permutation: Vec<usize>,
    },
    Reverse {
        axes: Vec<usize>,
    },
    Reduce {
        kind: Reduction,
        axes: Vec<usize>,
    },
    Cumsum {
        axis: usize,
    },
    Slice(Vec<SliceAxis>),
    SliceGradient(Vec<SliceAxis>),
    Pad(Vec<[i64; 2]>),
    DynamicSlice,
    Take {
        axis: usize,
    },
    TakeAlongAxis {
        axis: usize,
    },
    GatherGradient {
        axis: usize,
        batched: bool,
    },
    IndexLessEqualMask,
    IsFiniteMask,
    CompareMask(Comparison),
    Select,
    /// Structured conditional. Branch operation ranges precede this node in
    /// canonical SSA order and are moved into regions during construction.
    If {
        then_marker: usize,
        then_values: Vec<usize>,
        else_marker: usize,
        else_values: Vec<usize>,
        result_types: Vec<TensorType>,
    },
    /// Semantic placeholder for a non-leading result already created by the
    /// multi-result operation at `owner`.
    MultiResult {
        owner: usize,
        index: usize,
    },
    /// A backend-registered XLA FFI call with one tensor result.
    CustomCall {
        target: String,
        backend_config: String,
        has_side_effect: bool,
        api_version: i32,
        result_dtype: DType,
    },
    ArgMax {
        axis: usize,
    },
    SortedIndices {
        axis: usize,
        descending: bool,
    },
    DynamicUpdateSlice,
    Concatenate {
        axis: usize,
    },
    Conv2d(Conv2dOptions),
    Conv2dOihw(Conv2dOptions),
    ConvTranspose2d(ConvTranspose2dOptions),
    Conv2dInputGradient(Conv2dOptions),
    Conv2dKernelGradient(Conv2dOptions),
    MaxPool2d(Pool2dOptions),
    SumPool2d(Pool2dOptions),
}
