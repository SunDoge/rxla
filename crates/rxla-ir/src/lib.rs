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
        then_value: usize,
        else_marker: usize,
        else_value: usize,
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
