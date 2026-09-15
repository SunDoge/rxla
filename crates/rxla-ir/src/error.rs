use crate::ShardingError;
use rxla_pjrt::DType;
use snafu::Snafu;

/// Failures produced while constructing, inspecting, or lowering tensor IR.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum IrError {
    #[snafu(display("invalid SSA value while {operation}"))]
    InvalidValue { operation: &'static str },
    #[snafu(display("malformed Pliron attribute {attribute}"))]
    MalformedAttribute { attribute: &'static str },
    #[snafu(display("{operation} requires a ranked tensor type"))]
    ExpectedTensorType { operation: &'static str },
    #[snafu(display("malformed constant payload: {reason}"))]
    MalformedConstant { reason: &'static str },
    #[snafu(display("{operation} does not support dtype {dtype:?}"))]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
    #[snafu(display("unsupported Pliron operation {operation}"))]
    UnsupportedOperation { operation: String },
    #[snafu(display("StableHLO dialect conversion failed: {message}"))]
    Conversion { message: String },
    #[snafu(display("{stage} IR verification failed: {message}"))]
    Verification {
        stage: &'static str,
        message: String,
    },
    #[snafu(display("tensor value already has a different sharding constraint"))]
    ShardingConflict,
    #[snafu(transparent)]
    Sharding { source: ShardingError },
}

pub type Result<T> = std::result::Result<T, IrError>;
