use std::path::PathBuf;

use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("failed to read ONNX model `{}`: {source}", path.display()))]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[snafu(display("failed to decode ONNX protobuf `{}`: {source}", path.display()))]
    Decode {
        path: PathBuf,
        source: protobuf::Error,
    },
    #[snafu(display("failed to encode specialized ONNX protobuf: {source}"))]
    Encode { source: protobuf::Error },
    #[snafu(display("failed to import ONNX model `{}`: {source}", path.display()))]
    Parse {
        path: PathBuf,
        source: onnx_ir::Error,
    },
    #[snafu(display("ONNX model has no graph"))]
    MissingGraph,
    #[snafu(display("ONNX graph has no input named `{name}`"))]
    MissingInput { name: String },
    #[snafu(display(
        "ONNX input `{name}` has rank {actual}, but specialization supplied rank {expected}"
    ))]
    InputRank {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[snafu(display("cannot infer `{node}` ({operator}): missing shape for `{value}`"))]
    MissingShape {
        node: String,
        operator: String,
        value: String,
    },
    #[snafu(display("cannot infer `{node}` ({operator}): {message}"))]
    ShapeInference {
        node: String,
        operator: String,
        message: String,
    },
    #[snafu(display("shape inference for ONNX operator `{operator}` is not implemented"))]
    UnsupportedShapeOperator { operator: String },
    #[snafu(display("cannot import ONNX operator `{operator}` into RXLA IR"))]
    UnsupportedOperator { operator: String },
    #[snafu(display("cannot import `{node}` ({operator}): {message}"))]
    Import {
        node: String,
        operator: String,
        message: String,
    },
    #[snafu(display("RXLA IR rejected imported ONNX operation: {source}"))]
    Ir { source: rxla_ir::IrError },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
