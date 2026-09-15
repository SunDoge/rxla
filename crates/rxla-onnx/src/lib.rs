//! ONNX import boundary for RXLA.
//!
//! `onnx-ir` is deliberately private to this crate's public API. Parsed ONNX
//! graphs are transient frontend state; RXLA's Pliron SSA remains the compiler
//! IR consumed by optimization and lowering passes.

mod error;
mod import;
mod shape;

use std::{collections::BTreeMap, fs, path::Path};

pub use error::{Error, Result};
pub use import::{ImportedProgram, Parameter};
use protobuf::Message;
pub use shape::{ShapeMismatch, ShapeReport};
use snafu::ResultExt;

/// Parsed ONNX model waiting to be specialized and imported into RXLA IR.
pub struct Model {
    graph: onnx_ir::OnnxGraph,
}

/// Backend-independent inventory useful before attempting a lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSummary {
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub node_count: usize,
    pub operators: BTreeMap<String, usize>,
}

/// Public ONNX boundary metadata without exposing `onnx-ir` types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueInfo {
    pub name: String,
    /// `None` denotes a symbolic or otherwise unknown dimension.
    pub shape: Option<Vec<Option<usize>>>,
    pub dtype: String,
}

/// Concrete input dimensions used to specialize a dynamic ONNX model before
/// it enters RXLA's currently static-shape IR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

impl Model {
    /// Parse an ONNX file, including initializers stored as external data.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let graph =
            onnx_ir::OnnxGraphBuilder::new()
                .parse_file(path)
                .context(error::ParseSnafu {
                    path: path.to_owned(),
                })?;
        Ok(Self { graph })
    }

    /// Parse after replacing selected graph-input dimensions with concrete
    /// values. This also lets `onnx-ir` propagate those dimensions through its
    /// own shape inference before RXLA verifies them independently.
    ///
    /// This first implementation operates on an in-memory protobuf. Models
    /// using ONNX external-data files should use `load` until the upstream
    /// parser exposes specialization while retaining its source directory.
    pub fn load_specialized(path: impl AsRef<Path>, inputs: &[InputSpec]) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).context(error::ReadSnafu {
            path: path.to_owned(),
        })?;
        let mut model =
            onnx_ir::ModelProto::parse_from_bytes(&bytes).context(error::DecodeSnafu {
                path: path.to_owned(),
            })?;
        let graph = model.graph.as_mut().ok_or(Error::MissingGraph)?;
        for spec in inputs {
            let input = graph
                .input
                .iter_mut()
                .find(|input| input.name == spec.name)
                .ok_or_else(|| Error::MissingInput {
                    name: spec.name.clone(),
                })?;
            let tensor = input.type_.mut_or_insert_default().mut_tensor_type();
            let shape = tensor.shape.mut_or_insert_default();
            if shape.dim.len() != spec.shape.len() {
                return Err(Error::InputRank {
                    name: spec.name.clone(),
                    expected: spec.shape.len(),
                    actual: shape.dim.len(),
                });
            }
            for (dimension, &value) in shape.dim.iter_mut().zip(&spec.shape) {
                dimension.set_dim_value(value as i64);
            }
        }
        let bytes = model.write_to_bytes().context(error::EncodeSnafu)?;
        let graph = onnx_ir::OnnxGraphBuilder::new()
            .parse_bytes(&bytes)
            .context(error::ParseSnafu {
                path: path.to_owned(),
            })?;
        Ok(Self { graph })
    }

    pub fn summary(&self) -> ModelSummary {
        let mut operators = BTreeMap::new();
        for node in &self.graph.nodes {
            *operators.entry(node.node_type().to_string()).or_insert(0) += 1;
        }
        ModelSummary {
            inputs: self.graph.inputs.iter().map(value_info).collect(),
            outputs: self.graph.outputs.iter().map(value_info).collect(),
            node_count: self.graph.nodes.len(),
            operators,
        }
    }

    /// Recompute concrete tensor shapes from operator semantics instead of
    /// trusting ONNX value-info propagation.
    pub fn infer_shapes(&self) -> Result<ShapeReport> {
        shape::infer(&self.graph)
    }

    /// Import a specialized, static-shape model into RXLA's Pliron SSA.
    pub fn import(self) -> Result<ImportedProgram> {
        import::import(self.graph)
    }
}

fn value_info(argument: &onnx_ir::Argument) -> ValueInfo {
    use onnx_ir::ArgType;

    let (shape, dtype) = match &argument.ty {
        ArgType::Tensor(tensor) => (tensor.static_shape.clone(), format!("{:?}", tensor.dtype)),
        ArgType::ScalarTensor(dtype) => (Some(vec![Some(1)]), format!("{dtype:?}")),
        ArgType::ScalarNative(dtype) => (Some(Vec::new()), format!("{dtype:?}")),
        ArgType::Shape(rank) => (Some(vec![None; *rank]), "I64".to_owned()),
    };
    ValueInfo {
        name: argument.name.clone(),
        shape,
        dtype,
    }
}
