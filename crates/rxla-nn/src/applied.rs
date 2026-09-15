//! Applied model snapshots, compilation, and runtime argument binding.

use super::*;
use rxla_core::{Buffer, Compiler, DType, Executable, LoweredProgram};
use std::collections::BTreeMap;
use std::rc::Rc;

/// One apply trace plus its immutable Pliron program builder.
pub struct AppliedModel {
    graph: Graph,
    outputs: Vec<Tensor>,
    parameters: Vec<Tensor>,
    schema: ParamSchema,
}

/// Validated runtime buffers in the ABI order required by an [`AppliedModel`].
///
/// Construct this through [`AppliedModel::bind`], then pass
/// [`Self::as_slice`] to [`Executable::execute`].
pub struct ModelArguments<'a> {
    values: Vec<&'a Buffer>,
}

/// Path-validated parameters in schema order, reusable across executions.
pub struct BoundParameters<'model, 'parameters> {
    model: &'model AppliedModel,
    values: Vec<&'parameters Buffer>,
}

impl<'a> ModelArguments<'a> {
    pub fn as_slice(&self) -> &[&'a Buffer] {
        &self.values
    }
}

impl<'model, 'parameters> BoundParameters<'model, 'parameters> {
    /// Bind changing positional inputs without repeating parameter lookup or
    /// parameter buffer validation.
    pub fn bind<'inputs>(&self, inputs: &[&'inputs Buffer]) -> Result<ModelArguments<'inputs>>
    where
        'parameters: 'inputs,
    {
        self.model.bind_ordered(inputs, &self.values)
    }
}

impl AppliedModel {
    pub(crate) fn new(
        graph: Graph,
        outputs: Vec<Tensor>,
        parameters: Vec<Tensor>,
        schema: ParamSchema,
    ) -> Self {
        Self {
            graph,
            outputs,
            parameters,
            schema,
        }
    }

    pub fn outputs(&self) -> &[Tensor] {
        &self.outputs
    }

    /// The frozen effect schema used to produce this program.
    pub fn schema(&self) -> &ParamSchema {
        &self.schema
    }

    /// Clone the cheap graph handles for parameters selected for a transform.
    pub fn parameter_tensors(&self, selection: &ParameterSelection<'_>) -> Result<Vec<Tensor>> {
        ensure!(
            selection.schema() == &self.schema,
            SelectionSchemaMismatchSnafu
        );
        Ok(selection
            .ids()
            .iter()
            .map(|id| self.parameters[id.index()].clone())
            .collect())
    }

    /// Append an explicit runtime input for an IR transformation.
    ///
    /// Transform inputs follow the model ABI and are intentionally absent from
    /// [`ParamSchema`]: the transform that creates them owns their ordering and
    /// binding contract. This is suitable for optimizer state, not model data.
    pub fn transform_input(&self, shape: &[i64], dtype: DType) -> Result<Tensor> {
        Ok(self.graph.input_dtype(shape, dtype)?)
    }

    /// Lower model outputs while preserving the frozen schema ABI exactly.
    pub fn prepare(&self) -> Result<LoweredProgram> {
        Ok(self.graph.prepare_outputs(&self.outputs)?)
    }

    /// Compile this immutable model snapshot through the caller's cache-aware compiler.
    pub fn compile(&self, compiler: &mut Compiler) -> Result<Rc<Executable>> {
        Ok(compiler.compile_outputs(&self.graph, &self.outputs)?)
    }

    /// Lower graph-local outputs produced by a model transformation.
    pub fn prepare_tensors(&self, outputs: &[Tensor]) -> Result<LoweredProgram> {
        Ok(self.graph.prepare_outputs(outputs)?)
    }

    /// Compile graph-local outputs produced by a model transformation.
    pub fn compile_tensors(
        &self,
        compiler: &mut Compiler,
        outputs: &[Tensor],
    ) -> Result<Rc<Executable>> {
        Ok(compiler.compile_outputs(&self.graph, outputs)?)
    }

    /// Assemble positional inputs and path-addressed parameters into the model ABI.
    pub fn bind<'a>(
        &self,
        inputs: &[&'a Buffer],
        parameters: impl IntoIterator<Item = (&'a str, &'a Buffer)>,
    ) -> Result<ModelArguments<'a>> {
        self.validate_inputs(inputs)?;
        let parameters = self.order_parameters(parameters)?;
        self.assemble(inputs, &parameters)
    }

    /// Validate and order named parameter buffers once for repeated execution.
    pub fn bind_parameters<'model, 'parameters>(
        &'model self,
        parameters: impl IntoIterator<Item = (&'parameters str, &'parameters Buffer)>,
    ) -> Result<BoundParameters<'model, 'parameters>> {
        Ok(BoundParameters {
            model: self,
            values: self.order_parameters(parameters)?,
        })
    }

    fn order_parameters<'a>(
        &self,
        parameters: impl IntoIterator<Item = (&'a str, &'a Buffer)>,
    ) -> Result<Vec<&'a Buffer>> {
        let mut named = BTreeMap::new();
        for (path, buffer) in parameters {
            let spec = self
                .schema
                .get(path)
                .with_context(|| UnknownParameterSnafu { path })?;
            ensure!(
                named.insert(path, buffer).is_none(),
                DuplicateBindingSnafu { path }
            );
            validate_buffer(buffer, spec.shape(), spec.dtype(), "parameter", path)?;
        }
        self.schema
            .parameters
            .iter()
            .map(|spec| {
                named
                    .get(spec.path())
                    .copied()
                    .with_context(|| MissingBindingSnafu { path: spec.path() })
            })
            .collect()
    }

    fn bind_ordered<'a>(
        &self,
        inputs: &[&'a Buffer],
        parameters: &[&'a Buffer],
    ) -> Result<ModelArguments<'a>> {
        self.validate_inputs(inputs)?;
        self.assemble(inputs, parameters)
    }

    fn validate_inputs(&self, inputs: &[&Buffer]) -> Result<()> {
        ensure!(
            inputs.len() == self.schema.inputs.len(),
            InputCountSnafu {
                expected: self.schema.inputs.len(),
                actual: inputs.len(),
            }
        );
        for (index, (buffer, spec)) in inputs.iter().zip(&self.schema.inputs).enumerate() {
            validate_buffer(buffer, spec.shape(), spec.dtype(), "input", index)?;
        }
        Ok(())
    }

    fn assemble<'a>(
        &self,
        inputs: &[&'a Buffer],
        parameters: &[&'a Buffer],
    ) -> Result<ModelArguments<'a>> {
        ensure!(
            parameters.len() == self.schema.parameters.len(),
            ParameterCountSnafu {
                expected: self.schema.parameters.len(),
                actual: parameters.len(),
            }
        );
        let mut values = Vec::with_capacity(self.schema.arguments.len());
        for argument in &self.schema.arguments {
            match *argument {
                ModelArgument::Input(index) => values.push(inputs[index]),
                ModelArgument::Parameter(index) => values.push(parameters[index]),
            }
        }
        Ok(ModelArguments { values })
    }

    pub fn into_parts(self) -> (Graph, Vec<Tensor>) {
        (self.graph, self.outputs)
    }
}

fn validate_buffer(
    buffer: &Buffer,
    shape: &[i64],
    dtype: DType,
    kind: &'static str,
    identity: impl std::fmt::Display,
) -> Result<()> {
    ensure!(
        buffer.dimensions()? == shape,
        BufferShapeSnafu {
            kind,
            identity: identity.to_string(),
        }
    );
    ensure!(
        buffer.dtype()? == dtype,
        BufferDTypeSnafu {
            kind,
            identity: identity.to_string(),
        }
    );
    Ok(())
}
