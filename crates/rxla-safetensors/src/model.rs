//! SafeTensors loading for parameter-effect model schemas.

use super::*;
use rxla_nn::{CompiledModelSessionBuilder, ModelSchema, ParameterSelection};
use std::collections::HashMap;

/// Weight buffers keyed by the stable paths in a parameter-effect schema.
///
/// The values are intentionally not reordered: pass borrowed `(path, buffer)`
/// pairs to `AppliedModel::bind`, which owns the authoritative ABI ordering.
pub struct SchemaBuffers {
    pub parameters: Vec<(String, Buffer)>,
    pub states: Vec<(String, Buffer)>,
}

impl SchemaBuffers {
    /// Borrow named checkpoint buffers in the form accepted by
    /// `rxla_nn::AppliedModel::bind`.
    pub fn bindings(&self) -> Vec<(&str, &Buffer)> {
        self.parameters
            .iter()
            .map(|(path, buffer)| (path.as_str(), buffer))
            .collect()
    }

    /// Consume uploaded weights as canonical owned name/buffer pairs.
    pub fn into_parameters(self) -> impl ExactSizeIterator<Item = (String, Buffer)> {
        self.parameters.into_iter()
    }

    pub fn state_bindings(&self) -> Vec<(&str, &Buffer)> {
        self.states
            .iter()
            .map(|(path, buffer)| (path.as_str(), buffer))
            .collect()
    }

    /// Initialize a resident model session directly from this checkpoint load.
    pub fn initialize<'model>(
        self,
        builder: CompiledModelSessionBuilder<'model>,
    ) -> rxla_nn::Result<CompiledModelSessionBuilder<'model>> {
        let SchemaBuffers { parameters, states } = self;
        builder.parameters(parameters)?.states(states)
    }
}

impl<R: Read + Seek> SafeTensors<R> {
    /// Load an exact selected parameter subset, such as a standalone LoRA
    /// adapter checkpoint. Every selected path is validated before upload;
    /// tensors outside the selection are ignored.
    pub fn load_parameter_selection(
        &mut self,
        client: &Client,
        selection: &ParameterSelection,
    ) -> Result<SchemaBuffers> {
        for (_, spec) in selection.parameters() {
            let info = self
                .info(spec.path())
                .ok_or_else(|| invalid(format!("missing tensor {:?}", spec.path())))?;
            let shape = info
                .shape
                .iter()
                .map(|&dim| i64::try_from(dim).map_err(|_| invalid("dimension exceeds i64")))
                .collect::<Result<Vec<_>>>()?;
            if shape != spec.shape() {
                return Err(invalid(format!("tensor {:?}: shape mismatch", spec.path())));
            }
            match spec.dtype() {
                DType::U8 if info.dtype == Dtype::U8 => {}
                DType::F32 | DType::F16 | DType::BF16
                    if matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) => {}
                dtype => {
                    return Err(invalid(format!(
                        "tensor {:?}: incompatible checkpoint dtype for {dtype:?} storage",
                        spec.path()
                    )));
                }
            }
        }
        let parameters = selection
            .parameters()
            .map(|(_, spec)| {
                let source_dtype = self
                    .info(spec.path())
                    .ok_or_else(|| invalid(format!("missing tensor {:?}", spec.path())))?
                    .dtype;
                let buffer = match spec.dtype() {
                    DType::U8 => self.upload_u8(client, spec.path())?,
                    DType::F32 => self.upload_f32(client, spec.path())?,
                    DType::F16 => self.upload_f16(client, spec.path())?,
                    DType::BF16 if source_dtype == Dtype::BF16 => {
                        self.upload_bf16(client, spec.path())?
                    }
                    DType::BF16 => self.upload_bf16_converted(client, spec.path())?,
                    dtype => return Err(invalid(format!("unsupported parameter dtype {dtype:?}"))),
                };
                Ok((spec.path().to_owned(), buffer))
            })
            .collect::<Result<_>>()?;
        Ok(SchemaBuffers {
            parameters,
            states: Vec::new(),
        })
    }

    /// Validate an exact path/shape/dtype match without reading tensor payloads
    /// or requiring a PJRT client.
    pub fn validate_parameter_schema(&self, schema: &ModelSchema) -> Result<()> {
        let mapping = schema
            .parameters()
            .iter()
            .map(|spec| (spec.path().to_owned(), spec.path().to_owned()))
            .collect();
        self.validate_parameter_schema_with_mapping(schema, &mapping)
            .map(|_| ())
    }

    /// Upload parameters whose checkpoint keys exactly match their effect-schema
    /// paths. This is the conventional safetensors layout for new RXLA models.
    pub fn load_parameter_schema(
        &mut self,
        client: &Client,
        schema: &ModelSchema,
    ) -> Result<SchemaBuffers> {
        let mapping = schema
            .parameters()
            .iter()
            .map(|spec| (spec.path().to_owned(), spec.path().to_owned()))
            .collect();
        self.load_parameter_schema_with_mapping(client, schema, &mapping)
    }

    /// Upload parameters selected by stable schema path to explicit checkpoint
    /// keys. The mapping must be complete and contain no unknown schema paths.
    /// All headers are validated before the first payload read or device upload.
    pub fn load_parameter_schema_with_mapping(
        &mut self,
        client: &Client,
        schema: &ModelSchema,
        mapping: &HashMap<String, String>,
    ) -> Result<SchemaBuffers> {
        let sources = self.validate_parameter_schema_with_mapping(schema, mapping)?;

        let parameters = schema
            .parameters()
            .iter()
            .zip(sources)
            .map(|(spec, source)| {
                let source_dtype = self
                    .info(&source)
                    .ok_or_else(|| invalid(format!("missing tensor {source:?}")))?
                    .dtype;
                let buffer = match spec.dtype() {
                    DType::U8 => self.upload_u8(client, &source)?,
                    DType::F32 => self.upload_f32(client, &source)?,
                    DType::F16 => self.upload_f16(client, &source)?,
                    DType::BF16 if source_dtype == Dtype::BF16 => {
                        self.upload_bf16(client, &source)?
                    }
                    DType::BF16 => self.upload_bf16_converted(client, &source)?,
                    dtype => return Err(invalid(format!("unsupported parameter dtype {dtype:?}"))),
                };
                Ok((spec.path().to_owned(), buffer))
            })
            .collect::<Result<_>>()?;
        Ok(SchemaBuffers {
            parameters,
            states: Vec::new(),
        })
    }

    /// Upload every parameter and persistent state whose checkpoint key exactly
    /// matches its model-schema path. This is the preferred inference loader for
    /// stateful layers such as BatchNorm.
    pub fn load_model_schema(
        &mut self,
        client: &Client,
        schema: &ModelSchema,
    ) -> Result<SchemaBuffers> {
        self.validate_model_schema(schema)?;
        let parameters = self.load_parameter_schema(client, schema)?.parameters;
        let states = schema
            .states()
            .iter()
            .map(|spec| {
                Ok((
                    spec.path().to_owned(),
                    upload_typed(self, client, spec.path(), spec.dtype())?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SchemaBuffers { parameters, states })
    }

    /// Validate all parameter and persistent-state headers without reading or
    /// uploading payloads. Extra checkpoint tensors are permitted.
    pub fn validate_model_schema(&self, schema: &ModelSchema) -> Result<()> {
        self.validate_parameter_schema(schema)?;
        for spec in schema.states() {
            validate_tensor_info(self, spec.path(), spec.shape(), spec.dtype())?;
        }
        Ok(())
    }

    fn validate_parameter_schema_with_mapping(
        &self,
        schema: &ModelSchema,
        mapping: &HashMap<String, String>,
    ) -> Result<Vec<String>> {
        let mut sources = Vec::with_capacity(schema.parameters().len());
        for spec in schema.parameters() {
            let source = mapping
                .get(spec.path())
                .ok_or_else(|| invalid(format!("missing mapping for {:?}", spec.path())))?;
            let info = self
                .info(source)
                .ok_or_else(|| invalid(format!("missing tensor {source:?}")))?;
            let shape = info
                .shape
                .iter()
                .map(|&dim| i64::try_from(dim).map_err(|_| invalid("dimension exceeds i64")))
                .collect::<Result<Vec<_>>>()?;
            if shape != spec.shape() {
                return Err(invalid(format!(
                    "tensor {source:?}: shape mismatch for {:?}",
                    spec.path()
                )));
            }
            match spec.dtype() {
                DType::U8 if info.dtype == Dtype::U8 => {}
                DType::F32 if matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) => {}
                DType::F16 if matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) => {}
                DType::BF16 if matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) => {}
                dtype => {
                    return Err(invalid(format!(
                        "tensor {source:?}: incompatible checkpoint dtype for {dtype:?} storage"
                    )));
                }
            }
            sources.push(source.clone());
        }
        if mapping.len() != schema.parameters().len() {
            return Err(invalid(
                "mapping contains paths absent from parameter schema",
            ));
        }

        Ok(sources)
    }
}

fn validate_tensor_info<R: Read + Seek>(
    checkpoint: &SafeTensors<R>,
    path: &str,
    expected_shape: &[i64],
    expected_dtype: DType,
) -> Result<()> {
    let info = checkpoint
        .info(path)
        .ok_or_else(|| invalid(format!("missing tensor {path:?}")))?;
    let shape = info
        .shape
        .iter()
        .map(|&dimension| i64::try_from(dimension).map_err(|_| invalid("dimension exceeds i64")))
        .collect::<Result<Vec<_>>>()?;
    if shape != expected_shape {
        return Err(invalid(format!("tensor {path:?}: shape mismatch")));
    }
    let compatible = match expected_dtype {
        DType::U8 => info.dtype == Dtype::U8,
        DType::F32 | DType::F16 | DType::BF16 => {
            matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16)
        }
        _ => false,
    };
    if !compatible {
        return Err(invalid(format!(
            "tensor {path:?}: incompatible checkpoint dtype for {expected_dtype:?} storage"
        )));
    }
    Ok(())
}

fn upload_typed<R: Read + Seek>(
    checkpoint: &mut SafeTensors<R>,
    client: &Client,
    path: &str,
    dtype: DType,
) -> Result<Buffer> {
    let source_dtype = checkpoint
        .info(path)
        .ok_or_else(|| invalid(format!("missing tensor {path:?}")))?
        .dtype;
    match dtype {
        DType::U8 => checkpoint.upload_u8(client, path),
        DType::F32 => checkpoint.upload_f32(client, path),
        DType::F16 => checkpoint.upload_f16(client, path),
        DType::BF16 if source_dtype == Dtype::BF16 => checkpoint.upload_bf16(client, path),
        DType::BF16 => checkpoint.upload_bf16_converted(client, path),
        dtype => Err(invalid(format!("unsupported schema dtype {dtype:?}"))),
    }
}

/// Atomically save only `selection` from a live resident model session.
/// Optimizer slots, frozen base parameters, RNG and unrelated model state are
/// deliberately excluded.
#[cfg(feature = "export")]
pub fn save_parameter_selection_new(
    path: impl AsRef<std::path::Path>,
    model: &rxla_nn::AppliedModel,
    session: &rxla_core::Session,
    selection: &ParameterSelection,
    metadata: Option<&std::collections::HashMap<String, String>>,
) -> Result<()> {
    let buffers = model
        .selected_resident_parameter_buffers(session, selection)
        .map_err(|source| Error::Model { source })?;
    super::save_buffers_new(path, &buffers, metadata)
}
