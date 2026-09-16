//! SafeTensors loading for parameter-effect model schemas.

use super::*;
use rxla_nn::{ModelSessionBuilder, ParamSchema};
use std::collections::HashMap;

/// Weight buffers keyed by the stable paths in a parameter-effect schema.
///
/// The values are intentionally not reordered: pass borrowed `(path, buffer)`
/// pairs to `AppliedModel::bind`, which owns the authoritative ABI ordering.
pub struct SchemaBuffers {
    pub parameters: Vec<(String, Buffer)>,
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

    /// Initialize a resident model session directly from this checkpoint load.
    pub fn initialize<'model>(
        self,
        builder: ModelSessionBuilder<'model>,
    ) -> rxla_nn::Result<ModelSessionBuilder<'model>> {
        builder.parameters(self.into_parameters())
    }
}

impl<R: Read + Seek> SafeTensors<R> {
    /// Upload parameters whose checkpoint keys exactly match their effect-schema
    /// paths. This is the conventional safetensors layout for new RXLA models.
    pub fn load_parameter_schema(
        &mut self,
        client: &Client,
        schema: &ParamSchema,
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
        schema: &ParamSchema,
        mapping: &HashMap<String, String>,
    ) -> Result<SchemaBuffers> {
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
                DType::BF16 if info.dtype == Dtype::BF16 => {}
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

        let parameters = schema
            .parameters()
            .iter()
            .zip(sources)
            .map(|(spec, source)| {
                let buffer = match spec.dtype() {
                    DType::U8 => self.upload_u8(client, &source)?,
                    DType::F32 => self.upload_f32(client, &source)?,
                    DType::BF16 => self.upload_bf16_bits(client, &source)?,
                    dtype => return Err(invalid(format!("unsupported parameter dtype {dtype:?}"))),
                };
                Ok((spec.path().to_owned(), buffer))
            })
            .collect::<Result<_>>()?;
        Ok(SchemaBuffers { parameters })
    }
}
