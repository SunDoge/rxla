//! Experimental tensor metadata envelope around trusted backend-native code.
use super::*;

#[derive(Clone, PartialEq, Message)]
struct Artifact {
    #[prost(uint32, tag = "1")]
    version: u32,
    #[prost(bytes = "vec", tag = "2")]
    native: Vec<u8>,
    #[prost(message, repeated, tag = "3")]
    inputs: Vec<rxla_xla_proto::xla::ShapeProto>,
    #[prost(uint64, tag = "4")]
    output_count: u64,
}

fn decode(bytes: &[u8]) -> Result<(Artifact, Vec<TensorType>, usize)> {
    let artifact =
        Artifact::decode(bytes).map_err(|e| err(format!("invalid tensor artifact: {e}")))?;
    if artifact.version != 1 || artifact.native.is_empty() || artifact.output_count == 0 {
        return Err(err("unsupported or incomplete tensor artifact"));
    }
    let output_count = usize::try_from(artifact.output_count)
        .map_err(|_| err("artifact output count overflow"))?;
    let inputs = artifact
        .inputs
        .iter()
        .map(|shape| {
            let dtype = match shape.element_type {
                x if x == PrimitiveType::U8 as i32 => DType::U8,
                x if x == PrimitiveType::F16 as i32 => DType::F16,
                x if x == PrimitiveType::F32 as i32 => DType::F32,
                x if x == PrimitiveType::S32 as i32 => DType::I32,
                x if x == PrimitiveType::Bf16 as i32 => DType::BF16,
                _ => return Err(err("unsupported artifact input dtype")),
            };
            elements(&shape.dimensions)?;
            if !shape.tuple_shapes.is_empty() || !shape.is_dynamic_dimension.is_empty() {
                return Err(err("artifact inputs must be static arrays"));
            }
            Ok(TensorType {
                dims: shape.dimensions.clone(),
                dtype,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((artifact, inputs, output_count))
}

impl Executable {
    /// Export native code together with tensor input metadata and output count.
    /// Unlike `serialize`, these bytes are not a raw PJRT artifact. This is an
    /// experimental versioned envelope, not a portable model or automatic cache.
    pub fn serialize_with_metadata(&self) -> Result<Vec<u8>> {
        let inputs = self
            .inputs
            .iter()
            .map(|input| {
                Ok(rxla_xla_proto::xla::ShapeProto {
                    element_type: match input.dtype {
                        DType::U8 => PrimitiveType::U8 as i32,
                        DType::F16 => PrimitiveType::F16 as i32,
                        DType::F32 => PrimitiveType::F32 as i32,
                        DType::I32 => PrimitiveType::S32 as i32,
                        DType::BF16 => PrimitiveType::Bf16 as i32,
                        dtype => return Err(err(format!("cannot serialize dtype {dtype:?}"))),
                    },
                    dimensions: input.dims.clone(),
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Artifact {
            version: 1,
            native: self.raw.serialize()?,
            inputs,
            output_count: self.output_count as u64,
        }
        .encode_to_vec())
    }

    /// Restore a tensor executable without rebuilding its graph or compiling HLO.
    /// Metadata is checked before invoking the native deserializer. Restored
    /// executables retain normal tensor input validation and own their client.
    ///
    /// # Safety
    /// The entire envelope must be trusted and unmodified, originally produced
    /// by `serialize_with_metadata`. Its metadata must describe the embedded code.
    /// The native artifact must be compatible with this exact plugin build,
    /// platform, device and execution environment. Version checks are not code
    /// authentication or backend compatibility checks. Never load untrusted cache
    /// entries; native deserialization may execute code. No fingerprint is verified.
    pub unsafe fn deserialize_with_metadata(client: &Client, bytes: &[u8]) -> Result<Self> {
        let (artifact, inputs, output_count) = decode(bytes)?;
        let raw = unsafe { client.deserialize_executable(&artifact.native)? };
        Ok(Self {
            raw,
            client: client.clone(),
            inputs,
            output_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn valid() -> Artifact {
        Artifact {
            version: 1,
            native: vec![1],
            output_count: 2,
            inputs: vec![rxla_xla_proto::xla::ShapeProto {
                element_type: PrimitiveType::F32 as i32,
                dimensions: vec![2, 3],
                ..Default::default()
            }],
        }
    }
    #[test]
    fn envelope_validation_without_native_loading() {
        let (_, inputs, count) = decode(&valid().encode_to_vec()).unwrap();
        assert_eq!(inputs[0].dims, [2, 3]);
        assert_eq!(count, 2);
        assert!(decode(&[0xff]).is_err());
        assert!(decode(&[]).is_err());
        for mutate in [
            (|a: &mut Artifact| a.version = 2) as fn(&mut Artifact),
            |a| a.native.clear(),
            |a| a.output_count = 0,
            |a| a.inputs[0].element_type = PrimitiveType::F64 as i32,
            |a| a.inputs[0].dimensions = vec![-1],
            |a| a.inputs[0].dimensions = vec![i64::MAX, i64::MAX],
            |a| a.inputs[0].is_dynamic_dimension = vec![true, false],
        ] {
            let mut artifact = valid();
            mutate(&mut artifact);
            assert!(decode(&artifact.encode_to_vec()).is_err());
        }
    }
}
