use snafu::Snafu;
use std::collections::HashSet;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum ShardingError {
    #[snafu(display("mesh axis name cannot be empty"))]
    EmptyMeshAxisName,
    #[snafu(display("mesh axis {name:?} must have nonzero size"))]
    ZeroMeshAxisSize { name: String },
    #[snafu(display("mesh axis {name:?} is declared more than once"))]
    DuplicateMeshAxis { name: String },
    #[snafu(display("mesh device count overflows usize"))]
    DeviceCountOverflow,
    #[snafu(display("partition axis name cannot be empty"))]
    EmptyPartitionAxisName,
    #[snafu(display("{field} cannot be represented by the sharding IR encoding"))]
    EncodingOverflow { field: &'static str },
    #[snafu(display("unsupported sharding attribute version {version}"))]
    UnsupportedEncodingVersion { version: u8 },
    #[snafu(display("invalid {field} tag {value} in sharding attribute"))]
    InvalidEncodingTag { field: &'static str, value: u8 },
    #[snafu(display("{field} in sharding attribute overflows usize"))]
    DecodedValueOverflow { field: &'static str },
    #[snafu(display("trailing bytes in sharding attribute"))]
    TrailingBytes,
    #[snafu(display("partition spec rank {spec_rank} does not match tensor rank {tensor_rank}"))]
    RankMismatch {
        spec_rank: usize,
        tensor_rank: usize,
    },
    #[snafu(display("mesh axis {name:?} cannot partition multiple tensor dimensions"))]
    ReusedMeshAxis { name: String },
    #[snafu(display("partition spec references unknown mesh axis {name:?}"))]
    UnknownMeshAxis { name: String },
    #[snafu(display(
        "tensor dimension {dimension} is not divisible by mesh axis {axis:?} of size {axis_size}"
    ))]
    IndivisibleDimension {
        dimension: i64,
        axis: String,
        axis_size: usize,
    },
    #[snafu(display("truncated sharding attribute"))]
    TruncatedEncoding,
    #[snafu(display("sharding attribute contains invalid UTF-8"))]
    InvalidUtf8,
}

type Result<T> = std::result::Result<T, ShardingError>;

/// A named logical device mesh. Physical devices are assigned later by a planner.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Mesh {
    axes: Vec<MeshAxis>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshAxis {
    name: String,
    size: usize,
}

impl Mesh {
    pub fn new<const N: usize>(axes: [(&str, usize); N]) -> Result<Self> {
        let mut names = HashSet::with_capacity(N);
        let mut owned = Vec::with_capacity(N);
        for (name, size) in axes {
            if name.is_empty() {
                return Err(ShardingError::EmptyMeshAxisName);
            }
            if size == 0 {
                return Err(ShardingError::ZeroMeshAxisSize { name: name.into() });
            }
            if !names.insert(name) {
                return Err(ShardingError::DuplicateMeshAxis { name: name.into() });
            }
            owned.push(MeshAxis {
                name: name.into(),
                size,
            });
        }
        Ok(Self { axes: owned })
    }

    pub fn axes(&self) -> &[MeshAxis] {
        &self.axes
    }

    pub fn device_count(&self) -> Result<usize> {
        self.axes.iter().try_fold(1usize, |count, axis| {
            count
                .checked_mul(axis.size)
                .ok_or(ShardingError::DeviceCountOverflow)
        })
    }

    fn axis_size(&self, name: &str) -> Option<usize> {
        self.axes
            .iter()
            .find(|axis| axis.name == name)
            .map(|axis| axis.size)
    }
}

impl MeshAxis {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

/// Maps each tensor dimension to at most one named mesh axis. `None` replicates
/// that dimension. Rank is checked when attached to a Tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartitionSpec(Vec<Option<String>>);

impl PartitionSpec {
    pub fn new<const N: usize>(axes: [Option<&str>; N]) -> Result<Self> {
        if axes.iter().flatten().any(|name| name.is_empty()) {
            return Err(ShardingError::EmptyPartitionAxisName);
        }
        Ok(Self(
            axes.into_iter()
                .map(|axis| axis.map(str::to_owned))
                .collect(),
        ))
    }

    pub fn axes(&self) -> &[Option<String>] {
        &self.0
    }
}

/// A logical placement constraint, independent of PJRT clients and buffers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Sharding {
    Replicated { mesh: Mesh },
    Partitioned { mesh: Mesh, spec: PartitionSpec },
}

impl Sharding {
    pub fn replicated(mesh: Mesh) -> Self {
        Self::Replicated { mesh }
    }

    pub fn partitioned(mesh: Mesh, spec: PartitionSpec) -> Self {
        Self::Partitioned { mesh, spec }
    }

    pub fn mesh(&self) -> &Mesh {
        match self {
            Self::Replicated { mesh } | Self::Partitioned { mesh, .. } => mesh,
        }
    }

    #[doc(hidden)]
    pub fn encode_ir(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![1, u8::from(matches!(self, Self::Partitioned { .. }))];
        encode_len(&mut bytes, self.mesh().axes.len())?;
        for axis in &self.mesh().axes {
            encode_string(&mut bytes, &axis.name)?;
            bytes.extend_from_slice(
                &u64::try_from(axis.size)
                    .map_err(|_| ShardingError::EncodingOverflow {
                        field: "mesh axis size",
                    })?
                    .to_le_bytes(),
            );
        }
        if let Self::Partitioned { spec, .. } = self {
            encode_len(&mut bytes, spec.0.len())?;
            for axis in &spec.0 {
                bytes.push(u8::from(axis.is_some()));
                if let Some(axis) = axis {
                    encode_string(&mut bytes, axis)?;
                }
            }
        }
        Ok(bytes)
    }

    #[doc(hidden)]
    pub fn decode_ir(bytes: &[u8]) -> Result<Self> {
        let mut decoder = Decoder::new(bytes);
        let version = decoder.byte()?;
        if version != 1 {
            return Err(ShardingError::UnsupportedEncodingVersion { version });
        }
        let partitioned = match decoder.byte()? {
            0 => false,
            1 => true,
            value => {
                return Err(ShardingError::InvalidEncodingTag {
                    field: "sharding kind",
                    value,
                });
            }
        };
        let axis_count = decoder.len()?;
        let mut axes = Vec::with_capacity(axis_count);
        let mut axis_names = HashSet::with_capacity(axis_count);
        for _ in 0..axis_count {
            let name = decoder.string()?;
            let size = usize::try_from(decoder.u64()?).map_err(|_| {
                ShardingError::DecodedValueOverflow {
                    field: "mesh axis size",
                }
            })?;
            if name.is_empty() {
                return Err(ShardingError::EmptyMeshAxisName);
            }
            if size == 0 {
                return Err(ShardingError::ZeroMeshAxisSize { name });
            }
            if !axis_names.insert(name.clone()) {
                return Err(ShardingError::DuplicateMeshAxis { name });
            }
            axes.push(MeshAxis { name, size });
        }
        let mesh = Mesh { axes };
        let sharding = if partitioned {
            let spec_count = decoder.len()?;
            let mut spec = Vec::with_capacity(spec_count);
            let mut used = HashSet::new();
            for _ in 0..spec_count {
                let axis = match decoder.byte()? {
                    0 => None,
                    1 => Some(decoder.string()?),
                    value => {
                        return Err(ShardingError::InvalidEncodingTag {
                            field: "partition axis",
                            value,
                        });
                    }
                };
                if let Some(name) = axis.as_ref() {
                    if name.is_empty() {
                        return Err(ShardingError::EmptyPartitionAxisName);
                    }
                    if !axis_names.contains(name) {
                        return Err(ShardingError::UnknownMeshAxis { name: name.clone() });
                    }
                    if !used.insert(name.clone()) {
                        return Err(ShardingError::ReusedMeshAxis { name: name.clone() });
                    }
                }
                spec.push(axis);
            }
            Self::Partitioned {
                mesh,
                spec: PartitionSpec(spec),
            }
        } else {
            Self::Replicated { mesh }
        };
        if !decoder.remaining().is_empty() {
            return Err(ShardingError::TrailingBytes);
        }
        Ok(sharding)
    }

    #[doc(hidden)]
    pub fn validate_shape(&self, shape: &[i64]) -> Result<()> {
        let Self::Partitioned { mesh, spec } = self else {
            return Ok(());
        };
        if spec.0.len() != shape.len() {
            return Err(ShardingError::RankMismatch {
                spec_rank: spec.0.len(),
                tensor_rank: shape.len(),
            });
        }
        let mut used = HashSet::new();
        for (&dimension, axis) in shape.iter().zip(&spec.0) {
            let Some(axis) = axis else { continue };
            if !used.insert(axis) {
                return Err(ShardingError::ReusedMeshAxis { name: axis.clone() });
            }
            let size = mesh
                .axis_size(axis)
                .ok_or_else(|| ShardingError::UnknownMeshAxis { name: axis.clone() })?;
            if dimension < 0 || !(dimension as usize).is_multiple_of(size) {
                return Err(ShardingError::IndivisibleDimension {
                    dimension,
                    axis: axis.clone(),
                    axis_size: size,
                });
            }
        }
        Ok(())
    }
}

fn encode_len(bytes: &mut Vec<u8>, len: usize) -> Result<()> {
    bytes.extend_from_slice(
        &u64::try_from(len)
            .map_err(|_| ShardingError::EncodingOverflow {
                field: "attribute length",
            })?
            .to_le_bytes(),
    );
    Ok(())
}

fn encode_string(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    encode_len(bytes, value.len())?;
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

struct Decoder<'a>(&'a [u8]);

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }

    fn remaining(&self) -> &'a [u8] {
        self.0
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let (value, rest) = self
            .0
            .split_at_checked(len)
            .ok_or(ShardingError::TruncatedEncoding)?;
        self.0 = rest;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn len(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| ShardingError::DecodedValueOverflow {
            field: "attribute length",
        })
    }

    fn string(&mut self) -> Result<String> {
        let len = self.len()?;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| ShardingError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_mesh_and_partition_contracts_without_devices() {
        let mesh = Mesh::new([("data", 2), ("model", 4)]).unwrap();
        assert_eq!(mesh.device_count().unwrap(), 8);
        let sharding = Sharding::partitioned(
            mesh.clone(),
            PartitionSpec::new([Some("data"), Some("model")]).unwrap(),
        );
        assert!(sharding.validate_shape(&[8, 16]).is_ok());
        assert!(sharding.validate_shape(&[7, 16]).is_err());
        assert!(
            Sharding::partitioned(
                mesh,
                PartitionSpec::new([Some("data"), Some("data")]).unwrap()
            )
            .validate_shape(&[8, 16])
            .is_err()
        );
        assert!(matches!(
            Mesh::new([("data", 2), ("data", 4)]),
            Err(ShardingError::DuplicateMeshAxis { name }) if name == "data"
        ));
        assert!(matches!(
            Mesh::new([("data", 0)]),
            Err(ShardingError::ZeroMeshAxisSize { name }) if name == "data"
        ));
    }

    #[test]
    fn sharding_ir_encoding_round_trips() {
        let sharding = Sharding::partitioned(
            Mesh::new([("data", 2), ("model", 4)]).unwrap(),
            PartitionSpec::new([Some("data"), None, Some("model")]).unwrap(),
        );
        assert_eq!(
            Sharding::decode_ir(&sharding.encode_ir().unwrap()).unwrap(),
            sharding
        );
        assert!(matches!(
            Sharding::decode_ir(&[1, 1]),
            Err(ShardingError::TruncatedEncoding)
        ));

        let mut zero_axis = vec![1, 0];
        zero_axis.extend_from_slice(&1_u64.to_le_bytes());
        zero_axis.extend_from_slice(&4_u64.to_le_bytes());
        zero_axis.extend_from_slice(b"data");
        zero_axis.extend_from_slice(&0_u64.to_le_bytes());
        assert!(matches!(
            Sharding::decode_ir(&zero_axis),
            Err(ShardingError::ZeroMeshAxisSize { name }) if name == "data"
        ));
    }
}
