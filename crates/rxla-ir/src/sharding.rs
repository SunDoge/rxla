use snafu::Snafu;
use std::collections::HashSet;

#[derive(Debug, Snafu)]
#[snafu(display("invalid sharding: {message}"))]
pub struct ShardingError {
    message: String,
}

type Result<T> = std::result::Result<T, ShardingError>;

fn err(message: impl Into<String>) -> ShardingError {
    ShardingError {
        message: message.into(),
    }
}

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
            if name.is_empty() || size == 0 || !names.insert(name) {
                return Err(err(
                    "mesh axes require unique nonempty names and nonzero sizes",
                ));
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
                .ok_or_else(|| err("mesh device count overflows usize"))
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
            return Err(err("partition axis names cannot be empty"));
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
                    .map_err(|_| err("mesh axis size cannot be encoded"))?
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
        if decoder.byte()? != 1 {
            return Err(err("unsupported sharding attribute version"));
        }
        let partitioned = match decoder.byte()? {
            0 => false,
            1 => true,
            _ => return Err(err("invalid sharding attribute kind")),
        };
        let axis_count = decoder.len()?;
        let mut axes = Vec::with_capacity(axis_count);
        for _ in 0..axis_count {
            let name = decoder.string()?;
            let size = usize::try_from(decoder.u64()?)
                .map_err(|_| err("mesh axis size overflows usize"))?;
            axes.push(MeshAxis { name, size });
        }
        let mesh = Mesh { axes };
        let sharding = if partitioned {
            let spec_count = decoder.len()?;
            let mut spec = Vec::with_capacity(spec_count);
            for _ in 0..spec_count {
                spec.push(match decoder.byte()? {
                    0 => None,
                    1 => Some(decoder.string()?),
                    _ => return Err(err("invalid partition axis tag")),
                });
            }
            Self::Partitioned {
                mesh,
                spec: PartitionSpec(spec),
            }
        } else {
            Self::Replicated { mesh }
        };
        if !decoder.remaining().is_empty() {
            return Err(err("trailing bytes in sharding attribute"));
        }
        Ok(sharding)
    }

    #[doc(hidden)]
    pub fn validate_shape(&self, shape: &[i64]) -> Result<()> {
        let Self::Partitioned { mesh, spec } = self else {
            return Ok(());
        };
        if spec.0.len() != shape.len() {
            return Err(err("partition spec rank does not match tensor rank"));
        }
        let mut used = HashSet::new();
        for (&dimension, axis) in shape.iter().zip(&spec.0) {
            let Some(axis) = axis else { continue };
            if !used.insert(axis) {
                return Err(err(
                    "a mesh axis cannot partition multiple tensor dimensions",
                ));
            }
            let size = mesh
                .axis_size(axis)
                .ok_or_else(|| err("partition spec references an unknown mesh axis"))?;
            if dimension < 0 || !(dimension as usize).is_multiple_of(size) {
                return Err(err("tensor dimension is not divisible by its mesh axis"));
            }
        }
        Ok(())
    }
}

fn encode_len(bytes: &mut Vec<u8>, len: usize) -> Result<()> {
    bytes.extend_from_slice(
        &u64::try_from(len)
            .map_err(|_| err("sharding attribute length cannot be encoded"))?
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
            .ok_or_else(|| err("truncated sharding attribute"))?;
        self.0 = rest;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn len(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| err("sharding attribute length overflows usize"))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.len()?;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| err("sharding attribute contains invalid UTF-8"))
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
        assert!(Mesh::new([("data", 2), ("data", 4)]).is_err());
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
        assert!(Sharding::decode_ir(&[1, 1]).is_err());
    }
}
