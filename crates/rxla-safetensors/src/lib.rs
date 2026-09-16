//! Bounded-header safetensors reader. Only the requested tensor payload is read.
//!
//! SafeTensors is the initial storage backend. Enable `model` to bind its
//! contents to parameter-effect `rxla_nn` schemas.
#[cfg(feature = "export")]
mod export;
#[cfg(feature = "export")]
pub use export::{save_buffers_new, write_buffers, write_buffers_with_metadata};
#[cfg(feature = "model")]
mod model;
use half::{bf16, f16};
#[cfg(feature = "model")]
pub use model::SchemaBuffers;
#[cfg(feature = "model")]
use rxla_pjrt::DType;
use rxla_pjrt::{Buffer, Client};
use safetensors::tensor::Metadata;
pub use safetensors::tensor::{Dtype, TensorInfo};
use snafu::Snafu;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::{Duration, Instant},
};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("weight file I/O: {source}"))]
    Io { source: std::io::Error },
    #[snafu(display("weight file header: {source}"))]
    Header { source: serde_json::Error },
    #[snafu(display("weight file: {message}"))]
    Invalid { message: String },
    #[snafu(display("weight upload: {source}"))]
    Runtime { source: rxla_pjrt::Error },
}
pub type Result<T> = std::result::Result<T, Error>;
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid {
        message: message.into(),
    }
}
fn io(source: std::io::Error) -> Error {
    Error::Io { source }
}

pub struct HostF32 {
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
}

/// Exact I32 tensor payload, suitable for counters and indexing state. No float
/// conversion is performed, including for integers outside F32's exact range.
pub struct HostI32 {
    pub shape: Vec<i64>,
    pub values: Vec<i32>,
}

/// Exact native BF16 storage. No F32 conversion or NaN canonicalization is
/// performed; callers can inspect encodings with [`bf16::to_bits`].
pub struct HostBf16 {
    pub shape: Vec<i64>,
    pub values: Vec<bf16>,
}

/// Exact U8 payload used by explicitly quantized parameter schemas.
pub struct HostU8 {
    pub shape: Vec<i64>,
    pub values: Vec<u8>,
}

/// Cumulative host wall times for successfully completed operations. Header
/// parsing is excluded. Reading includes seek/allocation; decoding includes typed
/// output allocation. Upload timing starts after decoding and includes the
/// synchronous PJRT upload call, not device-kernel profiling. Repeated reads and
/// uploads count again; failed operations do not increment their stage counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadStats {
    pub reads: u64,
    pub payload_bytes: u64,
    pub read_time: Duration,
    pub decode_time: Duration,
    pub uploads: u64,
    pub uploaded_bytes: u64,
    pub upload_time: Duration,
}

/// Single safetensors file/seekable source. No mmap or full-checkpoint allocation.
/// Keep the source unchanged while using it for a consistent weight snapshot.
pub struct SafeTensors<R = File> {
    reader: R,
    metadata: Metadata,
    data_start: u64,
    stats: LoadStats,
}
impl SafeTensors<File> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(File::open(path).map_err(io)?)
    }
}
impl<R: Read + Seek> SafeTensors<R> {
    pub fn new(mut reader: R) -> Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0)).map_err(io)?;
        reader.seek(SeekFrom::Start(0)).map_err(io)?;
        let mut prefix = [0u8; 8];
        reader.read_exact(&mut prefix).map_err(io)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > 100_000_000 {
            return Err(invalid("header exceeds 100 MB limit"));
        }
        let data_start = header_len
            .checked_add(8)
            .ok_or_else(|| invalid("header size overflow"))?;
        if data_start > file_len {
            return Err(invalid("truncated header"));
        }
        let mut header = vec![0; header_len as usize];
        reader.read_exact(&mut header).map_err(io)?;
        // Metadata's Deserialize implementation validates offsets, tensor byte
        // counts and shape overflow using the upstream safetensors implementation.
        let metadata: Metadata =
            serde_json::from_slice(&header).map_err(|source| Error::Header { source })?;
        if data_start.checked_add(metadata.data_len() as u64) != Some(file_len) {
            return Err(invalid("payload length does not match metadata"));
        }
        Ok(Self {
            reader,
            metadata,
            data_start,
            stats: LoadStats::default(),
        })
    }
    pub fn names(&self) -> Vec<String> {
        self.metadata.offset_keys()
    }
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.metadata.info(name)
    }
    /// User-defined safetensors string metadata, read during bounded header
    /// parsing. No model/configuration schema is inferred from these entries.
    pub fn metadata(&self) -> Option<&std::collections::HashMap<String, String>> {
        self.metadata.metadata().as_ref()
    }

    /// Require exact string values for a selected subset of metadata keys before
    /// loading payloads. Extra file keys are allowed; missing/mismatched keys or
    /// duplicate expectations fail. No payload I/O, uploads or state changes occur.
    /// This is caller-defined compatibility checking, not authentication or a
    /// guarantee that the values accurately describe the tensor payloads.
    pub fn require_metadata(&self, expected: &[(&str, &str)]) -> Result<()> {
        let mut seen = std::collections::HashSet::with_capacity(expected.len());
        for &(key, value) in expected {
            if !seen.insert(key) {
                return Err(invalid(format!("duplicate metadata expectation {key:?}")));
            }
            if self.metadata().and_then(|m| m.get(key)).map(String::as_str) != Some(value) {
                return Err(invalid(format!(
                    "missing or incompatible checkpoint metadata {key:?}"
                )));
            }
        }
        Ok(())
    }
    pub fn stats(&self) -> LoadStats {
        self.stats
    }

    /// Decode little-endian F32/F16/BF16 into owned F32 host values. This does not
    /// quantize or change layouts; source axis order is retained.
    pub fn read_f32(&mut self, name: &str) -> Result<HostF32> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| invalid(format!("missing tensor {name:?}")))?;
        if !matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
            return Err(invalid(format!(
                "tensor {name:?}: unsupported dtype {:?}",
                info.dtype
            )));
        }
        let shape = info
            .shape
            .iter()
            .map(|&d| i64::try_from(d).map_err(|_| invalid("dimension exceeds i64")))
            .collect::<Result<Vec<_>>>()?;
        let (start, end) = info.data_offsets;
        let offset = self
            .data_start
            .checked_add(start as u64)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        let read_start = Instant::now();
        self.reader.seek(SeekFrom::Start(offset)).map_err(io)?;
        let mut bytes = vec![0; end - start];
        self.reader.read_exact(&mut bytes).map_err(io)?;
        let read_time = read_start.elapsed();
        let decode_start = Instant::now();
        let values = match info.dtype {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| f16::from_bits(u16::from_le_bytes(b)).to_f32())
                .collect(),
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| bf16::from_bits(u16::from_le_bytes(b)).to_f32())
                .collect(),
            _ => unreachable!(),
        };
        self.stats.reads = self.stats.reads.saturating_add(1);
        self.stats.payload_bytes = self.stats.payload_bytes.saturating_add(bytes.len() as u64);
        self.stats.read_time = self.stats.read_time.saturating_add(read_time);
        self.stats.decode_time = self
            .stats
            .decode_time
            .saturating_add(decode_start.elapsed());
        Ok(HostF32 { shape, values })
    }

    /// Read an exact U8 tensor without integer or floating-point conversion.
    pub fn read_u8(&mut self, name: &str) -> Result<HostU8> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| invalid(format!("missing tensor {name:?}")))?;
        if info.dtype != Dtype::U8 {
            return Err(invalid(format!(
                "tensor {name:?}: expected U8, got {:?}",
                info.dtype
            )));
        }
        let shape = info
            .shape
            .iter()
            .map(|&d| i64::try_from(d).map_err(|_| invalid("dimension exceeds i64")))
            .collect::<Result<Vec<_>>>()?;
        let (start, end) = info.data_offsets;
        let offset = self
            .data_start
            .checked_add(start as u64)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        let read_start = Instant::now();
        self.reader.seek(SeekFrom::Start(offset)).map_err(io)?;
        let mut values = vec![0; end - start];
        self.reader.read_exact(&mut values).map_err(io)?;
        self.stats.reads = self.stats.reads.saturating_add(1);
        self.stats.payload_bytes = self.stats.payload_bytes.saturating_add(values.len() as u64);
        self.stats.read_time = self.stats.read_time.saturating_add(read_start.elapsed());
        Ok(HostU8 { shape, values })
    }
    /// Read only the selected I32 payload. Other integer widths and floating
    /// tensors are rejected, not cast. Shape/range metadata is validated before
    /// payload I/O. Successful stages contribute to the shared load statistics.
    pub fn read_i32(&mut self, name: &str) -> Result<HostI32> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| invalid(format!("missing tensor {name:?}")))?;
        if info.dtype != Dtype::I32 {
            return Err(invalid(format!(
                "tensor {name:?}: expected I32, got {:?}",
                info.dtype
            )));
        }
        let shape = info
            .shape
            .iter()
            .map(|&d| i64::try_from(d).map_err(|_| invalid("dimension exceeds i64")))
            .collect::<Result<Vec<_>>>()?;
        let (start, end) = info.data_offsets;
        let offset = self
            .data_start
            .checked_add(start as u64)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        let read_start = Instant::now();
        self.reader.seek(SeekFrom::Start(offset)).map_err(io)?;
        let mut bytes = vec![0; end - start];
        self.reader.read_exact(&mut bytes).map_err(io)?;
        let read_time = read_start.elapsed();
        let decode_start = Instant::now();
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| i32::from_le_bytes(b))
            .collect();
        self.stats.reads = self.stats.reads.saturating_add(1);
        self.stats.payload_bytes = self.stats.payload_bytes.saturating_add(bytes.len() as u64);
        self.stats.read_time = self.stats.read_time.saturating_add(read_time);
        self.stats.decode_time = self
            .stats
            .decode_time
            .saturating_add(decode_start.elapsed());
        Ok(HostI32 { shape, values })
    }

    /// Read the selected payload as native BF16 values. Other dtypes, including
    /// F16 and F32, are rejected before payload I/O, not converted.
    pub fn read_bf16(&mut self, name: &str) -> Result<HostBf16> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| invalid(format!("missing tensor {name:?}")))?;
        if info.dtype != Dtype::BF16 {
            return Err(invalid(format!(
                "tensor {name:?}: expected BF16, got {:?}",
                info.dtype
            )));
        }
        let shape = info
            .shape
            .iter()
            .map(|&d| i64::try_from(d).map_err(|_| invalid("dimension exceeds i64")))
            .collect::<Result<Vec<_>>>()?;
        let (start, end) = info.data_offsets;
        let offset = self
            .data_start
            .checked_add(start as u64)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        let read_start = Instant::now();
        self.reader.seek(SeekFrom::Start(offset)).map_err(io)?;
        let mut bytes = vec![0; end - start];
        self.reader.read_exact(&mut bytes).map_err(io)?;
        let read_time = read_start.elapsed();
        let decode_start = Instant::now();
        let values = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&bytes| bf16::from_bits(u16::from_le_bytes(bytes)))
            .collect();
        self.stats.reads = self.stats.reads.saturating_add(1);
        self.stats.payload_bytes = self.stats.payload_bytes.saturating_add(bytes.len() as u64);
        self.stats.read_time = self.stats.read_time.saturating_add(read_time);
        self.stats.decode_time = self
            .stats
            .decode_time
            .saturating_add(decode_start.elapsed());
        Ok(HostBf16 { shape, values })
    }

    /// Upload BF16 without expanding to F32; uploaded byte statistics count two
    /// bytes per element. Does not mutate sessions or infer model key mappings.
    pub fn upload_bf16(&mut self, client: &Client, name: &str) -> Result<Buffer> {
        let tensor = self.read_bf16(name)?;
        let upload_start = Instant::now();
        let buffer = client
            .buffer(&tensor.shape, &tensor.values)
            .map_err(|source| Error::Runtime { source })?;
        self.stats.uploads = self.stats.uploads.saturating_add(1);
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .saturating_add((tensor.values.len() as u64).saturating_mul(2));
        self.stats.upload_time = self
            .stats
            .upload_time
            .saturating_add(upload_start.elapsed());
        Ok(buffer)
    }

    /// Upload exact U8 quantized storage without host-side dequantization.
    pub fn upload_u8(&mut self, client: &Client, name: &str) -> Result<Buffer> {
        let tensor = self.read_u8(name)?;
        let upload_start = Instant::now();
        let buffer = client
            .buffer(&tensor.shape, &tensor.values)
            .map_err(|source| Error::Runtime { source })?;
        self.stats.uploads = self.stats.uploads.saturating_add(1);
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .saturating_add(tensor.values.len() as u64);
        self.stats.upload_time = self
            .stats
            .upload_time
            .saturating_add(upload_start.elapsed());
        Ok(buffer)
    }

    /// Upload an exact I32 tensor to PJRT without mutating any session. This does
    /// not auto-discover state names or establish checkpoint/graph compatibility.
    pub fn upload_i32(&mut self, client: &Client, name: &str) -> Result<Buffer> {
        let tensor = self.read_i32(name)?;
        let upload_start = Instant::now();
        let buffer = client
            .buffer(&tensor.shape, &tensor.values)
            .map_err(|source| Error::Runtime { source })?;
        self.stats.uploads = self.stats.uploads.saturating_add(1);
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .saturating_add((tensor.values.len() as u64).saturating_mul(4));
        self.stats.upload_time = self
            .stats
            .upload_time
            .saturating_add(upload_start.elapsed());
        Ok(buffer)
    }

    pub fn upload_f32(&mut self, client: &Client, name: &str) -> Result<Buffer> {
        let tensor = self.read_f32(name)?;
        let upload_start = Instant::now();
        let buffer = client
            .buffer(&tensor.shape, &tensor.values)
            .map_err(|source| Error::Runtime { source })?;
        self.stats.uploads = self.stats.uploads.saturating_add(1);
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .saturating_add((tensor.values.len() as u64).saturating_mul(4));
        self.stats.upload_time = self
            .stats
            .upload_time
            .saturating_add(upload_start.elapsed());
        Ok(buffer)
    }
}
