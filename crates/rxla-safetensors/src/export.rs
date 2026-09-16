use super::*;

/// Save a new checkpoint without replacing any existing target, including a
/// symlink. Writes a temporary file in the target directory, synchronizes file
/// contents, then publishes with tempfile's no-clobber persistence. Ordinary
/// pre-publication errors remove the temporary file; buffers remain unchanged.
///
/// The directory must already exist and be trusted (no hostile path replacement).
/// No-overwrite is enforced again at publication, including competing writers.
/// Atomic publication is platform/filesystem dependent; tempfile may leave an
/// extra temporary hard link after a crash/cleanup failure. The parent directory
/// is not fsynced, so success does not promise survival of a power failure.
/// Use explicit versioned target names; there is no latest-pointer update,
/// retention policy, overwrite mode or distributed checkpoint coordination.
pub fn save_buffers_new(
    path: impl AsRef<Path>,
    buffers: &[(&str, &Buffer)],
    metadata: Option<&std::collections::HashMap<String, String>>,
) -> Result<()> {
    let path = path.as_ref();
    if path.file_name().is_none() {
        return Err(invalid("checkpoint target must have a file name"));
    }
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            return Err(io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "checkpoint target already exists",
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io(error)),
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(io)?;
    write_buffers_inner(temporary.as_file_mut(), buffers, metadata.cloned())?;
    temporary.as_file().sync_all().map_err(io)?;
    temporary
        .persist_noclobber(path)
        .map_err(|error| io(error.error))?;
    Ok(())
}

/// Stream explicitly named F32/I32/BF16 buffers as one safetensors file. Names must
/// be nonempty, unique and not `__metadata__`; entries are sorted by name.
/// Shapes/dtypes/header are checked before any write or download. Shared buffer
/// references under different names are saved separately, not deduplicated.
///
/// Downloads one tensor at a time; extra host storage is the largest tensor,
/// header, 64 KiB encoding scratch and caller writer buffering. No file is opened,
/// truncated, flushed, synchronized or atomically published. Failures may leave
/// partial output; discard it. Buffers are not mutated. The caller must provide
/// a coherent snapshot and all necessary state/configuration: this format does
/// not discover slots, encode executable code or guarantee training completeness.
pub fn write_buffers(writer: &mut dyn std::io::Write, buffers: &[(&str, &Buffer)]) -> Result<()> {
    write_buffers_inner(writer, buffers, None)
}

/// `write_buffers` with user-supplied safetensors string metadata, e.g. a versioned
/// model/optimizer description and input cursor. Included in the bounded header
/// and checked for size before output. Keys/values are opaque: no configuration
/// serialization, validation, secrets filtering or tensor/config consistency is
/// automatic. Metadata map serialization order is not a canonical hash format.
pub fn write_buffers_with_metadata(
    writer: &mut dyn std::io::Write,
    buffers: &[(&str, &Buffer)],
    metadata: &std::collections::HashMap<String, String>,
) -> Result<()> {
    write_buffers_inner(writer, buffers, Some(metadata.clone()))
}

fn write_buffers_inner(
    writer: &mut dyn std::io::Write,
    buffers: &[(&str, &Buffer)],
    user_metadata: Option<std::collections::HashMap<String, String>>,
) -> Result<()> {
    let mut buffers = buffers.to_vec();
    buffers.sort_by_key(|(name, _)| *name);
    let mut previous = None;
    let mut offset = 0usize;
    let mut metadata = Vec::with_capacity(buffers.len());
    let mut pending = Vec::with_capacity(buffers.len());
    for (name, buffer) in buffers {
        if name.is_empty() || name == "__metadata__" || previous == Some(name) {
            return Err(invalid(
                "export names must be nonempty, unique and not __metadata__",
            ));
        }
        previous = Some(name);
        let dtype = match buffer.dtype().map_err(|source| Error::Runtime { source })? {
            rxla_pjrt::DType::F32 => Dtype::F32,
            rxla_pjrt::DType::I32 => Dtype::I32,
            rxla_pjrt::DType::BF16 => Dtype::BF16,
            dtype => return Err(invalid(format!("unsupported export dtype {dtype:?}"))),
        };
        let shape = buffer
            .dimensions()
            .map_err(|source| Error::Runtime { source })?
            .into_iter()
            .map(|d| usize::try_from(d).map_err(|_| invalid("export dimension exceeds usize")))
            .collect::<Result<Vec<_>>>()?;
        let bytes = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .and_then(|n| n.checked_mul(if dtype == Dtype::BF16 { 2 } else { 4 }))
            .ok_or_else(|| invalid("export tensor size overflow"))?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| invalid("checkpoint byte count overflow"))?;
        metadata.push((
            name.to_owned(),
            TensorInfo {
                dtype,
                shape,
                data_offsets: (offset, end),
            },
        ));
        pending.push((dtype, buffer));
        offset = end;
    }
    let metadata = Metadata::new(user_metadata, metadata)
        .map_err(|source| invalid(format!("export metadata: {source}")))?;
    let mut header = serde_json::to_vec(&metadata).map_err(|source| Error::Header { source })?;
    let length = header
        .len()
        .checked_add(7)
        .map(|n| n / 8 * 8)
        .ok_or_else(|| invalid("export header size overflow"))?;
    if length > 100_000_000 {
        return Err(invalid("export header exceeds 100 MB limit"));
    }
    header.resize(length, b' ');
    writer
        .write_all(&(length as u64).to_le_bytes())
        .map_err(io)?;
    writer.write_all(&header).map_err(io)?;
    for (dtype, buffer) in pending {
        match dtype {
            Dtype::F32 => write_words(
                writer,
                buffer
                    .to_vec::<f32>()
                    .map_err(|source| Error::Runtime { source })?
                    .into_iter()
                    .map(f32::to_le_bytes),
            )?,
            Dtype::I32 => write_words(
                writer,
                buffer
                    .to_vec::<i32>()
                    .map_err(|source| Error::Runtime { source })?
                    .into_iter()
                    .map(i32::to_le_bytes),
            )?,
            Dtype::BF16 => write_words(
                writer,
                buffer
                    .to_vec::<half::bf16>()
                    .map_err(|source| Error::Runtime { source })?
                    .into_iter()
                    .map(|value| value.to_bits().to_le_bytes()),
            )?,
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn write_words<const N: usize>(
    writer: &mut dyn std::io::Write,
    values: impl Iterator<Item = [u8; N]>,
) -> Result<()> {
    let mut encoded = Vec::with_capacity(64 * 1024);
    for value in values {
        encoded.extend_from_slice(&value);
        if encoded.len() == 64 * 1024 {
            writer.write_all(&encoded).map_err(io)?;
            encoded.clear();
        }
    }
    if !encoded.is_empty() {
        writer.write_all(&encoded).map_err(io)?;
    }
    Ok(())
}
