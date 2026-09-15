//! Shared deployment-example checks; no compiler/frontend dependency.
use std::{fs::File, io::Read, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAX_PLUGIN_BYTES: u64 = 1024 * 1024 * 1024;

fn fingerprint_reader(reader: impl Read, limit: u64) -> Result<blake3::Hash> {
    let mut reader = reader.take(limit.checked_add(1).ok_or("invalid plugin byte limit")?);
    let mut buffer = [0u8; 64 * 1024];
    let mut hasher = blake3::Hasher::new();
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > limit {
            return Err("PJRT plugin exceeds byte limit".into());
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher.finalize())
}

/// Fingerprints a trusted, externally immutable regular file. This does not
/// bind dlopen to these bytes or fingerprint transitively loaded dependencies.
pub fn plugin_fingerprint(path: &Path) -> Result<blake3::Hash> {
    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > MAX_PLUGIN_BYTES {
        return Err("PJRT plugin must be a regular file of at most 1 GiB".into());
    }
    fingerprint_reader(File::open(path)?, MAX_PLUGIN_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fingerprint_is_streamed_bounded_and_reports_read_errors() {
        let bytes = vec![0xabu8; 200_000];
        assert_eq!(
            fingerprint_reader(bytes.as_slice(), bytes.len() as u64).unwrap(),
            blake3::hash(&bytes)
        );
        assert!(fingerprint_reader(bytes.as_slice(), bytes.len() as u64 - 1).is_err());
        assert_eq!(fingerprint_reader(&b""[..], 0).unwrap(), blake3::hash(b""));
        assert!(fingerprint_reader(&b"x"[..], 0).is_err());
        assert!(fingerprint_reader(&b""[..], u64::MAX).is_err());
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("read failure"))
            }
        }
        assert!(fingerprint_reader(Broken, 100).is_err());
    }
}
