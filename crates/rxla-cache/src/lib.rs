//! Storage primitives for immutable compilation artifacts.
//!
//! This crate deliberately knows nothing about PJRT, executable formats, cache
//! compatibility, or eviction policy. Those belong to the compiler using the
//! store. A future indexed backend can implement [`ArtifactStore`] without
//! changing compiler APIs.

use snafu::{ResultExt, Snafu};
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("{operation} {:?}: {source}", path))]
    Io {
        operation: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    #[snafu(display("artifact exceeds the configured entry limit"))]
    Oversized,
    #[snafu(display("artifact store requires a bounded positive entry limit"))]
    InvalidEntryLimit,
    #[snafu(display("artifact filename extension must be nonempty ASCII alphanumeric"))]
    InvalidExtension,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Collision-resistant, path-safe identity supplied by the compiler layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArtifactId([u8; 32]);

impl ArtifactId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn filename(self, extension: &str) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut name = String::with_capacity(67);
        for byte in self.0 {
            name.push(HEX[(byte >> 4) as usize] as char);
            name.push(HEX[(byte & 0xf) as usize] as char);
        }
        name.push('.');
        name.push_str(extension);
        name
    }

    fn from_filename(name: &str, extension: &str) -> Option<Self> {
        let hex = name.strip_suffix(extension)?.strip_suffix('.')?;
        if hex.len() != 64 {
            return None;
        }
        let mut bytes = [0; 32];
        let (pairs, remainder) = hex.as_bytes().as_chunks::<2>();
        debug_assert!(remainder.is_empty());
        for (output, pair) in bytes.iter_mut().zip(pairs) {
            *output = (nibble(pair[0])? << 4) | nibble(pair[1])?;
        }
        Some(Self(bytes))
    }
}

fn nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

/// Metadata needed by inspection and retention policies without reading data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactRecord {
    pub id: ArtifactId,
    pub encoded_bytes: u64,
    pub modified: SystemTime,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArtifactScan {
    pub records: Vec<ArtifactRecord>,
    pub ignored: u64,
}

/// Result of a bounded lookup. Missing records are normal cache misses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lookup {
    Hit(Vec<u8>),
    Missing,
    Oversized,
}

/// Synchronous storage contract for immutable, already-encoded artifacts.
pub trait ArtifactStore {
    fn lookup(&self, id: ArtifactId) -> Result<Lookup>;
    fn publish(&self, id: ArtifactId, bytes: &[u8]) -> Result<()>;
    fn remove(&self, id: ArtifactId) -> Result<bool>;
    fn scan(&self) -> Result<ArtifactScan>;
}

/// Flat-directory store with bounded reads and no-clobber atomic publication.
pub struct FileArtifactStore {
    directory: PathBuf,
    max_entry_bytes: usize,
    extension: String,
}

impl FileArtifactStore {
    pub fn new(directory: impl AsRef<Path>, max_entry_bytes: usize) -> Result<Self> {
        if max_entry_bytes == 0 || max_entry_bytes == usize::MAX {
            return Err(Error::InvalidEntryLimit);
        }
        std::fs::create_dir_all(directory.as_ref()).context(IoSnafu {
            operation: "create cache directory",
            path: directory.as_ref(),
        })?;
        let directory = std::fs::canonicalize(directory.as_ref()).context(IoSnafu {
            operation: "canonicalize cache directory",
            path: directory.as_ref(),
        })?;
        Ok(Self {
            directory,
            max_entry_bytes,
            extension: "bin".to_owned(),
        })
    }

    /// Select a path-safe filename extension. This supports stable formats
    /// owned by callers without exposing arbitrary relative paths.
    pub fn with_extension(mut self, extension: &str) -> Result<Self> {
        if extension.is_empty() || !extension.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
            return Err(Error::InvalidExtension);
        }
        self.extension = extension.to_owned();
        Ok(self)
    }

    pub fn path(&self, id: ArtifactId) -> PathBuf {
        self.directory.join(id.filename(&self.extension))
    }

    fn read_path(&self, path: &Path, encoded_bytes: u64) -> Result<Lookup> {
        if encoded_bytes > self.max_entry_bytes as u64 {
            return Ok(Lookup::Oversized);
        }
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Lookup::Missing);
            }
            Err(source) => {
                return Err(Error::Io {
                    operation: "open cache entry",
                    path: path.to_owned(),
                    source,
                });
            }
        };
        let mut bytes = Vec::new();
        file.take(self.max_entry_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .context(IoSnafu {
                operation: "read cache entry",
                path,
            })?;
        if bytes.len() > self.max_entry_bytes {
            Ok(Lookup::Oversized)
        } else {
            Ok(Lookup::Hit(bytes))
        }
    }
}

impl ArtifactStore for FileArtifactStore {
    fn lookup(&self, id: ArtifactId) -> Result<Lookup> {
        let path = self.path(id);
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Lookup::Missing);
            }
            Err(source) => {
                return Err(Error::Io {
                    operation: "inspect cache entry",
                    path,
                    source,
                });
            }
        };
        self.read_path(&path, metadata.len())
    }

    fn publish(&self, id: ArtifactId, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.max_entry_bytes {
            return Err(Error::Oversized);
        }
        let mut temporary = tempfile::NamedTempFile::new_in(&self.directory).context(IoSnafu {
            operation: "create temporary cache entry",
            path: &self.directory,
        })?;
        temporary
            .write_all(bytes)
            .and_then(|()| temporary.as_file().sync_all())
            .context(IoSnafu {
                operation: "write cache entry",
                path: temporary.path(),
            })?;
        match temporary.persist_noclobber(self.path(id)) {
            Ok(_) => Ok(()),
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(error) => Err(Error::Io {
                operation: "publish cache entry",
                path: error.file.path().to_owned(),
                source: error.error,
            }),
        }
    }

    fn remove(&self, id: ArtifactId) -> Result<bool> {
        let path = self.path(id);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(source) => Err(Error::Io {
                operation: "remove cache entry",
                path,
                source,
            }),
        }
    }

    fn scan(&self) -> Result<ArtifactScan> {
        let mut scan = ArtifactScan::default();
        for item in std::fs::read_dir(&self.directory).context(IoSnafu {
            operation: "scan cache directory",
            path: &self.directory,
        })? {
            let item = item.context(IoSnafu {
                operation: "read cache directory entry",
                path: &self.directory,
            })?;
            let file_type = item.file_type().context(IoSnafu {
                operation: "inspect cache directory entry",
                path: item.path(),
            })?;
            let id = item
                .file_name()
                .to_str()
                .and_then(|name| ArtifactId::from_filename(name, &self.extension));
            if !file_type.is_file() || id.is_none() {
                scan.ignored += 1;
                continue;
            }
            let metadata = item.metadata().context(IoSnafu {
                operation: "inspect cache entry",
                path: item.path(),
            })?;
            scan.records.push(ArtifactRecord {
                id: id.expect("checked above"),
                encoded_bytes: metadata.len(),
                modified: metadata.modified().context(IoSnafu {
                    operation: "read cache modification time",
                    path: item.path(),
                })?,
            });
        }
        Ok(scan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_store_is_bounded_atomic_and_path_safe() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileArtifactStore::new(directory.path(), 4).unwrap();
        let id = ArtifactId::new([0xab; 32]);
        assert_eq!(store.lookup(id).unwrap(), Lookup::Missing);
        store.publish(id, b"data").unwrap();
        store.publish(id, b"else").unwrap();
        assert_eq!(store.lookup(id).unwrap(), Lookup::Hit(b"data".to_vec()));
        assert_eq!(store.scan().unwrap().records.len(), 1);
        assert!(matches!(
            store.publish(ArtifactId::new([1; 32]), b"large"),
            Err(Error::Oversized)
        ));
        assert!(store.remove(id).unwrap());
        assert!(!store.remove(id).unwrap());
    }

    #[test]
    fn scan_ignores_unrelated_files_and_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileArtifactStore::new(directory.path(), 16).unwrap();
        std::fs::write(directory.path().join("unrelated"), b"x").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("unrelated", store.path(ArtifactId::new([2; 32]))).unwrap();
        let scan = store.scan().unwrap();
        assert!(scan.records.is_empty());
        assert_eq!(scan.ignored, if cfg!(unix) { 2 } else { 1 });
    }

    #[test]
    fn constructor_rejects_unbounded_or_empty_entries() {
        let directory = tempfile::tempdir().unwrap();
        assert!(matches!(
            FileArtifactStore::new(directory.path(), 0),
            Err(Error::InvalidEntryLimit)
        ));
        assert!(matches!(
            FileArtifactStore::new(directory.path(), usize::MAX),
            Err(Error::InvalidEntryLimit)
        ));
        assert!(matches!(
            FileArtifactStore::new(directory.path(), 1)
                .unwrap()
                .with_extension("../pb"),
            Err(Error::InvalidExtension)
        ));
    }
}
