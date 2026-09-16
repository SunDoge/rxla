//! Opt-in trusted native cache. Checksums detect corruption, not malicious code.
use super::*;
use rxla_cache::{ArtifactId, ArtifactStore, FileArtifactStore, Lookup};
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

#[derive(Clone, PartialEq, Message)]
struct Entry {
    #[prost(uint32, tag = "1")]
    version: u32,
    #[prost(string, tag = "2")]
    namespace: String,
    #[prost(bytes = "vec", tag = "3")]
    key: Vec<u8>,
    #[prost(bytes = "vec", tag = "4")]
    artifact: Vec<u8>,
    #[prost(bytes = "vec", tag = "5")]
    checksum: Vec<u8>,
    #[prost(bytes = "vec", optional, tag = "6")]
    xla_flags: Option<Vec<u8>>,
}

/// Optional persistent cache of backend-native tensor executables.
/// Automatic trimming is opt-in; `max_entry_bytes` limits each encoded file, not total
/// disk usage or backend memory. Read/write failures are nonfatal cache misses.
pub struct DiskCache {
    store: FileArtifactStore,
    namespace: String,
    xla_flags: Option<Vec<u8>>,
    auto_trim_bytes: Option<u64>,
}

/// Best-effort directory observation, not a transaction or native-code validator.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DiskCacheInspection {
    pub entry_files: u64,
    pub encoded_bytes: u64,
    pub compatible: u64,
    pub incompatible: u64,
    pub corrupt: u64,
    pub oversized: u64,
    pub ignored: u64,
}

/// Best-effort maintenance of validated entries for one namespace/flags pair.
/// Byte counts are logical encoded sizes from the scan, not allocated blocks.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DiskCacheTrim {
    pub removed_files: u64,
    pub removed_bytes: u64,
    pub already_absent: u64,
    pub remaining_compatible_bytes: u64,
}

type TrimCandidate = (std::time::SystemTime, ArtifactId, u64);
impl DiskCache {
    /// Opt into synchronous oldest-modified trimming after each successful
    /// publication attempt (including an already-existing entry). Only validated
    /// entries in this namespace/flags pair count toward `budget`; zero removes
    /// all such entries. This may evict the executable just published, without
    /// invalidating its live handle or the compiler's memory cache.
    ///
    /// Configuration itself does not delete anything. Cache hits do not trigger
    /// maintenance. Each publication scans the directory and may read complete
    /// bounded entries, so this trades miss-path latency for bounded observed
    /// retention. Concurrent writers and excluded files can exceed the budget:
    /// this is best-effort retention, not a transactional directory quota or LRU.
    /// Trimming failures are nonfatal compiler `disk_write_errors`; publication
    /// may already have succeeded. Default behavior remains no automatic trim.
    pub fn with_auto_trim_to_bytes(mut self, budget: u64) -> Self {
        self.auto_trim_bytes = Some(budget);
        self
    }

    /// Explicitly remove oldest-modified compatible entries until their observed
    /// encoded size is at most `budget`. Zero removes all validated compatible
    /// entries. This is modification/publication age, NOT last-access LRU.
    /// Also used by opt-in post-publication trimming. No recursive deletion or native loading occurs.
    /// The scan excludes incompatible, corrupt, oversized, symlink and unrelated entries.
    /// Those ignored files may keep total directory usage above the budget.
    ///
    /// The entire bounded-per-entry scan succeeds before any removal. Removal
    /// errors can leave a partially trimmed cache; errors report removal count.
    /// Loaded executables/memory caches are unaffected. Coordinate with writers
    /// for deterministic results: concurrent publishers can replace selected
    /// paths or recreate entries, and the returned byte counts are observations,
    /// not a transactional disk quota or a guarantee of reclaimed disk space.
    pub fn trim_compatible_to_bytes(&self, budget: u64) -> Result<DiskCacheTrim> {
        let (_, mut candidates) = self.scan(true)?;
        candidates.sort(); // deterministic path tie-break for equal mtimes
        let mut result = DiskCacheTrim::default();
        for (_, _, bytes) in &candidates {
            result.remaining_compatible_bytes = result
                .remaining_compatible_bytes
                .checked_add(*bytes)
                .ok_or_else(|| err("cache size overflow"))?;
        }
        for (_, id, bytes) in candidates {
            if result.remaining_compatible_bytes <= budget {
                break;
            }
            match self.store.remove(id) {
                Ok(true) => {
                    result.removed_files += 1;
                    result.removed_bytes += bytes;
                }
                Ok(false) => result.already_absent += 1,
                Err(e) => {
                    return Err(err(format!(
                        "cache trim after {} removals: {e}",
                        result.removed_files
                    )));
                }
            }
            result.remaining_compatible_bytes -= bytes;
        }
        Ok(result)
    }
    /// Explicitly delete one entry keyed by the exact lowered StableHLO program,
    /// this namespace and the captured XLA_FLAGS. Returns false if absent.
    /// This removes valid or invalid entries alike; it is not automatic repair.
    /// No recursive deletion, envelope parsing or native loading occurs.
    ///
    /// Already loaded executables and compiler memory caches are unaffected.
    /// Clear/recreate the compiler separately to force a subsequent disk miss.
    /// Concurrent publishers can recreate the entry; coordinate with writers
    /// when guaranteed invalidation is needed.
    pub fn invalidate(&self, program: &LoweredProgram) -> Result<bool> {
        self.store
            .remove(self.artifact_id(&program.cache_key()))
            .map_err(|e| err(format!("cache invalidate: {e}")))
    }

    /// Read-only scan of immediate regular files named as cache entries.
    /// Does not load a plugin, deserialize native code, recurse or delete files.
    /// Symlinks, directories and unrelated names are ignored. Each candidate
    /// read is bounded by max_entry_bytes; oversized files are not decoded.
    /// Compatible means envelope/checksum/key match this cache, not that the
    /// backend can load the artifact. Incompatible envelopes are not validated
    /// against their original namespace's filename. I/O errors abort the scan.
    /// Concurrent publication/removal can change observations; bytes are logical
    /// file sizes, not allocated filesystem blocks or total directory usage.
    pub fn inspect(&self) -> Result<DiskCacheInspection> {
        self.scan(false).map(|(inspection, _)| inspection)
    }

    fn scan(
        &self,
        collect_trim_candidates: bool,
    ) -> Result<(DiskCacheInspection, Vec<TrimCandidate>)> {
        let mut result = DiskCacheInspection::default();
        let mut candidates = Vec::new();
        let scan = self
            .store
            .scan()
            .map_err(|e| err(format!("cache scan: {e}")))?;
        result.ignored = scan.ignored;
        for record in scan.records {
            let size = record.encoded_bytes;
            result.entry_files += 1;
            result.encoded_bytes = result
                .encoded_bytes
                .checked_add(size)
                .ok_or_else(|| err("cache size overflow"))?;
            let bytes = match self
                .store
                .lookup(record.id)
                .map_err(|e| err(format!("cache scan: {e}")))?
            {
                Lookup::Hit(bytes) => bytes,
                Lookup::Oversized => {
                    result.oversized += 1;
                    continue;
                }
                Lookup::Missing => continue,
            };
            let Ok(entry) = Entry::decode(bytes.as_slice()) else {
                result.corrupt += 1;
                continue;
            };
            if entry.checksum != blake3::hash(&entry.artifact).as_bytes() {
                result.corrupt += 1;
            } else if entry.version != 2
                || entry.namespace != self.namespace
                || entry.xla_flags != self.xla_flags
            {
                result.incompatible += 1;
            } else if self.artifact_id(&entry.key) != record.id {
                result.corrupt += 1;
            } else {
                result.compatible += 1;
                if collect_trim_candidates {
                    candidates.push((record.modified, record.id, size));
                }
            }
        }
        Ok((result, candidates))
    }

    /// Create/use a private trusted directory with an explicit compatibility key.
    /// Namespace must cover the exact plugin build, target hardware/platform and
    /// environment details that can affect compilation. The exact `XLA_FLAGS`
    /// value at construction (including unset versus empty), library format and
    /// fixed compile-option revision are additionally included in cache keys.
    /// This does not hash files referenced by flags or other environment variables.
    ///
    /// # Safety
    /// Directory contents and ancestors must remain protected against untrusted
    /// modification for all uses of this cache. Every attached compiler must use
    /// a client/environment compatible with the namespace. Hashes do not validate
    /// device compatibility: include selected device placement in the namespace
    /// when clients target different devices. Placement validation rejects a
    /// mismatched loaded executable, but does not partition cache filenames.
    /// Hashes do not validate native code safety; do not use downloaded or shared
    /// untrusted cache files.
    /// Set `XLA_FLAGS` before initializing the plugin and keep it unchanged for
    /// the process lifetime: backends may parse flags only once. This snapshot
    /// is not a query of the backend's effective compilation options.
    pub unsafe fn new(
        directory: impl AsRef<Path>,
        namespace: impl Into<String>,
        max_entry_bytes: usize,
    ) -> Result<Self> {
        let namespace = namespace.into();
        if namespace.is_empty() || max_entry_bytes == 0 || max_entry_bytes == usize::MAX {
            return Err(err(
                "disk cache requires namespace and bounded positive entry size",
            ));
        }
        let store = FileArtifactStore::new(directory, max_entry_bytes)
            .and_then(|store| store.with_extension("pb"))
            .map_err(|e| err(format!("cache directory: {e}")))?;
        Ok(Self {
            store,
            namespace,
            xla_flags: std::env::var_os("XLA_FLAGS").map(|s| s.as_encoded_bytes().to_vec()),
            auto_trim_bytes: None,
        })
    }
    /// Derive a placement-specific namespace from an explicit compatibility key
    /// and owned client metadata. This lets different selected devices share a
    /// directory without colliding on identical HLO. Inspection, invalidation
    /// and trimming operate on the derived namespace, not every device's files.
    ///
    /// Metadata covers platform/version, PJRT API version, process index and the
    /// selected device ID/kind/description. It is not a complete plugin binary,
    /// driver, topology or environment fingerprint: the supplied compatibility
    /// key must still cover those requirements. Recreate this cache with the same
    /// key and metadata to reuse its files; the derived namespace is opaque.
    ///
    /// # Safety
    /// All trust/environment requirements of [`Self::new`] still apply. Attach
    /// the returned cache only to a compiler with compatible client metadata;
    /// this constructor does not bind ownership to or retain the client.
    pub unsafe fn new_for_client(
        directory: impl AsRef<Path>,
        compatibility_key: &str,
        client: &Client,
        max_entry_bytes: usize,
    ) -> Result<Self> {
        let namespace = client_namespace(compatibility_key, &client.info()?)?;
        unsafe { Self::new(directory, namespace, max_entry_bytes) }
    }

    fn artifact_id(&self, key: &[u8]) -> ArtifactId {
        let mut hash = blake3::Hasher::new();
        hash.update(b"rxla-core-disk-v2-selected-device-options-v2");
        hash.update(&(self.namespace.len() as u64).to_le_bytes());
        hash.update(self.namespace.as_bytes());
        match &self.xla_flags {
            None => {
                hash.update(&[0]);
            }
            Some(flags) => {
                hash.update(&[1]);
                hash.update(&(flags.len() as u64).to_le_bytes());
                hash.update(flags);
            }
        }
        hash.update(key);
        ArtifactId::new(*hash.finalize().as_bytes())
    }
    #[cfg(test)]
    fn path(&self, key: &[u8]) -> PathBuf {
        self.store.path(self.artifact_id(key))
    }
    pub(crate) fn load(&self, client: &Client, key: &[u8]) -> Result<Option<Executable>> {
        let bytes = match self
            .store
            .lookup(self.artifact_id(key))
            .map_err(|e| err(format!("cache read: {e}")))?
        {
            Lookup::Hit(bytes) => bytes,
            Lookup::Missing => return Ok(None),
            Lookup::Oversized => return Err(err("cache entry exceeds size limit")),
        };
        let entry = self.decode_entry(&bytes, key)?;
        // Constructor contract covers code trust and environment compatibility.
        unsafe { Executable::deserialize_with_metadata(client, &entry.artifact) }.map(Some)
    }
    fn decode_entry(&self, bytes: &[u8], key: &[u8]) -> Result<Entry> {
        let entry = Entry::decode(bytes).map_err(|e| err(format!("cache envelope: {e}")))?;
        if entry.version != 2
            || entry.namespace != self.namespace
            || entry.key != key
            || entry.xla_flags != self.xla_flags
            || entry.checksum != blake3::hash(&entry.artifact).as_bytes()
        {
            return Err(err("cache key or checksum mismatch"));
        }
        Ok(entry)
    }
    pub(crate) fn store(&self, key: &[u8], executable: &Executable) -> Result<()> {
        let artifact = executable.serialize_with_metadata()?;
        let entry = Entry {
            version: 2,
            namespace: self.namespace.clone(),
            key: key.to_vec(),
            checksum: blake3::hash(&artifact).as_bytes().to_vec(),
            artifact,
            xla_flags: self.xla_flags.clone(),
        };
        self.store
            .publish(self.artifact_id(key), &entry.encode_to_vec())
            .map_err(|e| err(format!("cache publish: {e}")))?;
        if let Some(budget) = self.auto_trim_bytes {
            self.trim_compatible_to_bytes(budget)?;
        }
        Ok(())
    }
}

fn client_namespace(key: &str, info: &ClientInfo) -> Result<String> {
    if key.is_empty() {
        return Err(err("client cache requires a nonempty compatibility key"));
    }
    let mut selected = info.addressable_devices.iter().filter(|d| d.selected);
    let device = selected
        .next()
        .ok_or_else(|| err("cache client has no selected device"))?;
    if selected.next().is_some() {
        return Err(err("cache client has multiple selected devices"));
    }
    let mut hash = blake3::Hasher::new();
    hash.update(b"rxla-core-client-namespace-v1");
    for field in [
        key,
        &info.platform,
        &info.platform_version,
        &device.kind,
        &device.description,
    ] {
        hash.update(&(field.len() as u64).to_le_bytes());
        hash.update(field.as_bytes());
    }
    for value in [
        info.api_major,
        info.api_minor,
        info.process_index,
        device.id,
    ] {
        hash.update(&value.to_le_bytes());
    }
    Ok(format!("client-v1-{}", hash.finalize().to_hex()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    #[test]
    fn client_namespace_is_stable_and_separates_placement_and_compatibility() {
        let info = ClientInfo {
            api_major: 0,
            api_minor: 115,
            platform: "cpu".into(),
            platform_version: "test-build".into(),
            process_index: 0,
            addressable_devices: vec![DeviceInfo {
                id: 0,
                kind: "cpu".into(),
                description: "test device".into(),
                selected: true,
            }],
        };
        let original = client_namespace("build-A", &info).unwrap();
        assert_eq!(
            original,
            client_namespace("build-A", &info.clone()).unwrap()
        );
        assert_ne!(original, client_namespace("build-B", &info).unwrap());
        for change in [
            |i: &mut ClientInfo| i.api_major += 1,
            |i: &mut ClientInfo| i.api_minor += 1,
            |i: &mut ClientInfo| i.process_index += 1,
            |i: &mut ClientInfo| i.platform.push('x'),
            |i: &mut ClientInfo| i.platform_version.push('x'),
            |i: &mut ClientInfo| i.addressable_devices[0].id += 1,
            |i: &mut ClientInfo| i.addressable_devices[0].kind.push('x'),
            |i: &mut ClientInfo| i.addressable_devices[0].description.push('x'),
        ] {
            let mut changed = info.clone();
            change(&mut changed);
            assert_ne!(original, client_namespace("build-A", &changed).unwrap());
        }
        assert!(client_namespace("", &info).is_err());
        let mut invalid = info.clone();
        invalid.addressable_devices[0].selected = false;
        assert!(client_namespace("build-A", &invalid).is_err());
        let mut invalid = info.clone();
        invalid
            .addressable_devices
            .push(info.addressable_devices[0].clone());
        assert!(client_namespace("build-A", &invalid).is_err());
    }

    #[test]
    fn invalidation_is_scoped_and_does_not_recurse() {
        let temp = tempfile::tempdir().unwrap();
        let cache = unsafe { DiskCache::new(temp.path(), "test", 512) }.unwrap();
        let other = unsafe { DiskCache::new(temp.path(), "other", 512) }.unwrap();
        let graph = Graph::default();
        let output = graph.input(&[1]).unwrap().exp().unwrap();
        let program = graph.prepare(&output).unwrap();
        let encoded = program.cache_key();
        let path = cache.path(&encoded);
        let other_path = other.path(&encoded);
        std::fs::write(&path, b"bad cache").unwrap();
        std::fs::write(&other_path, b"keep").unwrap();
        assert!(cache.invalidate(&program).unwrap());
        assert!(!cache.invalidate(&program).unwrap());
        assert_eq!(std::fs::read(&other_path).unwrap(), b"keep");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("keep"), b"nested").unwrap();
        assert!(cache.invalidate(&program).is_err());
        assert_eq!(std::fs::read(path.join("keep")).unwrap(), b"nested");
    }

    #[test]
    fn inspection_classifies_without_native_loading_or_writes() {
        let temp = tempfile::tempdir().unwrap();
        let cache = unsafe { DiskCache::new(temp.path(), "test", 512) }.unwrap();
        let make = |key: u8| Entry {
            version: 2,
            namespace: "test".into(),
            key: vec![key],
            artifact: vec![99],
            checksum: blake3::hash(&[99]).as_bytes().to_vec(),
            xla_flags: cache.xla_flags.clone(),
        };
        let mut foreign = make(2);
        foreign.namespace = "other".into();
        let mut bad = make(3);
        bad.checksum.clear();
        let files = [
            (cache.path(&[1]), make(1).encode_to_vec()),
            (cache.path(&[2]), foreign.encode_to_vec()),
            (cache.path(&[3]), bad.encode_to_vec()),
            (cache.path(&[4]), vec![255]),
            (cache.path(&[5]), make(6).encode_to_vec()),
            (cache.path(&[7]), vec![0; 513]),
        ];
        for (path, bytes) in &files {
            std::fs::write(path, bytes).unwrap();
        }
        std::fs::write(temp.path().join("notes.txt"), b"keep").unwrap();
        std::fs::create_dir(temp.path().join("subdir")).unwrap();
        assert_eq!(
            cache.inspect().unwrap(),
            DiskCacheInspection {
                entry_files: 6,
                encoded_bytes: files.iter().map(|(_, b)| b.len() as u64).sum(),
                compatible: 1,
                incompatible: 1,
                corrupt: 3,
                oversized: 1,
                ignored: 2,
            }
        );
        for (path, bytes) in &files {
            assert_eq!(&std::fs::read(path).unwrap(), bytes);
        }
        assert_eq!(
            std::fs::read(temp.path().join("notes.txt")).unwrap(),
            b"keep"
        );
    }

    #[test]
    fn trimming_uses_modification_age_and_preserves_other_files() {
        let temp = tempfile::tempdir().unwrap();
        let cache = unsafe { DiskCache::new(temp.path(), "trim", 512) }.unwrap();
        let other = unsafe { DiskCache::new(temp.path(), "other", 512) }.unwrap();
        let make = |id, namespace: &str| Entry {
            version: 2,
            namespace: namespace.into(),
            key: vec![id],
            artifact: vec![99],
            checksum: blake3::hash(&[99]).as_bytes().to_vec(),
            xla_flags: cache.xla_flags.clone(),
        };
        let mut sizes = [0u64; 3];
        for id in [3u8, 1, 2] {
            let bytes = make(id, "trim").encode_to_vec();
            sizes[id as usize - 1] = bytes.len() as u64;
            let path = cache.path(&[id]);
            std::fs::write(&path, bytes).unwrap();
            let time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(100 + id as u64);
            std::fs::OpenOptions::new()
                .write(true)
                .open(path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(time))
                .unwrap();
        }
        let protected = [
            (other.path(&[4]), make(4, "other").encode_to_vec()),
            (cache.path(&[5]), b"invalid protobuf".to_vec()),
            (cache.path(&[6]), vec![0; 513]),
            (temp.path().join("notes.txt"), b"notes".to_vec()),
        ];
        for (path, bytes) in &protected {
            std::fs::write(path, bytes).unwrap();
        }
        let nested = cache.path(&[7]);
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(nested.join("keep"), b"nested").unwrap();
        let before = cache.inspect().unwrap();
        assert_eq!(
            cache.trim_compatible_to_bytes(u64::MAX).unwrap(),
            DiskCacheTrim {
                remaining_compatible_bytes: sizes.iter().sum(),
                ..Default::default()
            }
        );
        assert_eq!(cache.inspect().unwrap(), before);
        assert_eq!(
            cache.trim_compatible_to_bytes(sizes[1] + sizes[2]).unwrap(),
            DiskCacheTrim {
                removed_files: 1,
                removed_bytes: sizes[0],
                already_absent: 0,
                remaining_compatible_bytes: sizes[1] + sizes[2],
            }
        );
        assert!(!cache.path(&[1]).exists());
        assert!(cache.path(&[2]).exists() && cache.path(&[3]).exists());
        let trimmed = cache.trim_compatible_to_bytes(0).unwrap();
        assert_eq!(trimmed.removed_files, 2);
        assert_eq!(trimmed.removed_bytes, sizes[1] + sizes[2]);
        assert_eq!(trimmed.remaining_compatible_bytes, 0);
        assert_eq!(
            cache.trim_compatible_to_bytes(0).unwrap(),
            DiskCacheTrim::default()
        );
        for (path, bytes) in protected {
            assert_eq!(std::fs::read(path).unwrap(), bytes);
        }
        assert_eq!(std::fs::read(nested.join("keep")).unwrap(), b"nested");
    }

    #[cfg(unix)]
    #[test]
    fn trimming_ignores_symlinks_and_other_flag_namespaces() {
        let temp = tempfile::tempdir().unwrap();
        let cache = unsafe { DiskCache::new(temp.path(), "trim", 512) }.unwrap();
        let mut other = unsafe { DiskCache::new(temp.path(), "trim", 512) }.unwrap();
        other.xla_flags = match cache.xla_flags {
            None => Some(vec![]),
            Some(_) => None,
        };
        let entry = Entry {
            version: 2,
            namespace: "trim".into(),
            key: vec![1],
            artifact: vec![1],
            checksum: blake3::hash(&[1]).as_bytes().to_vec(),
            xla_flags: other.xla_flags.clone(),
        }
        .encode_to_vec();
        let other_path = other.path(&[1]);
        std::fs::write(&other_path, &entry).unwrap();
        std::os::unix::fs::symlink(&other_path, cache.path(&[1])).unwrap();
        assert_eq!(
            cache.trim_compatible_to_bytes(0).unwrap(),
            DiskCacheTrim::default()
        );
        assert_eq!(std::fs::read(&other_path).unwrap(), entry);
        assert!(
            std::fs::symlink_metadata(cache.path(&[1]))
                .unwrap()
                .is_symlink()
        );
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn real_trim_preserves_live_executable_and_memory_cache() {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        let temp = tempfile::tempdir().unwrap();
        let disk =
            unsafe { DiskCache::new(temp.path(), "same-plugin-trim", 16 * 1024 * 1024) }.unwrap();
        let mut compiler =
            Compiler::new(client.clone(), CacheLimits::default()).with_disk_cache(disk);
        let graph = Graph::default();
        let x = graph.input(&[]).unwrap();
        let y = x.add_scalar(1.).unwrap();
        let executable = compiler.compile(&graph, &y).unwrap();
        assert_eq!(
            compiler.disk_cache().unwrap().inspect().unwrap().compatible,
            1
        );
        let trimmed = compiler
            .disk_cache()
            .unwrap()
            .trim_compatible_to_bytes(0)
            .unwrap();
        assert_eq!(trimmed.removed_files, 1);
        assert!(trimmed.removed_bytes > 0);
        assert_eq!(
            compiler
                .disk_cache()
                .unwrap()
                .inspect()
                .unwrap()
                .entry_files,
            0
        );
        let input = client.buffer(&[], &[2.]).unwrap();
        assert_eq!(
            executable.execute(&[&input]).unwrap()[0]
                .to_vec::<f32>()
                .unwrap(),
            [3.]
        );
        let cached = compiler.compile(&graph, &y).unwrap();
        assert!(std::sync::Arc::ptr_eq(&executable, &cached));
        assert_eq!(compiler.stats().misses, 1);
        compiler.clear();
        let recompiled = compiler.compile(&graph, &y).unwrap();
        assert_eq!(compiler.stats().misses, 2);
        assert_eq!(
            compiler.disk_cache().unwrap().inspect().unwrap().compatible,
            1
        );
        assert_eq!(
            recompiled.execute(&[&input]).unwrap()[0]
                .to_vec::<f32>()
                .unwrap(),
            [3.]
        );
    }

    #[cfg(unix)]
    #[test]
    fn inspection_ignores_symlinks() {
        let temp = tempfile::tempdir().unwrap();
        let cache = unsafe { DiskCache::new(temp.path(), "test", 512) }.unwrap();
        std::os::unix::fs::symlink("missing", cache.path(&[1])).unwrap();
        assert_eq!(
            cache.inspect().unwrap(),
            DiskCacheInspection {
                ignored: 1,
                ..Default::default()
            }
        );
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn real_trim_automatic_retention_preserves_execution_and_other_namespaces() {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        let temp = tempfile::tempdir().unwrap();
        let make = |namespace| unsafe {
            DiskCache::new(temp.path(), namespace, 16 * 1024 * 1024).unwrap()
        };
        let graph = Graph::default();
        let input = graph.input(&[]).unwrap();
        let output = input.add_scalar(1.).unwrap();
        let mut other =
            Compiler::new(client.clone(), CacheLimits::default()).with_disk_cache(make("other"));
        other.compile(&graph, &output).unwrap();
        let mut retained = Compiler::new(client.clone(), CacheLimits::default())
            .with_disk_cache(make("automatic").with_auto_trim_to_bytes(u64::MAX));
        retained.compile(&graph, &output).unwrap();
        let observed = retained.disk_cache().unwrap().inspect().unwrap();
        assert_eq!(observed.compatible, 1);
        assert_eq!(observed.incompatible, 1);
        let disk = make("automatic").with_auto_trim_to_bytes(0);
        // Merely configuring the policy must not perform deletion.
        assert_eq!(disk.inspect().unwrap().compatible, 1);
        let mut compiler =
            Compiler::new(client.clone(), CacheLimits::default()).with_disk_cache(disk);
        compiler.compile(&graph, &output).unwrap();
        assert_eq!(compiler.stats().disk_hits, 1);
        assert_eq!(
            compiler.disk_cache().unwrap().inspect().unwrap().compatible,
            1
        );
        // A genuinely new compilation triggers cleanup of both old and new entries.
        let next = input.add_scalar(2.).unwrap();
        let executable = compiler.compile(&graph, &next).unwrap();
        assert_eq!(
            compiler.disk_cache().unwrap().inspect().unwrap().compatible,
            0
        );
        assert_eq!(other.disk_cache().unwrap().inspect().unwrap().compatible, 1);
        assert_eq!(executable.run(&[&[3.]]).unwrap(), [5.]);
        let cached = compiler.compile(&graph, &next).unwrap();
        assert!(std::sync::Arc::ptr_eq(&executable, &cached));
        compiler.clear();
        compiler.compile(&graph, &next).unwrap();
        assert_eq!(compiler.stats().misses, 2);
        assert_eq!(compiler.stats().disk_write_errors, 0);
        assert_eq!(
            compiler.disk_cache().unwrap().inspect().unwrap().compatible,
            0
        );
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn real_trim_automatic_exact_budget_and_existing_publication() {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        let temp = tempfile::tempdir().unwrap();
        let make = || unsafe {
            DiskCache::new(temp.path(), "automatic-boundary", 16 * 1024 * 1024).unwrap()
        };
        let disk = make();
        let mut compiler =
            Compiler::new(client.clone(), CacheLimits::default()).with_disk_cache(make());
        let graph = Graph::default();
        let input = graph.input(&[]).unwrap();
        let mut entries = Vec::new();
        for value in [1., 2., 3.] {
            let output = input.add_scalar(value).unwrap();
            let executable = compiler.compile(&graph, &output).unwrap();
            let key = graph.prepare(&output).unwrap().cache_key();
            let path = disk.path(&key);
            let size = std::fs::metadata(&path).unwrap().len();
            let time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(value as u64);
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(time))
                .unwrap();
            entries.push((output, executable, key, path, size));
        }
        let budget = entries[1].4 + entries[2].4;
        let automatic = make().with_auto_trim_to_bytes(budget);
        // Simulate a second publisher arriving after the exact same key exists.
        // The existing file must win and maintenance must still run.
        let preserved = std::fs::read(&entries[2].3).unwrap();
        automatic.store(&entries[2].2, &entries[2].1).unwrap();
        assert!(!entries[0].3.exists());
        assert!(entries[1].3.exists());
        assert_eq!(std::fs::read(&entries[2].3).unwrap(), preserved);
        let inspected = automatic.inspect().unwrap();
        assert_eq!(inspected.compatible, 2);
        assert_eq!(inspected.encoded_bytes, budget);
        // Equality keeps both entries; one byte less evicts the older one.
        let tighter = make().with_auto_trim_to_bytes(budget - 1);
        tighter.store(&entries[2].2, &entries[2].1).unwrap();
        assert!(!entries[1].3.exists());
        assert_eq!(tighter.inspect().unwrap().encoded_bytes, entries[2].4);
        let mut fresh = Compiler::new(client, CacheLimits::default()).with_disk_cache(tighter);
        let restored = fresh.compile(&graph, &entries[2].0).unwrap();
        assert_eq!(fresh.stats().disk_hits, 1);
        assert_eq!(fresh.stats().misses, 0);
        assert_eq!(restored.run(&[&[4.]]).unwrap(), [7.]);
        // Eviction has no effect on any previously loaded native executable.
        assert_eq!(entries[0].1.run(&[&[4.]]).unwrap(), [5.]);
        assert_eq!(entries[1].1.run(&[&[4.]]).unwrap(), [6.]);
    }

    #[test]
    fn flag_values_partition_paths_and_envelopes() {
        let temp = tempfile::tempdir().unwrap();
        let mut cache = unsafe { DiskCache::new(temp.path(), "test", 1024) }.unwrap();
        let variants = [
            None,
            Some(vec![]),
            Some(b"--xla_gpu_enable_command_buffer=".to_vec()),
            Some(b"a\0b".to_vec()),
        ];
        let mut paths = std::collections::HashSet::new();
        for flags in &variants {
            cache.xla_flags = flags.clone();
            assert!(paths.insert(cache.path(&[1])));
            let entry = Entry {
                version: 2,
                namespace: "test".into(),
                key: vec![1],
                artifact: vec![2],
                checksum: blake3::hash(&[2]).as_bytes().to_vec(),
                xla_flags: flags.clone(),
            }
            .encode_to_vec();
            for other in &variants {
                cache.xla_flags = other.clone();
                assert_eq!(cache.decode_entry(&entry, &[1]).is_ok(), flags == other);
            }
        }
    }

    #[test]
    fn constructor_snapshots_flags_in_isolated_processes() {
        const CHILD: &str = "XLA_CACHE_FLAGS_SNAPSHOT_CHILD";
        if let Some(expected) = std::env::var_os(CHILD) {
            let temp = tempfile::tempdir().unwrap();
            let cache = unsafe { DiskCache::new(temp.path(), "test", 1024) }.unwrap();
            let expected = if expected == "unset" {
                None
            } else {
                Some(expected.as_encoded_bytes().to_vec())
            };
            assert_eq!(cache.xla_flags, expected);
            return;
        }
        for value in [None, Some(""), Some("--xla_gpu_enable_command_buffer=")] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "disk_cache::tests::constructor_snapshots_flags_in_isolated_processes",
                ])
                .env(CHILD, value.unwrap_or("unset"));
            match value {
                Some(value) => {
                    command.env("XLA_FLAGS", value);
                }
                None => {
                    command.env_remove("XLA_FLAGS");
                }
            }
            assert!(command.status().unwrap().success());
        }
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn real_concurrent_publish_keeps_one_complete_entry() {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        const CHILD: &str = "XLA_CACHE_PUBLISH_CHILD_PATH";
        const READY: &str = "XLA_CACHE_PUBLISH_READY";
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        let graph = Graph::default();
        let x = graph.input(&[2]).unwrap();
        let y = x.add_scalar(3.).unwrap();
        let key = graph.prepare(&y).unwrap().cache_key();
        if let Some(path) = std::env::var_os(CHILD) {
            let cache =
                unsafe { DiskCache::new(path, "same-plugin-publish-test", 1024 * 1024) }.unwrap();
            let executable = graph.compile(&client, &y).unwrap();
            // Both children finish compilation before the parent permits either
            // to publish. This exercises competing writers, not a lucky cache hit.
            println!("{READY}");
            std::io::stdout().flush().unwrap();
            let mut go = [0];
            std::io::stdin().read_exact(&mut go).unwrap();
            cache.store(&key, &executable).unwrap();
            assert_eq!(executable.run(&[&[1., 2.]]).unwrap(), [4., 5.]);
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let cache =
            unsafe { DiskCache::new(directory.path(), "same-plugin-publish-test", 1024 * 1024) }
                .unwrap();
        let mut children: Vec<_> = (0..2)
            .map(|_| {
                let mut child = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "disk_cache::tests::real_concurrent_publish_keeps_one_complete_entry",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env(CHILD, directory.path())
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()
                    .unwrap();
                let stdout = BufReader::new(child.stdout.take().unwrap());
                (child, stdout)
            })
            .collect();
        for (_, stdout) in &mut children {
            loop {
                let mut line = String::new();
                assert_ne!(
                    stdout.read_line(&mut line).unwrap(),
                    0,
                    "publisher exited before readiness"
                );
                if line.contains(READY) {
                    break;
                }
            }
        }
        assert!(!cache.path(&key).exists());
        for (child, _) in &mut children {
            child.stdin.take().unwrap().write_all(&[1]).unwrap();
        }
        for (child, stdout) in &mut children {
            let mut output = String::new();
            stdout.read_to_string(&mut output).unwrap();
            assert!(child.wait().unwrap().success(), "{output}");
        }
        let entries: Vec<_> = std::fs::read_dir(directory.path()).unwrap().collect();
        assert_eq!(entries.len(), 1, "no abandoned temporary files");
        assert_eq!(entries[0].as_ref().unwrap().path(), cache.path(&key));
        let restored = cache.load(&client, &key).unwrap().unwrap();
        assert_eq!(restored.run(&[&[10., 20.]]).unwrap(), [13., 23.]);
    }
    #[test]
    fn rejects_wrong_key_namespace_version_and_checksum_before_native_loading() {
        let temp = tempfile::tempdir().unwrap();
        let cache = unsafe { DiskCache::new(temp.path(), "test", 1024) }.unwrap();
        let entry = Entry {
            version: 2,
            namespace: "test".into(),
            key: vec![1],
            artifact: vec![2],
            checksum: blake3::hash(&[2]).as_bytes().to_vec(),
            xla_flags: cache.xla_flags.clone(),
        };
        assert!(cache.decode_entry(&entry.encode_to_vec(), &[1]).is_ok());
        for mutate in [
            (|e: &mut Entry| e.version = 1) as fn(&mut Entry),
            |e| e.namespace = "other".into(),
            |e| e.key = vec![3],
            |e| e.artifact = vec![3],
            |e| e.checksum.clear(),
        ] {
            let mut changed = entry.clone();
            mutate(&mut changed);
            assert!(cache.decode_entry(&changed.encode_to_vec(), &[1]).is_err());
        }
    }
}
