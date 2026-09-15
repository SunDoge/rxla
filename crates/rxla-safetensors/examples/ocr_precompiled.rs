//! Same-environment OCR restore smoke; no tensor graph or compiler imports.
//! Both plugin and artifact execute native code and must be trusted.
use rxla_pjrt::Client;
use rxla_safetensors::{Dtype, SafeTensors, TensorInfo};
use serde::Deserialize;
use std::{
    collections::HashSet,
    fs::File,
    io::{Cursor, Read, Seek},
    path::PathBuf,
    time::Instant,
};
mod ocr_artifact;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Weight {
    name: String,
    shape: Vec<i64>,
}
#[derive(Deserialize)]
struct Manifest {
    version: u32,
    executable_blake3: String,
    weights_blake3: String,
    plugin_blake3: String,
    platform: String,
    platform_version: String,
    api_major: i32,
    api_minor: i32,
    device_kind: String,
    input_shape: Vec<i64>,
    output_shape: Vec<i64>,
    weights: Vec<Weight>,
}

// Bounds cover this OCR prototype, not arbitrary future model packages.
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_PAYLOAD_BYTES: u64 = 256 * 1024 * 1024;

fn read_bounded(reader: impl Read, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(limit.checked_add(1).ok_or("invalid byte limit")?)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("artifact file exceeds byte limit".into());
    }
    Ok(bytes)
}

fn verify_payload(bytes: &[u8], digest: &str) -> Result<()> {
    let expected = blake3::Hash::from_hex(digest)?;
    if blake3::hash(bytes) != expected {
        return Err("artifact payload checksum mismatch".into());
    }
    Ok(())
}

fn check_tensor(info: Option<&TensorInfo>, shape: &[i64], name: &str) -> Result<()> {
    let info = info.ok_or_else(|| format!("missing tensor {name:?}"))?;
    if info.dtype != Dtype::F32
        || info.shape.len() != shape.len()
        || info
            .shape
            .iter()
            .zip(shape)
            .any(|(&actual, &expected)| usize::try_from(expected).ok() != Some(actual))
    {
        return Err(format!("tensor {name:?} must have F32 shape {shape:?}").into());
    }
    Ok(())
}

// Bound each declared F32 tensor before allocating/uploading input or executing
// native code. This is not an aggregate host/device memory quota or ABI proof.
fn check_f32_shape(shape: &[i64]) -> Result<()> {
    let elements = shape.iter().try_fold(1u64, |count, &dim| {
        count.checked_mul(u64::try_from(dim).ok()?)
    });
    let bytes = elements.and_then(|count| count.checked_mul(4));
    if bytes.is_none_or(|bytes| bytes > MAX_PAYLOAD_BYTES) {
        return Err("invalid or oversized artifact F32 tensor shape".into());
    }
    Ok(())
}

// Inspect parsed headers only, before plugin loading or native deserialization.
// This checks consistency, not authenticity or the executable's actual ABI.
fn preflight<R: Read + Seek, S: Read + Seek>(
    manifest: &Manifest,
    weights: &SafeTensors<R>,
    input: &SafeTensors<S>,
    input_name: &str,
) -> Result<()> {
    check_f32_shape(&manifest.input_shape)?;
    check_f32_shape(&manifest.output_shape)?;
    let mut names = HashSet::new();
    for weight in &manifest.weights {
        check_f32_shape(&weight.shape)?;
        if weight.name.is_empty() || !names.insert(weight.name.as_str()) {
            return Err("empty or duplicate manifest weight name".into());
        }
        check_tensor(weights.info(&weight.name), &weight.shape, &weight.name)?;
    }
    if weights.names().len() != names.len() {
        return Err("unexpected tensors in weight package".into());
    }
    check_tensor(input.info(input_name), &manifest.input_shape, input_name)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let check_only = args.first().is_some_and(|arg| arg == "--check-package");
    if !((check_only && args.len() == 4)
        || (args.len() == 5 && args[0] == "--trusted-same-environment"))
    {
        return Err(
            "usage: ocr_precompiled --check-package ARTIFACT INPUT.safetensors INPUT_NAME | --trusted-same-environment ARTIFACT INPUT.safetensors INPUT_NAME REPORT.json".into(),
        );
    }
    let artifact = PathBuf::from(&args[1]);
    let input_path = PathBuf::from(&args[2]);
    let input_name = args[3].to_str().ok_or("input name must be UTF-8")?;
    let report = args.get(4).map(PathBuf::from);
    if report.as_ref().is_some_and(|path| path.exists()) {
        return Err("report already exists".into());
    }
    let manifest_bytes = read_bounded(
        File::open(artifact.join("manifest.json"))?,
        MAX_MANIFEST_BYTES,
    )?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.version != 4 {
        return Err("unsupported artifact version".into());
    }
    let native = read_bounded(
        File::open(artifact.join("executable.bin"))?,
        MAX_PAYLOAD_BYTES,
    )?;
    let weight_bytes = read_bounded(
        File::open(artifact.join("weights.safetensors"))?,
        MAX_PAYLOAD_BYTES,
    )?;
    verify_payload(&native, &manifest.executable_blake3)?;
    verify_payload(&weight_bytes, &manifest.weights_blake3)?;
    // Consume exactly the checked bytes, not a reopened or mutable file handle.
    let mut checkpoint = SafeTensors::new(Cursor::new(weight_bytes))?;
    let mut input_checkpoint = SafeTensors::open(input_path)?;
    preflight(&manifest, &checkpoint, &input_checkpoint, input_name)?;
    // Check syntax even in host-only mode; the actual plugin is neither located
    // nor loaded here. Checksums establish consistency, not native-code trust.
    let expected_plugin = blake3::Hash::from_hex(&manifest.plugin_blake3)?;
    if check_only {
        serde_json::to_writer(
            std::io::stdout().lock(),
            &serde_json::json!({
                "package_consistent": true,
                "native_compatibility_checked": false,
                "platform": manifest.platform,
                "input_shape": manifest.input_shape,
                "output_shape": manifest.output_shape,
                "weight_count": manifest.weights.len(),
                "scope": "Host-only bounded payload/checksum and tensor-header checks. No PJRT plugin loading, native deserialization, execution, or output correctness check. Not authentication."
            }),
        )?;
        return Ok(());
    }
    // Explicit CLI contract: trusted artifact and exactly compatible plugin,
    // device and native dependencies. These diagnostics alone cannot prove it.
    let plugin_path = std::fs::canonicalize(std::env::var("PJRT_PLUGIN_PATH")?)?;
    let plugin_digest = ocr_artifact::plugin_fingerprint(&plugin_path)?;
    if plugin_digest != expected_plugin {
        return Err("artifact PJRT plugin fingerprint mismatch".into());
    }
    // File and ancestors must remain protected/immutable through native loading.
    // Hash checking is compatibility diagnostics, not a TOCTOU defense.
    let client = unsafe { Client::load(&plugin_path)? };
    let info = client.info()?;
    if info.platform != manifest.platform
        || info.platform_version != manifest.platform_version
        || info.api_major != manifest.api_major
        || info.api_minor != manifest.api_minor
        || !info
            .addressable_devices
            .iter()
            .any(|d| d.selected && d.kind == manifest.device_kind)
    {
        return Err("artifact target metadata mismatch".into());
    }
    let start = Instant::now();
    let executable = unsafe { client.deserialize_executable(&native)? };
    let restore_seconds = start.elapsed().as_secs_f64();
    let mut buffers = Vec::new();
    let input = input_checkpoint.upload_f32(&client, input_name)?;
    if input.dimensions()? != manifest.input_shape {
        return Err("input shape mismatch".into());
    }
    buffers.push(input);
    for weight in &manifest.weights {
        let buffer = checkpoint.upload_f32(&client, &weight.name)?;
        if buffer.dimensions()? != weight.shape {
            return Err("weight shape mismatch".into());
        }
        buffers.push(buffer);
    }
    let outputs = executable.execute(&buffers.iter().collect::<Vec<_>>())?;
    if outputs.len() != 1 || outputs[0].dimensions()? != manifest.output_shape {
        return Err("output signature mismatch".into());
    }
    let values = outputs[0].to_vec::<f32>()?;
    if !values.iter().all(|x| x.is_finite()) {
        return Err("nonfinite output".into());
    }
    serde_json::to_writer(
        File::create_new(report.ok_or("missing report path")?)?,
        &serde_json::json!({
            "restore_seconds": restore_seconds, "platform": info.platform,
            "plugin_blake3": plugin_digest.to_hex().to_string(),
            "shape": manifest.output_shape, "output": values,
            "scope": "Trusted same-environment restoration. Artifact supplies weights; a separate named input tensor is supplied by the caller. No graph construction, HLO lowering, compiler call or compilation fallback in this loader. Backend-internal load work is not asserted absent. Output must be compared independently. Not a signed portable deployment package."
        }),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn checkpoint(entries: &[(&str, &str, usize)]) -> SafeTensors<Cursor<Vec<u8>>> {
        let mut header = serde_json::Map::new();
        let mut end = 0;
        for &(name, dtype, count) in entries {
            let start = end;
            end += count * if dtype == "F16" { 2 } else { 4 };
            header.insert(
                name.into(),
                serde_json::json!({
                    "dtype": dtype, "shape": [count], "data_offsets": [start, end]
                }),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + end, 0);
        SafeTensors::new(Cursor::new(bytes)).unwrap()
    }

    fn manifest() -> Manifest {
        serde_json::from_value(serde_json::json!({
            "version": 4, "platform": "cpu", "platform_version": "test",
            "plugin_blake3": blake3::hash(b"plugin").to_hex().to_string(),
            "executable_blake3": blake3::hash(b"native").to_hex().to_string(),
            "weights_blake3": blake3::hash(b"weights").to_hex().to_string(),
            "api_major": 0, "api_minor": 0, "device_kind": "test",
            "input_shape": [2], "output_shape": [2],
            "weights": [{"name": "w", "shape": [2]}]
        }))
        .unwrap()
    }

    #[test]
    fn valid_headers_need_no_payload_reads_or_plugin() {
        let weights = checkpoint(&[("w", "F32", 2)]);
        let input = checkpoint(&[("x", "F32", 2), ("other", "I32", 1)]);
        preflight(&manifest(), &weights, &input, "x").unwrap();
        assert_eq!(weights.stats().reads, 0);
        assert_eq!(input.stats().reads, 0);
    }

    #[test]
    fn bounded_reads_and_checksums_reject_corruption_without_plugin() {
        let bytes = b"native bytes";
        let digest = blake3::hash(bytes).to_hex().to_string();
        verify_payload(bytes, &digest).unwrap();
        assert!(verify_payload(b"Native bytes", &digest).is_err());
        assert!(verify_payload(&bytes[..bytes.len() - 1], &digest).is_err());
        assert!(verify_payload(b"native bytes!", &digest).is_err());
        assert!(verify_payload(bytes, "not a digest").is_err());
        assert!(verify_payload(b"weight bytes", &digest).is_err());
        assert_eq!(
            read_bounded(bytes.as_slice(), bytes.len() as u64).unwrap(),
            bytes
        );
        assert!(read_bounded(bytes.as_slice(), bytes.len() as u64 - 1).is_err());
        assert!(read_bounded(bytes.as_slice(), u64::MAX).is_err());
        assert!(read_bounded([].as_slice(), 0).unwrap().is_empty());
    }

    #[test]
    fn rejects_missing_extra_wrong_shape_and_wrong_dtype_weights() {
        let input = checkpoint(&[("x", "F32", 2)]);
        for entries in [
            vec![("missing", "F32", 2)],
            vec![("w", "F32", 2), ("expected/0", "F32", 1)],
            vec![("w", "F32", 3)],
            vec![("w", "F16", 2)],
            vec![("w", "I32", 2)],
        ] {
            assert!(preflight(&manifest(), &checkpoint(&entries), &input, "x").is_err());
        }
    }

    #[test]
    fn rejects_invalid_manifest_names_and_dimensions() {
        let weights = checkpoint(&[("w", "F32", 2)]);
        let input = checkpoint(&[("x", "F32", 2)]);
        let mut m = manifest();
        m.weights.push(Weight {
            name: "w".into(),
            shape: vec![2],
        });
        assert!(preflight(&m, &weights, &input, "x").is_err());
        m.weights.pop();
        m.weights[0].name.clear();
        assert!(preflight(&m, &weights, &input, "x").is_err());
        m = manifest();
        m.weights[0].shape = vec![-1];
        assert!(preflight(&m, &weights, &input, "x").is_err());
        m.weights[0].shape = vec![1, 2];
        assert!(preflight(&m, &weights, &input, "x").is_err());
    }

    #[test]
    fn rejects_missing_or_incompatible_input() {
        let weights = checkpoint(&[("w", "F32", 2)]);
        for entry in [
            ("other", "F32", 2),
            ("x", "F32", 3),
            ("x", "F16", 2),
            ("x", "I32", 2),
        ] {
            assert!(preflight(&manifest(), &weights, &checkpoint(&[entry]), "x").is_err());
        }
    }

    #[test]
    fn tensor_byte_limits_include_output_before_native_loading() {
        for shape in [vec![], vec![0, 3], vec![(MAX_PAYLOAD_BYTES / 4) as i64]] {
            check_f32_shape(&shape).unwrap();
        }
        let weights = checkpoint(&[("w", "F32", 2)]);
        let input = checkpoint(&[("x", "F32", 2)]);
        for shape in [
            vec![-1],
            vec![0, -1],
            vec![i64::MAX, i64::MAX],
            vec![(MAX_PAYLOAD_BYTES / 4 + 1) as i64],
        ] {
            assert!(check_f32_shape(&shape).is_err());
            let mut m = manifest();
            m.output_shape = shape;
            assert!(preflight(&m, &weights, &input, "x").is_err());
        }
        assert_eq!(weights.stats().reads, 0);
        assert_eq!(input.stats().reads, 0);
    }
}
