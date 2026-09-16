//! Restricted exported OCR graph runner. Not a general ONNX importer.
use rxla_core::{Client, Conv2dOptions, ConvTranspose2dOptions, Pool2dOptions, Tensor, Tracer};
use rxla_safetensors::SafeTensors;
use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};
mod ocr_artifact;
mod ocr_publish;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Deserialize)]
struct Bundle {
    version: u32,
    probe: Option<Value>,
    input_name: String,
    input_shape: Vec<i64>,
    output_name: String,
    weights: Vec<String>,
    nodes: Vec<Node>,
    cases: usize,
}
#[derive(Deserialize)]
struct Node {
    name: String,
    op: String,
    inputs: Vec<String>,
    output: String,
    attributes: Value,
}

fn integers(a: &Value, name: &str, default: &[i64]) -> Result<Vec<i64>> {
    match a.get(name) {
        None => Ok(default.to_vec()),
        Some(value) => value
            .as_array()
            .ok_or("attribute must be an array")?
            .iter()
            .map(|v| {
                v.as_i64()
                    .ok_or_else(|| "attribute must contain integers".into())
            })
            .collect(),
    }
}
fn pair(a: &Value, name: &str, default: [i64; 2]) -> Result<[i64; 2]> {
    integers(a, name, &default)?
        .try_into()
        .map_err(|_| "attribute must have two values".into())
}
fn integer(a: &Value, name: &str, default: i64) -> Result<i64> {
    a.get(name).map_or(Ok(default), |v| {
        v.as_i64().ok_or_else(|| "attribute must be integer".into())
    })
}
fn coefficient(a: &Value, name: &str, default: f32) -> Result<f32> {
    a.get(name).map_or(Ok(default), |v| {
        v.as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| "attribute must be numeric".into())
    })
}
fn padding(
    a: &Value,
    spatial: &[i64],
    kernel: [i64; 2],
    strides: [i64; 2],
    dilation: [i64; 2],
) -> Result<[[i64; 2]; 2]> {
    match a
        .get("auto_pad")
        .and_then(Value::as_str)
        .unwrap_or("NOTSET")
    {
        "NOTSET" => {
            let pads: [i64; 4] = integers(a, "pads", &[0; 4])?
                .try_into()
                .map_err(|_| "expected 4 pads")?;
            Ok([[pads[0], pads[2]], [pads[1], pads[3]]])
        }
        "SAME_UPPER" => {
            let mut pads = [[0; 2]; 2];
            for axis in 0..2 {
                if strides[axis] <= 0 || dilation[axis] <= 0 || kernel[axis] <= 0 {
                    return Err("invalid window".into());
                }
                let output =
                    (spatial[axis] as i128 + strides[axis] as i128 - 1) / strides[axis] as i128;
                let extent = (kernel[axis] as i128 - 1) * dilation[axis] as i128 + 1;
                let total =
                    ((output - 1) * strides[axis] as i128 + extent - spatial[axis] as i128).max(0);
                let total = i64::try_from(total)?;
                pads[axis] = [total / 2, total - total / 2];
            }
            Ok(pads)
        }
        _ => Err("unsupported auto_pad".into()),
    }
}
fn broadcast(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let rank = a.shape().len().max(b.shape().len());
    let mut dims = vec![1; rank];
    for shape in [a.shape(), b.shape()] {
        for (axis, &size) in shape.iter().enumerate() {
            let slot = &mut dims[rank - shape.len() + axis];
            if *slot == 1 {
                *slot = size;
            } else if size != 1 && size != *slot {
                return Err("incompatible broadcast".into());
            }
        }
    }
    Ok((a.broadcast_to(&dims)?, b.broadcast_to(&dims)?))
}
fn lower(node: &Node, input: &[Tensor]) -> Result<Tensor> {
    let a = &node.attributes;
    let expected = match node.op.as_str() {
        "Add" | "Mul" | "Div" => 2..=2,
        "Conv" | "ConvTranspose" => 2..=3,
        "Concat" => 1..=usize::MAX,
        _ => 1..=1,
    };
    if !expected.contains(&input.len()) {
        return Err("wrong node arity".into());
    }
    let x = &input[0];
    Ok(match node.op.as_str() {
        "Relu" => x.relu()?,
        "Sigmoid" => x.sigmoid()?,
        "Erf" => x.erf()?,
        "HardSigmoid" => {
            x.hard_sigmoid(coefficient(a, "alpha", 0.2)?, coefficient(a, "beta", 0.5)?)?
        }
        "Add" | "Mul" | "Div" => {
            let (x, y) = broadcast(x, &input[1])?;
            match node.op.as_str() {
                "Add" => x.add(&y)?,
                "Mul" => x.mul(&y)?,
                _ => x.div(&y)?,
            }
        }
        "GlobalAveragePool" => {
            if x.shape().len() != 4 {
                return Err("expected NCHW".into());
            }
            x.mean(&[2, 3], true)?
        }
        "ReduceMean" => {
            let axes: Vec<usize> = integers(a, "axes", &[])?
                .into_iter()
                .map(usize::try_from)
                .collect::<std::result::Result<_, _>>()?;
            if axes.is_empty() {
                return Err("explicit reduction axes required".into());
            }
            x.mean(&axes, integer(a, "keepdims", 1)? == 1)?
        }
        "Concat" => Tensor::concatenate(input, usize::try_from(integer(a, "axis", 1)?)?)?,
        "Resize" => x
            .transpose(&[0, 2, 3, 1])?
            .upsample_nearest2d(pair(a, "scales", [1, 1])?)?
            .transpose(&[0, 3, 1, 2])?,
        "Conv" | "ConvTranspose" | "MaxPool" => {
            if x.shape().len() != 4 {
                return Err("expected NCHW".into());
            }
            let strides = pair(a, "strides", [1, 1])?;
            let dilation = pair(a, "dilations", [1, 1])?;
            let kernel = pair(a, "kernel_shape", [1, 1])?;
            let pads = padding(a, &x.shape()[2..], kernel, strides, dilation)?;
            let x = x.transpose(&[0, 2, 3, 1])?;
            let mut y = if node.op == "MaxPool" {
                if integer(a, "ceil_mode", 0)? != 0 {
                    return Err("ceil mode unsupported".into());
                }
                x.max_pool2d(Pool2dOptions {
                    window: kernel,
                    strides,
                    padding: pads,
                })?
            } else {
                let w = &input[1];
                if w.shape().len() != 4 || w.shape()[2..] != kernel {
                    return Err("kernel shape mismatch".into());
                }
                let w = w.transpose(&[2, 3, 1, 0])?;
                let groups = integer(a, "group", 1)?;
                if node.op == "Conv" {
                    x.conv2d(
                        &w,
                        Conv2dOptions {
                            strides,
                            dilation,
                            padding: pads,
                            groups,
                        },
                    )?
                } else {
                    if groups != 1 {
                        return Err("grouped transposed convolution unsupported".into());
                    }
                    x.conv_transpose2d(
                        &w,
                        ConvTranspose2dOptions {
                            strides,
                            dilation,
                            padding: pads,
                            output_padding: [0, 0],
                        },
                    )?
                }
            };
            if input.len() == 3 {
                if input[2].shape() != [y.shape()[3]] {
                    return Err("invalid convolution bias".into());
                }
                y = y.add(&input[2].broadcast_to(y.shape())?)?;
            }
            y.transpose(&[0, 3, 1, 2])?
        }
        op => return Err(format!("unsupported operation: {op}").into()),
    })
}

fn benchmark_options(args: &[std::ffi::OsString]) -> Result<(bool, usize, Option<PathBuf>)> {
    let mut include_output = false;
    let mut runs = 0;
    let mut export = None;
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        if flag == "--include-output" && !include_output {
            include_output = true;
        } else if flag == "--benchmark-runs" && runs == 0 {
            runs = args
                .next()
                .and_then(|s| s.to_str())
                .ok_or("missing benchmark count")?
                .parse()?;
            if !(1..=1000).contains(&runs) {
                return Err("benchmark count must be in 1..=1000".into());
            }
        } else if flag == "--export-artifact" && export.is_none() {
            export = Some(PathBuf::from(
                args.next().ok_or("missing artifact directory")?,
            ));
        } else {
            return Err("unknown or duplicate option".into());
        }
    }
    Ok((include_output, runs, export))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() < 2 {
        return Err(
            "usage: ocr_graph BUNDLE REPORT.json [--include-output] [--benchmark-runs N]".into(),
        );
    }
    let (include_output, benchmark_runs, export) = benchmark_options(&args[2..])?;
    if export.as_ref().is_some_and(|path| path.exists()) {
        return Err("artifact directory already exists".into());
    }
    let directory = PathBuf::from(&args[0]);
    let report_path = PathBuf::from(&args[1]);
    if report_path.exists() {
        return Err("report already exists".into());
    }
    let bundle: Bundle = serde_json::from_reader(File::open(directory.join("graph.json"))?)?;
    if bundle.version != 1 || bundle.cases == 0 {
        return Err("invalid fixture version/cases".into());
    }
    let plugin_path = std::fs::canonicalize(std::env::var("PJRT_PLUGIN_PATH")?)?;
    // Deployment assumes this trusted file remains immutable through loading
    // and export. Its dependencies are not fingerprinted by this check.
    let plugin_digest = export
        .as_ref()
        .map(|_| ocr_artifact::plugin_fingerprint(&plugin_path))
        .transpose()?;
    let client = unsafe { Client::load(&plugin_path)? };
    let info = client.info()?;
    if !matches!(info.platform.as_str(), "cpu" | "cuda") {
        return Err("CPU or CUDA validation only".into());
    }
    let mut checkpoint = SafeTensors::open(directory.join("tensors.safetensors"))?;
    let graph = Tracer::default();
    let mut values = HashMap::from([(bundle.input_name, graph.input(&bundle.input_shape)?)]);
    let mut weights = Vec::new();
    for name in &bundle.weights {
        let buffer = checkpoint.upload_f32(&client, &format!("weight/{name}"))?;
        if values
            .insert(name.clone(), graph.input(&buffer.dimensions()?)?)
            .is_some()
        {
            return Err("duplicate weight/input".into());
        }
        weights.push(buffer);
    }
    for node in &bundle.nodes {
        let inputs: Vec<_> = node
            .inputs
            .iter()
            .map(|name| {
                values
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("missing value {name}"))
            })
            .collect::<std::result::Result<_, _>>()?;
        let output =
            lower(node, &inputs).map_err(|e| format!("{} ({}): {e}", node.name, node.op))?;
        if values.insert(node.output.clone(), output).is_some() {
            return Err("duplicate node output".into());
        }
    }
    let output = values
        .get(&bundle.output_name)
        .ok_or("missing graph output")?;
    eprintln!(
        "Built {} nodes; output {:?}; compiling",
        bundle.nodes.len(),
        output.shape()
    );
    let start = Instant::now();
    let executable = graph.compile(&client, output)?;
    let compile_seconds = start.elapsed().as_secs_f64();
    if let Some(directory) = &export {
        // This experimental export is trusted native code, not a portable model.
        let native = executable.serialize()?;
        let mut manifest = serde_json::json!({
            "version": 4, "platform": info.platform,
            "plugin_blake3": plugin_digest.as_ref().ok_or("missing export plugin fingerprint")?.to_hex().to_string(),
            "executable_blake3": blake3::hash(&native).to_hex().to_string(),
            "probe": bundle.probe,
            "platform_version": info.platform_version,
            "api_major": info.api_major, "api_minor": info.api_minor,
            "device_kind": info.addressable_devices.iter().find(|d| d.selected).ok_or("missing selected device")?.kind,
            "input_shape": bundle.input_shape, "output_shape": output.shape(),
            "weights": bundle.weights.iter().zip(&weights).map(|(name, buffer)| {
                Ok(serde_json::json!({"name": name, "shape": buffer.dimensions()?}))
            }).collect::<Result<Vec<_>>>()?,
            "scope": "Experimental trusted same-environment native export. Diagnostic metadata is not a complete ABI/hardware/runtime fingerprint or a signature. Numerical acceptance is reported separately."
        });
        ocr_publish::publish_new(directory, |directory| {
            std::fs::write(directory.join("executable.bin"), native)?;
            let named_weights: Vec<_> = bundle
                .weights
                .iter()
                .zip(&weights)
                .map(|(name, buffer)| (name.as_str(), buffer))
                .collect();
            rxla_safetensors::save_buffers_new(
                directory.join("weights.safetensors"),
                &named_weights,
                None,
            )?;
            manifest["weights_blake3"] =
                blake3::hash(&std::fs::read(directory.join("weights.safetensors"))?)
                    .to_hex()
                    .to_string()
                    .into();
            serde_json::to_writer_pretty(
                File::create_new(directory.join("manifest.json"))?,
                &manifest,
            )?;
            Ok(())
        })?;
    }
    let mut comparisons = Vec::new();
    let mut outputs = include_output.then(Vec::new);
    let mut benchmarks = Vec::new();
    for case in 0..bundle.cases {
        let input = checkpoint.upload_f32(&client, &format!("input/{case}"))?;
        let expected = checkpoint.read_f32(&format!("expected/{case}"))?;
        if expected.shape != output.shape() {
            return Err("reference shape mismatch".into());
        }
        let mut args = vec![&input];
        args.extend(weights.iter());
        let actual = executable.execute(&args)?[0].to_vec::<f32>()?;
        let comparison = compare(&actual, &expected.values)?;
        eprintln!(
            "Case {case}: max error {}, mismatches {}, zero-baseline mismatches {}",
            comparison.max_abs_error, comparison.mismatches, comparison.zero_baseline_mismatches
        );
        comparisons.push(comparison);
        if benchmark_runs > 0 {
            let host_input = checkpoint.read_f32(&format!("input/{case}"))?;
            let mut resident_ms = Vec::with_capacity(benchmark_runs);
            let mut roundtrip_ms = Vec::with_capacity(benchmark_runs);
            for trial in 0..=benchmark_runs {
                let start = Instant::now();
                let result = executable.execute(&args)?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                // Execute waits for completion. Verification download is outside
                // this interval, but may affect the next trial's cache state.
                if result[0].to_vec::<f32>()? != actual {
                    return Err("resident benchmark differs from correctness output".into());
                }
                if trial > 0 {
                    resident_ms.push(elapsed);
                }
            }
            for trial in 0..=benchmark_runs {
                let start = Instant::now();
                let uploaded = client.buffer(&host_input.shape, &host_input.values)?;
                let mut inputs = vec![&uploaded];
                inputs.extend(weights.iter());
                let result = executable.execute(&inputs)?;
                let downloaded = result[0].to_vec::<f32>()?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                if downloaded != actual {
                    return Err("roundtrip benchmark differs from correctness output".into());
                }
                if trial > 0 {
                    roundtrip_ms.push(elapsed);
                }
            }
            benchmarks.push(serde_json::json!({
                "case": case, "input_shape": host_input.shape,
                "warmup_runs_per_mode": 1, "resident_execute_ms": resident_ms,
                "upload_execute_download_ms": roundtrip_ms,
                "outputs_match_correctness_run": true,
            }));
        }
        if let Some(outputs) = &mut outputs {
            outputs.push(actual);
        }
    }
    let passed = comparisons
        .iter()
        .all(|c| c.mismatches == 0 && c.zero_baseline_mismatches > 0);
    let report = serde_json::json!({"passed": passed, "bundle": directory, "nodes": bundle.nodes.len(), "cases": bundle.cases, "comparisons": comparisons, "outputs": outputs, "absolute_tolerance": 2e-7, "relative_tolerance": 1e-4, "compile_seconds": compile_seconds, "output_shape": output.shape(), "pjrt_platform": info.platform, "pjrt_platform_version": info.platform_version, "pjrt_devices": info.addressable_devices.iter().map(|d| serde_json::json!({"id": d.id, "kind": d.kind, "selected": d.selected})).collect::<Vec<_>>(), "protocol": "FP32 Rust CPU/CUDA outputs vs ONNX Runtime CPU fixtures; no OCR postprocessing or performance comparison"});
    let mut report = report;
    report["probe"] = serde_json::to_value(&bundle.probe)?;
    report["benchmark"] = serde_json::json!({
        "cases": benchmarks,
        "scope": "Optional same-input repeated trials with one warmup per mode. Resident mode measures synchronous PJRT execute with resident input and weights, including launch/completion and output allocation; verification download is outside timing. Roundtrip mode includes input allocation/upload, execution and full output download. Both exclude model loading, compilation, preprocessing, postprocessing, file IO, output verification and buffer destruction. Modes run in separate blocks; not a pure GPU kernel timer or a cross-backend comparison. Strict ORT agreement status is unchanged."
    });
    serde_json::to_writer_pretty(File::create_new(report_path)?, &report)?;
    if !passed {
        return Err("OCR graph numerical mismatch".into());
    }
    println!("OCR tensor graph correctness: PASS");
    Ok(())
}

#[derive(serde::Serialize)]
struct Comparison {
    max_abs_error: f32,
    mismatches: usize,
    zero_baseline_mismatches: usize,
    mismatch_examples: Vec<serde_json::Value>,
}
fn compare(actual: &[f32], expected: &[f32]) -> Result<Comparison> {
    if actual.len() != expected.len() || expected.is_empty() {
        return Err("invalid reference size".into());
    }
    let mut result = Comparison {
        max_abs_error: 0.,
        mismatches: 0,
        zero_baseline_mismatches: 0,
        mismatch_examples: Vec::new(),
    };
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err("nonfinite output".into());
        }
        let error = (a - b).abs();
        let tolerance = 2e-7 + 1e-4 * b.abs();
        result.max_abs_error = result.max_abs_error.max(error);
        result.mismatches += usize::from(error > tolerance);
        if error > tolerance && result.mismatch_examples.len() < 16 {
            result.mismatch_examples.push(serde_json::json!({"flat_index": index, "actual": a, "expected": b, "absolute_error": error, "tolerance": tolerance}));
        }
        result.zero_baseline_mismatches += usize::from(b.abs() > tolerance);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn benchmark_option_validation() {
        let parse = |args: &[&str]| {
            super::benchmark_options(
                &args
                    .iter()
                    .map(std::ffi::OsString::from)
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(parse(&[]).unwrap(), (false, 0, None));
        assert_eq!(
            parse(&["--export-artifact", "bundle"]).unwrap(),
            (false, 0, Some("bundle".into()))
        );
        assert!(parse(&["--export-artifact"]).is_err());
        assert!(parse(&["--export-artifact", "a", "--export-artifact", "b"]).is_err());
        assert_eq!(
            parse(&["--include-output", "--benchmark-runs", "20"]).unwrap(),
            (true, 20, None)
        );
        assert_eq!(
            parse(&["--benchmark-runs", "1", "--include-output"]).unwrap(),
            (true, 1, None)
        );
        for args in [
            vec!["--benchmark-runs"],
            vec!["--benchmark-runs", "0"],
            vec!["--benchmark-runs", "1001"],
            vec!["--benchmark-runs", "-1"],
            vec!["--wat"],
            vec!["--include-output", "--include-output"],
            vec!["--benchmark-runs", "1", "--benchmark-runs", "2"],
        ] {
            assert!(parse(&args).is_err());
        }
    }
    use super::*;
    #[test]
    fn mismatch_samples_are_bounded_without_truncating_failure_count() {
        let mut actual = vec![0.; 21];
        actual[0] = 1.;
        let comparison = compare(&actual, &[1.; 21]).unwrap();
        assert_eq!(comparison.mismatches, 20);
        assert_eq!(comparison.mismatch_examples.len(), 16);
        for (offset, sample) in comparison.mismatch_examples.iter().enumerate() {
            assert_eq!(sample["flat_index"], offset + 1);
            assert_eq!(sample["actual"], 0.);
            assert_eq!(sample["expected"], 1.);
            assert_eq!(sample["absolute_error"], 1.);
            assert!(sample["tolerance"].as_f64().unwrap() < 1.);
        }
        assert!(compare(&[1.], &[1.]).unwrap().mismatch_examples.is_empty());
    }
    #[test]
    fn small_probability_gate_rejects_zero_predictor() {
        let reference = [1e-5, 3e-4];
        assert_eq!(compare(&[0., 0.], &reference).unwrap().mismatches, 2);
        let good = compare(&[1.01e-5, 3.001e-4], &reference).unwrap();
        assert_eq!(good.mismatches, 0);
        assert_eq!(good.zero_baseline_mismatches, 2);
        assert!(compare(&[f32::NAN], &[0.]).is_err());
        assert!(compare(&[], &[]).is_err());
    }
}
