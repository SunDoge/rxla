use super::*;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_nn::{AppliedModel, Model, ModelInput};

fn trace(config: &QwenImage21Config) -> Result<AppliedModel> {
    Model::new(|cx: Cx, sample, timestep, step_size, context| {
        qwen_image_2_1_euler_step(cx, &sample, &timestep, &step_size, &context, config)
    })
    .inputs((
        ModelInput::new([2, 6, 8]),
        ModelInput::new([2]),
        ModelInput::new([2]),
        ModelInput::new([2, 5, 24]),
    ))
    .trace()
}

#[test]
fn fused_projections_preserve_schema_and_reduce_contractions()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    for config in [
        QwenImage21Config::tiny(2, 3),
        QwenImage21Config::tiny_w8(2, 3),
    ] {
        let baseline = trace(&config)?;
        for (qkv, mlp, expected_dots) in [(true, false, 26), (false, true, 28), (true, true, 24)] {
            let fused = trace(&config.clone().fuse_qkv(qkv).fuse_mlp(mlp))?;
            let schema = |model: &AppliedModel| {
                model
                    .schema()
                    .parameters()
                    .iter()
                    .map(|p| (p.path().to_owned(), p.shape().to_vec(), p.storage_dtype()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(schema(&baseline), schema(&fused));
            let lowered = fused.prepare()?;
            let code = std::str::from_utf8(lowered.code())?;
            assert_eq!(code.matches("stablehlo.dot_general").count(), expected_dots);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
fn fused_projections_match_original_on_cpu() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let client = unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH")?) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    for quantized in [false, true] {
        for dtype in [DType::F32, DType::BF16] {
            let config = if quantized {
                QwenImage21Config::tiny_w8(2, 3)
            } else {
                QwenImage21Config::tiny(2, 3)
            }
            .compute_dtype(dtype);
            let baseline = trace(&config)?;
            let weights = baseline
                .schema()
                .parameters()
                .iter()
                .map(|p| {
                    let len = p.shape().iter().product::<i64>() as usize;
                    let buffer = if p.storage_dtype() == DType::U8 {
                        client.buffer(
                            p.shape(),
                            &(0..len).map(|i| 125 + (i % 7) as u8).collect::<Vec<_>>(),
                        )?
                    } else {
                        let values = (0..len)
                            .map(|i| {
                                if p.path().contains("norm") && !p.path().contains("norm_out") {
                                    1.0_f32
                                } else if p.path().ends_with("scale") {
                                    0.03
                                } else {
                                    (i as f32 * 0.17).sin() * 0.03
                                }
                            })
                            .collect::<Vec<_>>();
                        client.buffer(p.shape(), &values)?
                    };
                    Ok((p.path().to_owned(), buffer))
                })
                .collect::<std::result::Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            let bindings = || weights.iter().map(|(name, buffer)| (name.as_str(), buffer));
            let input = client.buffer(
                &[2, 6, 8],
                &(0..96).map(|i| (i as f32 * 0.11).sin()).collect::<Vec<_>>(),
            )?;
            let context = client.buffer(
                &[2, 5, 24],
                &(0..240)
                    .map(|i| (i as f32 * 0.13).cos())
                    .collect::<Vec<_>>(),
            )?;
            let time = client.buffer(&[2], &[0.8_f32, 0.4])?;
            let step_size = client.buffer(&[2], &[-0.1_f32, -0.2])?;
            let runner = baseline
                .compile(&mut compiler)?
                .bind_parameters(bindings())?;
            let reference: Buffer = runner.run((&input, &time, &step_size, &context))?;
            let reference = reference.to_vec::<f32>()?;
            for (qkv, mlp) in [(true, false), (false, true), (true, true)] {
                let fused = trace(&config.clone().fuse_qkv(qkv).fuse_mlp(mlp))?;
                let runner = fused.compile(&mut compiler)?.bind_parameters(bindings())?;
                let output: Buffer = runner.run((&input, &time, &step_size, &context))?;
                let actual = output.to_vec::<f32>()?;
                assert_eq!(actual.len(), reference.len());
                assert!(
                    actual
                        .iter()
                        .chain(&reference)
                        .all(|value| value.is_finite())
                );
                let error = actual
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                let tolerance = if dtype == DType::F32 { 1e-5 } else { 2.5e-3 };
                assert!(
                    error < tolerance,
                    "quantized={quantized} dtype={dtype:?} qkv={qkv} mlp={mlp} error={error}"
                );
            }
        }
    }
    Ok(())
}
