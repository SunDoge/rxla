use super::*;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_nn::{Model, ModelInput};

#[test]
#[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
fn prepared_schedule_matches_original_steps() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let client = unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH")?) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    for (batch, dtype) in [
        (1, DType::F32),
        (2, DType::F32),
        (1, DType::BF16),
        (2, DType::BF16),
    ] {
        let config = QwenImage21Config::tiny(2, 3).compute_dtype(dtype);
        let model = Model::new(|cx: Cx, sample, timestep, step_size, context| {
            qwen_image_2_1_euler_step(cx, &sample, &timestep, &step_size, &context, &config)
        })
        .inputs((
            ModelInput::new([batch, 6, 8]),
            ModelInput::new([batch]),
            ModelInput::new([batch]),
            ModelInput::new([batch, 5, 24]),
        ))
        .trace()?;
        let weights = model
            .schema()
            .parameters()
            .iter()
            .map(|parameter| {
                let len = parameter.shape().iter().product::<i64>() as usize;
                let values = (0..len)
                    .map(|i| {
                        if parameter.path().contains("norm")
                            && !parameter.path().contains("norm_out")
                        {
                            1.0
                        } else {
                            (i as f32 * 0.17).sin() * 0.03
                        }
                    })
                    .collect::<Vec<_>>();
                Ok((
                    parameter.path().to_owned(),
                    client.buffer(parameter.shape(), &values)?,
                ))
            })
            .collect::<std::result::Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        let bindings = || weights.iter().map(|(name, buffer)| (name.as_str(), buffer));
        let stages = model.compile_staged(
            &mut compiler,
            &["schedule_modulation", "schedule_output_scale"],
        )?;
        assert_eq!(stages.preparation_inputs(), [1]);
        assert_eq!(stages.execution_inputs(), [0, 2, 3]);
        let staged = stages.bind_parameters(bindings())?;
        // Compiling the original after splitting also checks source immutability.
        let original = model.compile(&mut compiler)?.bind_parameters(bindings())?;
        let context = client.buffer(&[batch, 5, 24], &vec![0.2_f32; batch as usize * 120])?;
        let mut reference = client.buffer(&[batch, 6, 8], &vec![0.1_f32; batch as usize * 48])?;
        let mut actual = reference.clone();
        for time in [1000.0_f32, 700.0, 400.0, 100.0] {
            let timestep = client.buffer(&[batch], &vec![time; batch as usize])?;
            let step_size = client.buffer(&[batch], &vec![-0.25_f32; batch as usize])?;
            let prepared = staged.prepare(&timestep)?;
            reference = original.run((&reference, &timestep, &step_size, &context))?;
            actual = staged.run((&actual, &step_size, &context), &prepared)?;
            for (a, b) in actual
                .to_vec::<f32>()?
                .iter()
                .zip(reference.to_vec::<f32>()?)
            {
                assert!(
                    (a - b).abs() < 1e-5,
                    "batch={batch} time={time}: {a} != {b}"
                );
            }
            assert!(staged.prepare((&timestep, &step_size)).is_err());
            assert!(
                staged
                    .run::<_, Buffer>((&actual, &step_size), &prepared)
                    .is_err()
            );
            let other = model
                .compile_staged(
                    &mut compiler,
                    &["schedule_modulation", "schedule_output_scale"],
                )?
                .bind_parameters(bindings())?;
            assert!(matches!(
                other.run::<_, Buffer>((&actual, &step_size, &context), &prepared),
                Err(rxla_nn::Error::PreparedModelMismatch)
            ));
        }
    }
    Ok(())
}
