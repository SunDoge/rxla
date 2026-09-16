use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PndmSampleSource {
    Current,
    Initial,
}

/// One device-side PLMS update description.
///
/// `model_coefficients` multiply the current guided noise prediction followed
/// by as many previous retained predictions as are nonzero, newest first.
#[derive(Clone, Copy, Debug)]
pub struct PndmStep {
    pub timestep: i64,
    pub sample_source: PndmSampleSource,
    pub sample_coefficient: f32,
    pub model_coefficient: f32,
    pub model_coefficients: [f32; 5],
    pub retain_model_output: bool,
}

impl PndmStep {
    /// Pack coefficients for one fused Tensor program. The last value is the
    /// classifier-free guidance scale, deliberately runtime-configurable.
    pub fn tensor_coefficients(self, guidance_scale: f32) -> Result<[f32; 8]> {
        if !guidance_scale.is_finite() {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::NonFiniteGuidanceScale,
            });
        }
        Ok([
            self.sample_coefficient,
            self.model_coefficient,
            self.model_coefficients[0],
            self.model_coefficients[1],
            self.model_coefficients[2],
            self.model_coefficients[3],
            self.model_coefficients[4],
            guidance_scale,
        ])
    }
}

/// Diffusers-compatible PNDM/PLMS schedule used by Stable Diffusion.
///
/// Planning is pure host arithmetic; tensor payloads and the four retained noise
/// predictions can remain on the device. This implements `scaled_linear` betas,
/// leading timestep spacing, `steps_offset=1`, `skip_prk_steps=true`, epsilon
/// prediction, and `set_alpha_to_one=false` from the tiny reference checkpoint.
#[derive(Clone, Debug)]
pub struct PndmScheduler {
    timesteps: Vec<i64>,
    alphas_cumprod: Vec<f32>,
    step_ratio: i64,
}

impl PndmScheduler {
    pub fn stable_diffusion(inference_steps: usize) -> Result<Self> {
        const TRAIN_STEPS: usize = 1000;
        if !(2..=TRAIN_STEPS).contains(&inference_steps) {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidPndmInferenceSteps,
            });
        }
        let step_ratio = TRAIN_STEPS / inference_steps;
        if step_ratio == 0 {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidPndmTimestepRatio,
            });
        }
        let base: Vec<i64> = (0..inference_steps)
            .map(|index| {
                i64::try_from(index * step_ratio + 1).map_err(|_| Error::InvalidModel {
                    kind: ModelDefinitionError::PndmTimestepTooLarge,
                })
            })
            .collect::<Result<_>>()?;
        let mut timesteps = Vec::with_capacity(inference_steps + 1);
        timesteps.push(base[inference_steps - 1]);
        timesteps.push(base[inference_steps - 2]);
        timesteps.extend(base[..inference_steps - 1].iter().rev().copied());

        let beta_start = 0.00085_f32.sqrt();
        let beta_end = 0.012_f32.sqrt();
        let mut product = 1_f32;
        let mut alphas_cumprod = Vec::with_capacity(TRAIN_STEPS);
        for index in 0..TRAIN_STEPS {
            let fraction = index as f32 / (TRAIN_STEPS - 1) as f32;
            let beta = (beta_start + (beta_end - beta_start) * fraction).powi(2);
            product *= 1. - beta;
            alphas_cumprod.push(product);
        }
        Ok(Self {
            timesteps,
            alphas_cumprod,
            step_ratio: i64::try_from(step_ratio).map_err(|_| Error::InvalidModel {
                kind: ModelDefinitionError::PndmTimestepRatioTooLarge,
            })?,
        })
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    pub fn step(&self, index: usize) -> Result<PndmStep> {
        let &listed_timestep = self.timesteps.get(index).ok_or(Error::InvalidModel {
            kind: ModelDefinitionError::PndmStepOutOfRange,
        })?;
        let (timestep, previous, sample_source) = if index == 1 {
            (
                listed_timestep
                    .checked_add(self.step_ratio)
                    .ok_or(Error::InvalidModel {
                        kind: ModelDefinitionError::PndmTimestepOverflow,
                    })?,
                listed_timestep,
                PndmSampleSource::Initial,
            )
        } else {
            (
                listed_timestep,
                listed_timestep - self.step_ratio,
                PndmSampleSource::Current,
            )
        };
        let alpha = self.alpha(timestep)?;
        let alpha_previous = if previous >= 0 {
            self.alpha(previous)?
        } else {
            self.alphas_cumprod[0]
        };
        let beta = 1. - alpha;
        let beta_previous = 1. - alpha_previous;
        let sample_coefficient = (alpha_previous / alpha).sqrt();
        let denominator = alpha * beta_previous.sqrt() + (alpha * beta * alpha_previous).sqrt();
        let model_coefficient = -(alpha_previous - alpha) / denominator;
        if !sample_coefficient.is_finite() || !model_coefficient.is_finite() {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::NonFinitePndmCoefficients,
            });
        }
        let model_coefficients = match index {
            0 => [1., 0., 0., 0., 0.],
            1 => [0.5, 0.5, 0., 0., 0.],
            2 => [1.5, -0.5, 0., 0., 0.],
            3 => [23. / 12., -16. / 12., 5. / 12., 0., 0.],
            _ => [55. / 24., -59. / 24., 37. / 24., -9. / 24., 0.],
        };
        Ok(PndmStep {
            timestep: listed_timestep,
            sample_source,
            sample_coefficient,
            model_coefficient,
            model_coefficients,
            retain_model_output: index != 1,
        })
    }

    fn alpha(&self, timestep: i64) -> Result<f32> {
        usize::try_from(timestep)
            .ok()
            .and_then(|index| self.alphas_cumprod.get(index).copied())
            .ok_or(Error::InvalidModel {
                kind: ModelDefinitionError::PndmTimestepOutsideSchedule,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_step_schedule_matches_diffusers() {
        let scheduler = PndmScheduler::stable_diffusion(5).unwrap();
        assert_eq!(scheduler.timesteps(), [801, 601, 601, 401, 201, 1]);
        let expected = [
            (2.0911621, -1.1359788),
            (2.0911621, -1.1359788),
            (1.626_667, -0.7313447),
            (1.3336473, -0.5152973),
            (1.1520715, -0.5322822),
            (1.0004276, -0.01214178),
        ];
        for (index, expected) in expected.into_iter().enumerate() {
            let step = scheduler.step(index).unwrap();
            assert!((step.sample_coefficient - expected.0).abs() < 2e-6);
            assert!((step.model_coefficient - expected.1).abs() < 2e-6);
        }
        assert_eq!(
            scheduler.step(1).unwrap().sample_source,
            PndmSampleSource::Initial
        );
        assert!(!scheduler.step(1).unwrap().retain_model_output);
        assert!(matches!(
            scheduler.step(6),
            Err(Error::InvalidModel {
                kind: ModelDefinitionError::PndmStepOutOfRange
            })
        ));
    }
}
