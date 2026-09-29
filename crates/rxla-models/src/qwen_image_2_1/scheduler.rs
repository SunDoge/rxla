use super::{Error, ModelDefinitionError, Result};

/// One deterministic FlowMatch Euler update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenImage21FlowStep {
    /// Normalized model timestep in `[0, 1]`.
    pub timestep: f32,
    /// `next_sigma - sigma`, multiplied by the model prediction.
    pub step_size: f32,
}

/// Official Qwen-Image-2.1 resolution-dependent FlowMatch schedule.
#[derive(Clone, Debug)]
pub struct QwenImage21FlowScheduler {
    sigmas: Vec<f32>,
}

impl QwenImage21FlowScheduler {
    const BASE_SEQUENCE_LENGTH: f32 = 256.0;
    const MAX_SEQUENCE_LENGTH: f32 = 8192.0;
    const BASE_SHIFT: f32 = 0.5;
    const MAX_SHIFT: f32 = 0.9;
    const TERMINAL_SIGMA: f32 = 0.02;
    /// Build the default Diffusers schedule for a target latent sequence.
    pub fn new(steps: usize, image_tokens: usize) -> Result<Self> {
        if steps == 0 || image_tokens == 0 {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidQwenImage21Schedule,
            });
        }
        let image_tokens = image_tokens as f32;
        let slope = (Self::MAX_SHIFT - Self::BASE_SHIFT)
            / (Self::MAX_SEQUENCE_LENGTH - Self::BASE_SEQUENCE_LENGTH);
        let intercept = Self::BASE_SHIFT - slope * Self::BASE_SEQUENCE_LENGTH;
        let mu = slope * image_tokens + intercept;
        let exponential = mu.exp();

        // Qwen-Image-2.1 supplies custom sigmas `linspace(1, 1/steps,
        // steps)` rather than using the scheduler's training-time 0.001 tail.
        let denominator = steps as f32;
        let mut sigmas = (0..steps)
            .map(|index| {
                let sigma = 1.0 - index as f32 / denominator;
                exponential / (exponential + (1.0 / sigma - 1.0))
            })
            .collect::<Vec<_>>();

        // Match FlowMatchEulerDiscreteScheduler::stretch_shift_to_terminal.
        let scale = (1.0 - sigmas[sigmas.len() - 1]) / (1.0 - Self::TERMINAL_SIGMA);
        for sigma in &mut sigmas {
            *sigma = 1.0 - (1.0 - *sigma) / scale;
        }
        sigmas.push(0.0);
        if sigmas.iter().any(|sigma| !sigma.is_finite()) {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidQwenImage21Schedule,
            });
        }
        Ok(Self { sigmas })
    }

    pub fn len(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn step(&self, index: usize) -> Result<QwenImage21FlowStep> {
        let Some((&sigma, &next_sigma)) = self.sigmas.get(index).zip(self.sigmas.get(index + 1))
        else {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::QwenImage21StepOutOfRange,
            });
        };
        Ok(QwenImage21FlowStep {
            timestep: sigma,
            step_size: next_sigma - sigma,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forty_step_2048_square_schedule_matches_diffusers() {
        // 2048px / spatial factor 16 = 128, hence 16,384 latent tokens.
        let scheduler = QwenImage21FlowScheduler::new(40, 128 * 128).unwrap();
        assert_eq!(scheduler.len(), 40);
        assert_eq!(scheduler.sigmas().len(), 41);
        for (actual, expected) in scheduler.sigmas()[..5].iter().zip([
            1.0,
            0.992_645_98,
            0.985_012_95,
            0.977_084_8,
            0.968_844_1,
        ]) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert!((scheduler.sigmas()[39] - 0.02).abs() < 1e-6);
        assert_eq!(scheduler.sigmas()[40], 0.0);
        assert!((scheduler.step(39).unwrap().step_size + 0.02).abs() < 1e-6);
    }
}
