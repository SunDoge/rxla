use super::{Error, ModelDefinitionError, Result};
use rxla_core::Tensor;

/// One rectified-flow Euler update in the FLUX.2 inference schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flux2EulerStep {
    pub timestep: f32,
    pub next_timestep: f32,
}

impl Flux2EulerStep {
    /// Keep the sample and prediction on-device; only the scalar step size is
    /// host-planned and embedded into the traced program.
    pub fn update(self, sample: &Tensor, prediction: &Tensor) -> Result<Tensor> {
        Ok(sample.add(&prediction.mul_scalar(self.next_timestep - self.timestep)?)?)
    }
}

/// Official image-sequence-dependent FLUX.2 time-shifted Euler schedule.
#[derive(Clone, Debug)]
pub struct Flux2EulerScheduler {
    timesteps: Vec<f32>,
}

impl Flux2EulerScheduler {
    pub fn new(steps: usize, image_sequence_length: usize) -> Result<Self> {
        if steps == 0 || image_sequence_length == 0 {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidFlux2Schedule,
            });
        }
        let mu = empirical_mu(image_sequence_length as f64, steps as f64);
        let exp_mu = mu.exp();
        let mut timesteps = Vec::with_capacity(steps + 1);
        for index in 0..=steps {
            let timestep = 1.0 - index as f64 / steps as f64;
            let shifted = if index == steps {
                0.0
            } else {
                exp_mu / (exp_mu + (1.0 / timestep - 1.0))
            };
            timesteps.push(shifted as f32);
        }
        Ok(Self { timesteps })
    }

    /// Distilled Klein models use four steps.
    pub fn klein_4b(image_sequence_length: usize) -> Result<Self> {
        Self::new(4, image_sequence_length)
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub fn step(&self, index: usize) -> Result<Flux2EulerStep> {
        let Some((&timestep, &next_timestep)) =
            self.timesteps.get(index).zip(self.timesteps.get(index + 1))
        else {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::Flux2StepOutOfRange,
            });
        };
        Ok(Flux2EulerStep {
            timestep,
            next_timestep,
        })
    }
}

fn empirical_mu(image_sequence_length: f64, steps: f64) -> f64 {
    const A1: f64 = 8.738_095_24e-5;
    const B1: f64 = 1.898_333_33;
    const A2: f64 = 0.000_169_27;
    const B2: f64 = 0.456_666_66;
    if image_sequence_length > 4300.0 {
        return A2 * image_sequence_length + B2;
    }
    let m_200 = A2 * image_sequence_length + B2;
    let m_10 = A1 * image_sequence_length + B1;
    let slope = (m_200 - m_10) / 190.0;
    let intercept = m_200 - 200.0 * slope;
    slope * steps + intercept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_step_4096_token_schedule_matches_reference() {
        let scheduler = Flux2EulerScheduler::klein_4b(4096).unwrap();
        let expected = [1.0, 0.967_384, 0.908_143_94, 0.767_199_93, 0.0];
        for (actual, expected) in scheduler.timesteps().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(scheduler.step(0).unwrap().timestep, 1.0);
        assert!(matches!(
            scheduler.step(4),
            Err(Error::InvalidModel {
                kind: ModelDefinitionError::Flux2StepOutOfRange
            })
        ));
    }

    #[test]
    fn schedule_switches_to_large_sequence_fit() {
        let scheduler = Flux2EulerScheduler::klein_4b(4352).unwrap();
        assert!((scheduler.timesteps()[1] - 0.908_207_5).abs() < 1e-6);
    }
}
