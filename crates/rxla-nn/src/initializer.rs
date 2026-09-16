use crate::{Error, Result};
use rxla_core::{Buffer, Client, DType, bf16};

/// Deterministic initialization policy recorded with a parameter declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Initializer {
    Zeros,
    Ones,
    Uniform { low: u32, high: u32 },
    Normal { mean: u32, standard_deviation: u32 },
    KaimingUniform,
}

impl Initializer {
    pub fn uniform(low: f32, high: f32) -> Self {
        Self::Uniform {
            low: low.to_bits(),
            high: high.to_bits(),
        }
    }

    pub fn normal(mean: f32, standard_deviation: f32) -> Self {
        Self::Normal {
            mean: mean.to_bits(),
            standard_deviation: standard_deviation.to_bits(),
        }
    }

    pub(crate) fn initialize(
        self,
        client: &Client,
        path: &str,
        shape: &[i64],
        dtype: DType,
        seed: u64,
    ) -> Result<Buffer> {
        let count = self.validate(path, shape, dtype)?;
        let mut random = SplitMix64(seed);
        let values = match self {
            Self::Zeros => vec![0.0; count],
            Self::Ones => vec![1.0; count],
            Self::Uniform { low, high } => {
                let low = f32::from_bits(low);
                let high = f32::from_bits(high);
                (0..count)
                    .map(|_| low + (high - low) * random.unit_f32())
                    .collect()
            }
            Self::Normal {
                mean,
                standard_deviation,
            } => {
                let mean = f32::from_bits(mean);
                let standard_deviation = f32::from_bits(standard_deviation);
                (0..count)
                    .map(|_| mean + standard_deviation * random.normal_f32())
                    .collect()
            }
            Self::KaimingUniform => {
                let fan_in = checked_fan_in(shape).expect("initializer validated above");
                let bound = (1.0 / fan_in as f32).sqrt();
                (0..count)
                    .map(|_| (random.unit_f32() * 2.0 - 1.0) * bound)
                    .collect()
            }
        };
        match dtype {
            DType::F32 => Ok(client.buffer(shape, &values)?),
            DType::BF16 => Ok(client.buffer(
                shape,
                &values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
            )?),
            DType::I32 if matches!(self, Self::Zeros | Self::Ones) => Ok(client.buffer(
                shape,
                &values
                    .into_iter()
                    .map(|value| value as i32)
                    .collect::<Vec<_>>(),
            )?),
            DType::U8 if matches!(self, Self::Zeros | Self::Ones) => Ok(client.buffer(
                shape,
                &values
                    .into_iter()
                    .map(|value| value as u8)
                    .collect::<Vec<_>>(),
            )?),
            _ => Err(Error::InvalidInitializer {
                path: path.to_owned(),
                message: format!("initialization does not support {dtype:?} storage"),
            }),
        }
    }

    pub(crate) fn validate(self, path: &str, shape: &[i64], dtype: DType) -> Result<usize> {
        let invalid = |message: &str| Error::InvalidInitializer {
            path: path.to_owned(),
            message: message.into(),
        };
        let count = shape
            .iter()
            .try_fold(1usize, |count, &dimension| {
                count.checked_mul(usize::try_from(dimension).ok()?)
            })
            .ok_or_else(|| invalid("element count overflows usize"))?;
        match self {
            Self::Uniform { low, high } => {
                let (low, high) = (f32::from_bits(low), f32::from_bits(high));
                if !low.is_finite() || !high.is_finite() || low > high {
                    return Err(invalid("uniform bounds must be finite and ordered"));
                }
            }
            Self::Normal {
                mean,
                standard_deviation,
            } => {
                let (mean, standard_deviation) =
                    (f32::from_bits(mean), f32::from_bits(standard_deviation));
                if !mean.is_finite() || !standard_deviation.is_finite() || standard_deviation < 0.0
                {
                    return Err(invalid(
                        "normal parameters must be finite with nonnegative deviation",
                    ));
                }
            }
            Self::KaimingUniform if checked_fan_in(shape).is_none() => {
                return Err(invalid("Kaiming initialization requires positive fan-in"));
            }
            _ => {}
        }
        match dtype {
            DType::F32 | DType::BF16 => {}
            DType::I32 | DType::U8 if matches!(self, Self::Zeros | Self::Ones) => {}
            _ => {
                return Err(invalid(&format!(
                    "initialization does not support {dtype:?} storage"
                )));
            }
        }
        Ok(count)
    }
}

fn checked_fan_in(shape: &[i64]) -> Option<i64> {
    let fan_in = shape
        .get(1..)?
        .iter()
        .try_fold(1i64, |value, &dimension| value.checked_mul(dimension))?;
    (fan_in > 0).then_some(fan_in)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) * (1.0 / 16_777_216.0)
    }

    fn normal_f32(&mut self) -> f32 {
        let radius = (-2.0 * self.unit_f32().ln()).sqrt();
        let angle = std::f32::consts::TAU * self.unit_f32();
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cx, Model};
    use rxla_core::{CacheLimits, Compiler, Tensor};

    #[test]
    fn validation_rejects_invalid_declarations_without_a_client() {
        assert!(matches!(
            Initializer::uniform(2.0, 1.0).validate("weight", &[2, 3], DType::F32),
            Err(Error::InvalidInitializer { path, .. }) if path == "weight"
        ));
        assert!(
            Initializer::normal(0.0, -1.0)
                .validate("weight", &[2, 3], DType::F32)
                .is_err()
        );
        assert!(
            Initializer::KaimingUniform
                .validate("weight", &[2, 0], DType::F32)
                .is_err()
        );
        assert!(
            Initializer::normal(0.0, 1.0)
                .validate("weight", &[2, 3], DType::I32)
                .is_err()
        );
        assert_eq!(
            Initializer::Zeros
                .validate("state", &[2, 3], DType::I32)
                .unwrap(),
            6
        );
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn schema_initialization_is_deterministic_and_executes() {
        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .unwrap();
        let definition =
            Model::new(|cx: &mut Cx, input: Tensor| cx.layer("head")?.linear(2).apply(&input))
                .inputs(crate::ModelInput::new([1, 3]));
        let (schema, model) = definition.trace().unwrap();
        let first = schema.initialize(&client, 42).unwrap();
        let second = schema.initialize(&client, 42).unwrap();
        assert_eq!(first.len(), 2);
        for ((first_name, first), (second_name, second)) in first.iter().zip(&second) {
            assert_eq!(first_name, second_name);
            assert_eq!(
                first.to_vec::<f32>().unwrap(),
                second.to_vec::<f32>().unwrap()
            );
        }

        let compiled = model
            .compile(&mut Compiler::new(client.clone(), CacheLimits::default()))
            .unwrap();
        let runner = compiled
            .bind_parameters(first.iter().map(|(path, buffer)| (path.as_str(), buffer)))
            .unwrap();
        let input = client.buffer(&[1, 3], &[1.0, 2.0, 3.0]).unwrap();
        let output: Buffer = runner.run(&input).unwrap();
        assert_eq!(output.dimensions().unwrap(), [1, 2]);

        let uninitialized = Model::new(|cx: &mut Cx| cx.param("external", &[1]));
        let schema = uninitialized.init().unwrap();
        assert!(matches!(
            schema.initialize(&client, 42),
            Err(Error::MissingInitializer { path }) if path == "external"
        ));

        let definition =
            Model::new(|cx: &mut Cx, input: Tensor| cx.layer("norm")?.batch_norm().apply(&input))
                .inputs(crate::ModelInput::new([1, 2, 2, 3]));
        let (schema, _, model) = definition
            .trace_resident(|schema| schema.select_all())
            .unwrap();
        assert_eq!(
            schema.get("norm.weight").unwrap().initializer(),
            Some(Initializer::Ones)
        );
        assert_eq!(
            schema
                .states()
                .iter()
                .find(|state| state.path() == "norm.running_variance")
                .unwrap()
                .initializer(),
            Initializer::Ones
        );
        let compiled = model
            .compile_stateful(&mut Compiler::new(client.clone(), CacheLimits::default()))
            .unwrap();
        let session = compiled
            .session()
            .parameters(schema.initialize(&client, 42).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let states = model.state_buffers(session.raw()).unwrap();
        let variance = states
            .iter()
            .find(|(path, _)| *path == "norm.running_variance")
            .unwrap()
            .1
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(variance, [1.0; 3]);
    }
}
