//! Functional Adam for parameter-effect models.

use rxla_core::{Compiler, DType, Executable, LoweredProgram, Tensor};
use rxla_nn::{AppliedModel, ParameterId, ParameterSelection};
use snafu::{Snafu, ensure};
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
pub struct AdamOptions {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for AdamOptions {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum ModelAdamError {
    #[snafu(display("model Adam requires at least one selected parameter"))]
    EmptySelection,
    #[snafu(display("model Adam loss must be a scalar F32 tensor"))]
    InvalidLoss,
    #[snafu(display("invalid Adam hyperparameters"))]
    InvalidOptions,
    #[snafu(transparent)]
    Model { source: rxla_nn::Error },
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
}

pub type ModelAdamResult<T, E = ModelAdamError> = std::result::Result<T, E>;

/// Graph inputs holding one parameter's first and second moments.
pub struct ModelAdamState {
    first_moment: Tensor,
    second_moment: Tensor,
}

impl ModelAdamState {
    pub fn first_moment(&self) -> &Tensor {
        &self.first_moment
    }

    pub fn second_moment(&self) -> &Tensor {
        &self.second_moment
    }
}

/// Replacement parameter and optimizer state for one selected identity.
pub struct ModelAdamUpdate {
    id: ParameterId,
    path: String,
    parameter: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
}

impl ModelAdamUpdate {
    pub fn id(&self) -> ParameterId {
        self.id
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn parameter(&self) -> &Tensor {
        &self.parameter
    }

    pub fn first_moment(&self) -> &Tensor {
        &self.first_moment
    }

    pub fn second_moment(&self) -> &Tensor {
        &self.second_moment
    }
}

/// One fused forward/backward/Adam program.
///
/// Runtime arguments are the ordinary model arguments followed by each state's
/// `(first_moment, second_moment)` pair and finally a scalar F32 step number.
/// Outputs repeat `(parameter, first_moment, second_moment)` per selection entry.
pub struct ModelAdamStep {
    states: Vec<ModelAdamState>,
    step: Tensor,
    updates: Vec<ModelAdamUpdate>,
}

impl ModelAdamStep {
    pub fn states(&self) -> &[ModelAdamState] {
        &self.states
    }

    pub fn step_input(&self) -> &Tensor {
        &self.step
    }

    pub fn updates(&self) -> &[ModelAdamUpdate] {
        &self.updates
    }

    pub fn outputs(&self) -> Vec<Tensor> {
        self.updates
            .iter()
            .flat_map(|update| {
                [
                    update.parameter.clone(),
                    update.first_moment.clone(),
                    update.second_moment.clone(),
                ]
            })
            .collect()
    }

    pub fn prepare(&self, model: &AppliedModel) -> ModelAdamResult<LoweredProgram> {
        Ok(model.prepare_tensors(&self.outputs())?)
    }

    pub fn compile(
        &self,
        model: &AppliedModel,
        compiler: &mut Compiler,
    ) -> ModelAdamResult<Rc<Executable>> {
        Ok(model.compile_tensors(compiler, &self.outputs())?)
    }
}

/// Differentiate `loss` and append a bias-corrected Adam update to the model IR.
pub fn prepare_model_adam(
    model: &AppliedModel,
    selection: &ParameterSelection<'_>,
    loss: &Tensor,
    options: AdamOptions,
) -> ModelAdamResult<ModelAdamStep> {
    ensure!(!selection.is_empty(), EmptySelectionSnafu);
    ensure!(
        loss.shape().is_empty() && loss.dtype() == DType::F32,
        InvalidLossSnafu
    );
    ensure!(
        options.learning_rate.is_finite()
            && options.learning_rate >= 0.0
            && options.beta1.is_finite()
            && (0.0..1.0).contains(&options.beta1)
            && options.beta2.is_finite()
            && (0.0..1.0).contains(&options.beta2)
            && options.epsilon.is_finite()
            && options.epsilon > 0.0,
        InvalidOptionsSnafu
    );

    let parameters = model.parameter_tensors(selection)?;
    let gradients = loss.grad(&parameters)?;
    let states = parameters
        .iter()
        .map(|parameter| {
            Ok(ModelAdamState {
                first_moment: model.transform_input(parameter.shape(), DType::F32)?,
                second_moment: model.transform_input(parameter.shape(), DType::F32)?,
            })
        })
        .collect::<ModelAdamResult<Vec<_>>>()?;
    let step = model.transform_input(&[], DType::F32)?;
    let beta1_power = step.mul_scalar(options.beta1.ln())?.exp()?;
    let beta2_power = step.mul_scalar(options.beta2.ln())?.exp()?;
    let correction1 = beta1_power.mul_scalar(-1.0)?.add_scalar(1.0)?;
    let correction2 = beta2_power.mul_scalar(-1.0)?.add_scalar(1.0)?;

    let updates = selection
        .parameters()
        .zip(parameters.iter().zip(gradients).zip(&states))
        .map(|((id, spec), ((parameter, gradient), state))| {
            let first_moment = state
                .first_moment
                .mul_scalar(options.beta1)?
                .add(&gradient.mul_scalar(1.0 - options.beta1)?)?;
            let second_moment = state
                .second_moment
                .mul_scalar(options.beta2)?
                .add(&gradient.square()?.mul_scalar(1.0 - options.beta2)?)?;
            let corrected_first =
                first_moment.div(&correction1.broadcast_to(parameter.shape())?)?;
            let corrected_second =
                second_moment.div(&correction2.broadcast_to(parameter.shape())?)?;
            let direction =
                corrected_first.div(&corrected_second.sqrt()?.add_scalar(options.epsilon)?)?;
            let parameter = parameter.sub(&direction.mul_scalar(options.learning_rate)?)?;
            Ok(ModelAdamUpdate {
                id,
                path: spec.path().to_owned(),
                parameter,
                first_moment,
                second_moment,
            })
        })
        .collect::<ModelAdamResult<Vec<_>>>()?;

    Ok(ModelAdamStep {
        states,
        step,
        updates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{CacheLimits, Client, Compiler};
    use rxla_nn::{Cx, Model, Result};

    fn linear_loss(cx: &mut Cx) -> Result<Tensor> {
        let input = cx.input(&[2, 3])?;
        let prediction = cx.named("linear")?.linear(1).bias(false).apply(&input)?;
        Ok(prediction.square()?.sum(&[0, 1], false)?)
    }

    #[test]
    fn adam_lowers_parameter_and_moment_updates_together() {
        let (schema, model) = Model::new(linear_loss).trace().unwrap();
        let selection = schema.select_under("linear");
        let step = prepare_model_adam(
            &model,
            &selection,
            &model.outputs()[0],
            AdamOptions::default(),
        )
        .unwrap();

        assert_eq!(step.states().len(), 1);
        assert_eq!(step.updates().len(), 1);
        assert_eq!(step.updates()[0].path(), "linear.weight");
        assert_eq!(step.outputs().len(), 3);
        let lowered = step.prepare(&model).unwrap();
        assert_eq!(lowered.input_count(), model.schema().arguments().len() + 3);
        assert_eq!(lowered.output_count(), 3);
        let stablehlo = std::str::from_utf8(lowered.code()).unwrap();
        assert!(stablehlo.contains("stablehlo.sqrt"));
        assert!(stablehlo.contains("stablehlo.exponential"));
    }

    #[test]
    fn adam_rejects_invalid_options() {
        let (schema, model) = Model::new(linear_loss).trace().unwrap();
        let options = AdamOptions {
            beta2: 1.0,
            ..AdamOptions::default()
        };
        assert!(matches!(
            prepare_model_adam(&model, &schema.select_all(), &model.outputs()[0], options),
            Err(ModelAdamError::InvalidOptions)
        ));
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn adam_executes_as_one_device_program() {
        fn regression_loss(cx: &mut Cx) -> Result<Tensor> {
            let input = cx.input(&[1, 1])?;
            let target = cx.input(&[1, 1])?;
            let prediction = cx.named("linear")?.linear(1).bias(false).apply(&input)?;
            Ok(prediction.sub(&target)?.square()?.sum(&[0, 1], false)?)
        }

        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .unwrap();
        let (schema, model) = Model::new(regression_loss).trace().unwrap();
        let selection = schema.select_under("linear");
        let step = prepare_model_adam(
            &model,
            &selection,
            &model.outputs()[0],
            AdamOptions {
                learning_rate: 0.1,
                ..AdamOptions::default()
            },
        )
        .unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let executable = step.compile(&model, &mut compiler).unwrap();
        let input = client.buffer(&[1, 1], &[2.0]).unwrap();
        let target = client.buffer(&[1, 1], &[4.0]).unwrap();
        let mut weight = client.buffer(&[1, 1], &[0.0]).unwrap();
        let mut first = client.buffer(&[1, 1], &[0.0]).unwrap();
        let mut second = client.buffer(&[1, 1], &[0.0]).unwrap();

        for iteration in 1..=100 {
            let iteration = client.buffer(&[], &[iteration as f32]).unwrap();
            let model_arguments = model
                .bind(&[&input, &target], [("linear.weight", &weight)])
                .unwrap();
            let mut arguments = model_arguments.as_slice().to_vec();
            arguments.extend([&first, &second, &iteration]);
            let mut outputs = executable.execute(&arguments).unwrap().into_iter();
            weight = outputs.next().unwrap();
            first = outputs.next().unwrap();
            second = outputs.next().unwrap();
        }

        let actual = weight.to_vec::<f32>().unwrap()[0];
        assert!((actual - 2.0).abs() < 0.02, "weight = {actual}");
    }
}
