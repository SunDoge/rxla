//! Minimal functional SGD for effect-based models.

use rxla_core::{Compiler, Executable, LoweredProgram, Tensor};
use rxla_nn::{AppliedModel, ParameterId, ParameterSelection};
use snafu::{Snafu, ensure};
use std::rc::Rc;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum ModelSgdError {
    #[snafu(display("model SGD requires at least one selected parameter"))]
    EmptySelection,
    #[snafu(display("model SGD loss must be a scalar F32 tensor"))]
    InvalidLoss,
    #[snafu(display("model SGD learning rate must be finite and nonnegative"))]
    InvalidLearningRate,
    #[snafu(transparent)]
    Model { source: rxla_nn::Error },
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
}

pub type ModelSgdResult<T, E = ModelSgdError> = std::result::Result<T, E>;

/// One selected parameter and its functional SGD replacement value.
pub struct ModelSgdUpdate {
    id: ParameterId,
    path: String,
    value: Tensor,
}

impl ModelSgdUpdate {
    pub fn id(&self) -> ParameterId {
        self.id
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn value(&self) -> &Tensor {
        &self.value
    }
}

/// A stateless SGD transform whose outputs are replacement parameter values.
///
/// Execute the compiled step with the original model ABI. Its output buffers
/// correspond to [`Self::updates`] and can replace those named bindings for the
/// next step.
pub struct ModelSgdStep {
    updates: Vec<ModelSgdUpdate>,
}

impl ModelSgdStep {
    pub fn updates(&self) -> &[ModelSgdUpdate] {
        &self.updates
    }

    pub fn outputs(&self) -> Vec<Tensor> {
        self.updates
            .iter()
            .map(|update| update.value.clone())
            .collect()
    }

    pub fn prepare(&self, model: &AppliedModel) -> ModelSgdResult<LoweredProgram> {
        Ok(model.prepare_tensors(&self.outputs())?)
    }

    pub fn compile(
        &self,
        model: &AppliedModel,
        compiler: &mut Compiler,
    ) -> ModelSgdResult<Rc<Executable>> {
        Ok(model.compile_tensors(compiler, &self.outputs())?)
    }
}

/// Differentiate a scalar loss with respect to `selection` and build plain SGD
/// replacements `parameter - learning_rate * gradient`.
pub fn prepare_model_sgd(
    model: &AppliedModel,
    selection: &ParameterSelection<'_>,
    loss: &Tensor,
    learning_rate: f32,
) -> ModelSgdResult<ModelSgdStep> {
    ensure!(!selection.is_empty(), EmptySelectionSnafu);
    ensure!(
        loss.shape().is_empty() && loss.dtype() == rxla_core::DType::F32,
        InvalidLossSnafu
    );
    ensure!(
        learning_rate.is_finite() && learning_rate >= 0.0,
        InvalidLearningRateSnafu
    );

    let parameters = model.parameter_tensors(selection)?;
    let gradients = loss.grad(&parameters)?;
    let updates = selection
        .parameters()
        .zip(parameters.iter().zip(gradients))
        .map(|((id, spec), (parameter, gradient))| {
            let value = parameter.sub(&gradient.mul_scalar(learning_rate)?)?;
            Ok(ModelSgdUpdate {
                id,
                path: spec.path().to_owned(),
                value,
            })
        })
        .collect::<ModelSgdResult<Vec<_>>>()?;
    Ok(ModelSgdStep { updates })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{CacheLimits, Client};
    use rxla_nn::{Cx, Model, Result};

    fn linear_loss(cx: &mut Cx) -> Result<Tensor> {
        let input = cx.input(&[2, 3])?;
        let prediction = cx.named("linear")?.linear(1).bias(false).apply(&input)?;
        Ok(prediction.mul(&prediction)?.sum(&[0, 1], false)?)
    }

    #[test]
    fn linear_sgd_selects_differentiates_and_lowers_one_weight() {
        let (schema, model) = Model::new(linear_loss).trace().unwrap();
        let selection = schema.select_under("linear");
        let step = prepare_model_sgd(&model, &selection, &model.outputs()[0], 0.1).unwrap();

        assert_eq!(step.updates().len(), 1);
        assert_eq!(step.updates()[0].path(), "linear.weight");
        assert_eq!(step.updates()[0].value().shape(), [1, 3]);
        let lowered = step.prepare(&model).unwrap();
        assert_eq!(lowered.output_count(), 1);
        let stablehlo = std::str::from_utf8(lowered.code()).unwrap();
        assert!(stablehlo.contains("stablehlo.dot_general"));
        assert!(stablehlo.contains("stablehlo.subtract"));
    }

    #[test]
    fn linear_sgd_rejects_empty_selection_and_bad_rate() {
        let (schema, model) = Model::new(linear_loss).trace().unwrap();
        let empty = schema.select_under("missing");
        assert!(matches!(
            prepare_model_sgd(&model, &empty, &model.outputs()[0], 0.1),
            Err(ModelSgdError::EmptySelection)
        ));
        assert!(matches!(
            prepare_model_sgd(&model, &schema.select_all(), &model.outputs()[0], f32::NAN),
            Err(ModelSgdError::InvalidLearningRate)
        ));
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn linear_sgd_executes_as_one_device_program() {
        fn regression_loss(cx: &mut Cx) -> Result<Tensor> {
            let input = cx.input(&[1, 1])?;
            let target = cx.input(&[1, 1])?;
            let prediction = cx.named("linear")?.linear(1).bias(false).apply(&input)?;
            let error = prediction.sub(&target)?;
            Ok(error.mul(&error)?.sum(&[0, 1], false)?)
        }

        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .expect("load CPU plugin");
        let (schema, model) = Model::new(regression_loss).trace().unwrap();
        let selection = schema.select_under("linear");
        let step = prepare_model_sgd(&model, &selection, &model.outputs()[0], 0.1).unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let loss_executable = model.compile(&mut compiler).unwrap();
        let step_executable = step.compile(&model, &mut compiler).unwrap();
        let input = client.buffer(&[1, 1], &[2.0]).unwrap();
        let target = client.buffer(&[1, 1], &[4.0]).unwrap();
        let mut weight = client.buffer(&[1, 1], &[0.0]).unwrap();

        let initial_arguments = model
            .bind(&[&input, &target], [("linear.weight", &weight)])
            .unwrap();
        let initial_loss = loss_executable
            .execute(initial_arguments.as_slice())
            .unwrap()[0]
            .to_vec::<f32>()
            .unwrap()[0];

        for _ in 0..8 {
            let arguments = model
                .bind(&[&input, &target], [("linear.weight", &weight)])
                .unwrap();
            weight = step_executable
                .execute(arguments.as_slice())
                .unwrap()
                .remove(0);
        }

        let final_arguments = model
            .bind(&[&input, &target], [("linear.weight", &weight)])
            .unwrap();
        let final_loss = loss_executable.execute(final_arguments.as_slice()).unwrap()[0]
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            final_loss < initial_loss * 1e-6,
            "{initial_loss} -> {final_loss}"
        );
    }
}
