//! Functional Adam for parameter-effect models.

use rxla_core::{Buffer, Compiler, DType, PreparedStateGraph, StateProgram, Tensor};
use rxla_nn::{AppliedModel, ParameterId, ParameterSelection, TransformState};
use snafu::{Snafu, ensure};
use std::collections::BTreeMap;

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
    #[snafu(display("model Adam can only update F32 parameter {path:?}, found {dtype:?}"))]
    UnsupportedParameterDType { path: String, dtype: DType },
    #[snafu(display(
        "model Adam execution returned {actual} buffers, expected {expected} ({visible} visible and {updates} update buffers)"
    ))]
    UnexpectedOutputCount {
        actual: usize,
        expected: usize,
        visible: usize,
        updates: usize,
    },
    #[snafu(display("model Adam parameter store is missing {path:?}"))]
    MissingParameter { path: String },
    #[snafu(transparent)]
    Model { source: rxla_nn::Error },
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
}

pub type ModelAdamResult<T, E = ModelAdamError> = std::result::Result<T, E>;

/// Resident first and second moments for one parameter.
pub struct ModelAdamState {
    first_moment: TransformState,
    second_moment: TransformState,
}

impl ModelAdamState {
    pub fn first_moment(&self) -> &Tensor {
        self.first_moment.value()
    }

    pub fn second_moment(&self) -> &Tensor {
        self.second_moment.value()
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
/// Parameters remain explicit model inputs and replacements remain visible
/// outputs. Moments and the step counter are resident state committed by the
/// same state transaction as model state and RNG effects.
pub struct ModelAdamStep {
    states: Vec<ModelAdamState>,
    step: TransformState,
    updates: Vec<ModelAdamUpdate>,
}

/// Visible results and the complete Adam replacement ABI for one compiled step.
pub struct ModelAdamOutputPlan<'step> {
    step: &'step ModelAdamStep,
    tensors: Vec<Tensor>,
    visible: usize,
}

impl ModelAdamStep {
    pub fn states(&self) -> &[ModelAdamState] {
        &self.states
    }

    pub fn step_state(&self) -> &TransformState {
        &self.step
    }

    pub fn updates(&self) -> &[ModelAdamUpdate] {
        &self.updates
    }

    pub fn outputs(&self) -> Vec<Tensor> {
        self.updates
            .iter()
            .map(|update| update.parameter.clone())
            .collect()
    }

    pub fn outputs_with<'step>(&'step self, visible: &[Tensor]) -> ModelAdamOutputPlan<'step> {
        let tensors = visible.iter().cloned().chain(self.outputs()).collect();
        ModelAdamOutputPlan {
            step: self,
            tensors,
            visible: visible.len(),
        }
    }

    /// Prepare Adam replacements and model state transitions together.
    pub fn prepare_stateful(&self, model: &AppliedModel) -> ModelAdamResult<PreparedStateGraph> {
        Ok(model.prepare_stateful_tensors(&self.outputs())?)
    }

    /// Compile one Adam step while preserving resident model state and RNG
    /// transitions as hidden outputs.
    pub fn compile_stateful(
        &self,
        model: &AppliedModel,
        compiler: &mut Compiler,
    ) -> ModelAdamResult<StateProgram> {
        Ok(model.compile_stateful_tensors(compiler, &self.outputs())?)
    }
}

impl ModelAdamOutputPlan<'_> {
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    pub fn visible_count(&self) -> usize {
        self.visible
    }

    pub fn prepare_stateful(&self, model: &AppliedModel) -> ModelAdamResult<PreparedStateGraph> {
        Ok(model.prepare_stateful_tensors(self.tensors())?)
    }

    pub fn compile_stateful(
        &self,
        model: &AppliedModel,
        compiler: &mut Compiler,
    ) -> ModelAdamResult<StateProgram> {
        Ok(model.compile_stateful_tensors(compiler, self.tensors())?)
    }

    /// Validate and install every parameter replacement.
    pub fn commit(
        &self,
        mut outputs: Vec<Buffer>,
        parameters: &mut BTreeMap<String, Buffer>,
    ) -> ModelAdamResult<Vec<Buffer>> {
        let expected = self.tensors.len();
        ensure!(
            outputs.len() == expected,
            UnexpectedOutputCountSnafu {
                actual: outputs.len(),
                expected,
                visible: self.visible,
                updates: self.step.updates.len(),
            }
        );
        for update in &self.step.updates {
            ensure!(
                parameters.contains_key(update.path()),
                MissingParameterSnafu {
                    path: update.path().to_owned(),
                }
            );
        }
        let mut replacements = outputs.split_off(self.visible).into_iter();
        for update in &self.step.updates {
            let parameter = replacements.next().expect("validated Adam output ABI");
            parameters.insert(update.path().to_owned(), parameter);
        }
        Ok(outputs)
    }
}

/// Differentiate `loss` and append a bias-corrected Adam update to the model IR.
pub fn prepare_model_adam(
    model: &mut AppliedModel,
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
    let states = selection
        .parameters()
        .zip(&parameters)
        .map(|((_id, spec), parameter)| {
            Ok(ModelAdamState {
                first_moment: model.transform_state(
                    format!("__optimizer.adam.{}.first_moment", spec.path()),
                    parameter.shape(),
                    DType::F32,
                )?,
                second_moment: model.transform_state(
                    format!("__optimizer.adam.{}.second_moment", spec.path()),
                    parameter.shape(),
                    DType::F32,
                )?,
            })
        })
        .collect::<ModelAdamResult<Vec<_>>>()?;
    let step = model.transform_state("__optimizer.adam.step", &[], DType::F32)?;
    let next_step = step.value().add_scalar(1.0)?;
    let beta1_power = next_step.mul_scalar(options.beta1.ln())?.exp()?;
    let beta2_power = next_step.mul_scalar(options.beta2.ln())?.exp()?;
    let correction1 = beta1_power.mul_scalar(-1.0)?.add_scalar(1.0)?;
    let correction2 = beta2_power.mul_scalar(-1.0)?.add_scalar(1.0)?;

    let updates = selection
        .parameters()
        .zip(parameters.iter().zip(gradients).zip(&states))
        .map(|((id, spec), ((parameter, gradient), state))| {
            let first_moment = state
                .first_moment()
                .mul_scalar(options.beta1)?
                .add(&gradient.mul_scalar(1.0 - options.beta1)?)?;
            let second_moment = state
                .second_moment()
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

    let mut state_updates = Vec::with_capacity(states.len() * 2 + 1);
    for (state, update) in states.iter().zip(&updates) {
        state_updates.push((&state.first_moment, &update.first_moment));
        state_updates.push((&state.second_moment, &update.second_moment));
    }
    state_updates.push((&step, &next_step));
    model.write_transform_states(&state_updates)?;

    Ok(ModelAdamStep {
        states,
        step,
        updates,
    })
}

/// Add Adam parameter writes to an already-resident model trace.
///
/// Parameters, moments, the step counter, model state and RNG are all hidden
/// state roots committed by one session execution. Ordinary model outputs stay
/// as the complete visible result ABI.
pub fn apply_model_adam(
    model: &mut AppliedModel,
    selection: &ParameterSelection<'_>,
    loss: &Tensor,
    options: AdamOptions,
) -> ModelAdamResult<()> {
    model.validate_resident_parameters(selection)?;
    for (_, parameter) in selection.parameters() {
        ensure!(
            parameter.dtype() == DType::F32,
            UnsupportedParameterDTypeSnafu {
                path: parameter.path(),
                dtype: parameter.dtype(),
            }
        );
    }
    let step = prepare_model_adam(model, selection, loss, options)?;
    let values = step.outputs();
    model.write_resident_parameters(selection, &values)?;
    Ok(())
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
        let (schema, mut model) = Model::new(linear_loss).trace().unwrap();
        let selection = schema.select_under("linear");
        let loss = model.outputs()[0].clone();
        let step =
            prepare_model_adam(&mut model, &selection, &loss, AdamOptions::default()).unwrap();

        assert_eq!(step.states().len(), 1);
        assert_eq!(step.updates().len(), 1);
        assert_eq!(step.updates()[0].path(), "linear.weight");
        assert_eq!(step.outputs().len(), 1);
        assert_eq!(
            model.states().map(|(path, _)| path).collect::<Vec<_>>(),
            [
                "__optimizer.adam.linear.weight.first_moment",
                "__optimizer.adam.linear.weight.second_moment",
                "__optimizer.adam.step",
            ]
        );
        let prepared = step.prepare_stateful(&model).unwrap();
        assert_eq!(
            prepared.input_indices().len(),
            model.schema().arguments().len()
        );
        let slots = model
            .states()
            .map(|(_, slot)| slot.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            prepared.state_layout(&slots).unwrap(),
            vec![
                (DType::F32, vec![1, 3]),
                (DType::F32, vec![1, 3]),
                (DType::F32, vec![]),
            ]
        );

        let outputs = step.outputs_with(model.outputs());
        assert_eq!(outputs.visible_count(), 1);
        assert_eq!(outputs.tensors().len(), 2);
        let error = match outputs.commit(Vec::new(), &mut BTreeMap::new()) {
            Ok(_) => panic!("truncated Adam outputs unexpectedly committed"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            ModelAdamError::UnexpectedOutputCount {
                actual: 0,
                expected: 2,
                visible: 1,
                updates: 1,
            }
        ));
    }

    #[test]
    fn adam_rejects_invalid_options() {
        let (schema, mut model) = Model::new(linear_loss).trace().unwrap();
        let options = AdamOptions {
            beta2: 1.0,
            ..AdamOptions::default()
        };
        let loss = model.outputs()[0].clone();
        assert!(matches!(
            prepare_model_adam(&mut model, &schema.select_all(), &loss, options),
            Err(ModelAdamError::InvalidOptions)
        ));
    }

    #[test]
    fn resident_adam_hides_parameter_replacements_from_the_result_abi() {
        let definition = Model::new(linear_loss);
        let schema = definition.init().unwrap();
        let selection = schema.select_under("linear");
        let mut model = definition.apply_resident(&schema, &selection).unwrap();
        let loss = model.outputs()[0].clone();

        apply_model_adam(&mut model, &selection, &loss, AdamOptions::default()).unwrap();

        let prepared = model.prepare_stateful().unwrap();
        assert!(prepared.output_spec(0).is_some());
        assert!(prepared.output_spec(1).is_none());
        assert_eq!(prepared.input_indices(), [0]);
        assert_eq!(model.resident_parameters().count(), 1);
        assert_eq!(model.states().count(), 3);
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
        let (schema, mut model) = Model::new(regression_loss).trace().unwrap();
        let selection = schema.select_under("linear");
        let loss = model.outputs()[0].clone();
        let step = prepare_model_adam(
            &mut model,
            &selection,
            &loss,
            AdamOptions {
                learning_rate: 0.1,
                ..AdamOptions::default()
            },
        )
        .unwrap();
        let output_plan = step.outputs_with(&[]);
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let program = output_plan.compile_stateful(&model, &mut compiler).unwrap();
        let mut session = model.session(&program).build().unwrap();
        let input = client.buffer(&[1, 1], &[2.0]).unwrap();
        let target = client.buffer(&[1, 1], &[4.0]).unwrap();
        let mut parameters = BTreeMap::from([(
            "linear.weight".to_owned(),
            client.buffer(&[1, 1], &[0.0]).unwrap(),
        )]);

        for _ in 0..100 {
            let model_arguments = model
                .bind(
                    [&input, &target],
                    parameters
                        .iter()
                        .map(|(name, value)| (name.as_str(), value)),
                )
                .unwrap();
            let outputs = session.run(model_arguments.as_slice()).unwrap();
            assert!(
                output_plan
                    .commit(outputs, &mut parameters)
                    .unwrap()
                    .is_empty()
            );
        }

        let actual = parameters["linear.weight"].to_vec::<f32>().unwrap()[0];
        assert!((actual - 2.0).abs() < 0.02, "weight = {actual}");
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn resident_adam_commits_parameters_and_optimizer_state_in_session() {
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
        let definition = Model::new(regression_loss);
        let schema = definition.init().unwrap();
        let selection = schema.select_under("linear");
        let mut model = definition.apply_resident(&schema, &selection).unwrap();
        let loss = model.outputs()[0].clone();
        apply_model_adam(
            &mut model,
            &selection,
            &loss,
            AdamOptions {
                learning_rate: 0.1,
                ..AdamOptions::default()
            },
        )
        .unwrap();
        let step_slot = model
            .states()
            .find(|(path, _)| *path == "__optimizer.adam.step")
            .unwrap()
            .1
            .clone();

        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let program = model.compile_stateful(&mut compiler).unwrap();
        let mut session = model
            .session(&program)
            .parameter("linear.weight", client.buffer(&[1, 1], &[0.0]).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let input = client.buffer(&[1, 1], &[2.0]).unwrap();
        let target = client.buffer(&[1, 1], &[4.0]).unwrap();

        for _ in 0..100 {
            let outputs = session.run(&[&input, &target]).unwrap();
            assert_eq!(outputs.len(), 1);
        }

        let (_, _, parameter_slot) = model.resident_parameters().next().unwrap();
        let actual = session
            .state(parameter_slot)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!((actual - 2.0).abs() < 0.02, "weight = {actual}");
        assert_eq!(
            session.state(&step_slot).unwrap().to_vec::<f32>().unwrap(),
            [100.0]
        );
    }
}
