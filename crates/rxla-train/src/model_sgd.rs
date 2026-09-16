//! Minimal functional SGD for effect-based models.

use rxla_core::{
    Buffer, Compiler, DType, Executable, LoweredProgram, PreparedStateGraph, StateProgram, Tensor,
};
use rxla_nn::{AppliedModel, ParameterId, ParameterSelection};
use snafu::{Snafu, ensure};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum ModelSgdError {
    #[snafu(display("model SGD requires at least one selected parameter"))]
    EmptySelection,
    #[snafu(display("model SGD loss must be a scalar F32 tensor"))]
    InvalidLoss,
    #[snafu(display("model SGD learning rate must be finite and nonnegative"))]
    InvalidLearningRate,
    #[snafu(display("model SGD can only update F32 parameter {path:?}, found {dtype:?}"))]
    UnsupportedParameterDType { path: String, dtype: DType },
    #[snafu(display(
        "model SGD execution returned {actual} buffers, expected {expected} ({visible} visible and {updates} updates)"
    ))]
    UnexpectedOutputCount {
        actual: usize,
        expected: usize,
        visible: usize,
        updates: usize,
    },
    #[snafu(display("model SGD parameter store is missing {path:?}"))]
    MissingParameter { path: String },
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

/// One explicit visible-result layout followed by this SGD step's replacements.
///
/// The same value supplies compilation roots and decodes execution results, so
/// the visible/update boundary cannot diverge between those two operations.
pub struct ModelSgdOutputPlan<'step> {
    step: &'step ModelSgdStep,
    tensors: Vec<Tensor>,
    visible: usize,
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

    /// Compose caller-visible results with the optimizer replacement ABI.
    ///
    /// The visible prefix is returned unchanged by
    /// [`ModelSgdOutputPlan::commit`]; callers never need to calculate where
    /// parameter replacements begin.
    pub fn outputs_with<'step>(&'step self, visible: &[Tensor]) -> ModelSgdOutputPlan<'step> {
        let tensors = visible
            .iter()
            .cloned()
            .chain(self.updates.iter().map(|update| update.value.clone()))
            .collect();
        ModelSgdOutputPlan {
            step: self,
            tensors,
            visible: visible.len(),
        }
    }
}

impl ModelSgdOutputPlan<'_> {
    /// Ordered compilation roots: visible results, then parameter replacements.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    pub fn visible_count(&self) -> usize {
        self.visible
    }

    pub fn prepare(&self, model: &AppliedModel) -> ModelSgdResult<LoweredProgram> {
        Ok(model.prepare_tensors(self.tensors())?)
    }

    pub fn compile(
        &self,
        model: &AppliedModel,
        compiler: &mut Compiler,
    ) -> ModelSgdResult<Arc<Executable>> {
        Ok(model.compile_tensors(compiler, self.tensors())?)
    }

    /// Prepare visible results, replacements, and model state transitions.
    pub fn prepare_stateful(&self, model: &AppliedModel) -> ModelSgdResult<PreparedStateGraph> {
        Ok(model.prepare_stateful_tensors(self.tensors())?)
    }

    /// Compile visible results, replacements, and model state transitions.
    pub fn compile_stateful(
        &self,
        model: &AppliedModel,
        compiler: &mut Compiler,
    ) -> ModelSgdResult<StateProgram> {
        Ok(model.compile_stateful_tensors(compiler, self.tensors())?)
    }

    /// Validate and atomically install parameter replacements from one run.
    ///
    /// No parameter is changed unless the complete output ABI is present and
    /// every selected path already exists in `parameters`. The returned buffers
    /// are exactly the caller-visible prefix supplied to
    /// [`ModelSgdStep::outputs_with`].
    pub fn commit(
        &self,
        mut outputs: Vec<Buffer>,
        parameters: &mut BTreeMap<String, Buffer>,
    ) -> ModelSgdResult<Vec<Buffer>> {
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
        let replacements = outputs.split_off(self.visible);
        for (update, value) in self.step.updates.iter().zip(replacements) {
            parameters.insert(update.path().to_owned(), value);
        }
        Ok(outputs)
    }
}

impl ModelSgdStep {
    pub fn prepare(&self, model: &AppliedModel) -> ModelSgdResult<LoweredProgram> {
        Ok(model.prepare_tensors(&self.outputs())?)
    }

    pub fn compile(
        &self,
        model: &AppliedModel,
        compiler: &mut Compiler,
    ) -> ModelSgdResult<Arc<Executable>> {
        Ok(model.compile_tensors(compiler, &self.outputs())?)
    }

    /// Prepare parameter replacements and model state transitions together.
    pub fn prepare_stateful(&self, model: &AppliedModel) -> ModelSgdResult<PreparedStateGraph> {
        Ok(model.prepare_stateful_tensors(&self.outputs())?)
    }

    /// Compile one step whose visible outputs are replacement parameters and
    /// whose hidden outputs commit model state and RNG transitions.
    pub fn compile_stateful(
        &self,
        model: &AppliedModel,
        compiler: &mut Compiler,
    ) -> ModelSgdResult<StateProgram> {
        Ok(model.compile_stateful_tensors(compiler, &self.outputs())?)
    }
}

/// Differentiate a scalar loss with respect to `selection` and build plain SGD
/// replacements `parameter - learning_rate * gradient`.
pub fn prepare_model_sgd(
    model: &AppliedModel,
    selection: &ParameterSelection,
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

/// Add SGD writes to parameters already interpreted as resident model state.
///
/// The updates become hidden state roots. Compiling the mutated model with its
/// ordinary outputs therefore produces one transaction containing forward,
/// autodiff, parameter updates and any model state/RNG updates. No replacement
/// parameter buffers are exposed in the execution result.
pub fn apply_model_sgd(
    model: &mut AppliedModel,
    selection: &ParameterSelection,
    loss: &Tensor,
    learning_rate: f32,
) -> ModelSgdResult<()> {
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
    let step = prepare_model_sgd(model, selection, loss, learning_rate)?;
    let values = step.outputs();
    model.write_resident_parameters(selection, &values)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{CacheLimits, Client, DType};
    use rxla_nn::{Cx, Model, Result};

    fn linear_loss(cx: &mut Cx) -> Result<Tensor> {
        let input = cx.input(&[2, 3])?;
        let prediction = cx.layer("linear")?.linear(1).bias(false).apply(&input)?;
        Ok(prediction.mul(&prediction)?.sum(&[0, 1], false)?)
    }

    #[test]
    fn linear_sgd_selects_differentiates_and_lowers_one_weight() {
        let model = Model::new(linear_loss).trace().unwrap();
        let schema = model.schema().clone();
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
        let model = Model::new(linear_loss).trace().unwrap();
        let schema = model.schema().clone();
        let empty = schema.select_under("missing");
        assert!(matches!(
            prepare_model_sgd(&model, &empty, &model.outputs()[0], 0.1),
            Err(ModelSgdError::EmptySelection)
        ));
        assert!(matches!(
            prepare_model_sgd(&model, &schema.select_all(), &model.outputs()[0], f32::NAN),
            Err(ModelSgdError::InvalidLearningRate)
        ));

        let step =
            prepare_model_sgd(&model, &schema.select_all(), &model.outputs()[0], 0.1).unwrap();
        let outputs = step.outputs_with(model.outputs());
        assert_eq!(
            outputs.tensors().len(),
            model.outputs().len() + step.updates().len()
        );
        assert_eq!(outputs.visible_count(), 1);
        let error = match outputs.commit(Vec::new(), &mut BTreeMap::new()) {
            Ok(_) => panic!("truncated SGD outputs unexpectedly committed"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            ModelSgdError::UnexpectedOutputCount {
                actual: 0,
                expected: 2,
                visible: 1,
                updates: 1,
            }
        ));
    }

    #[test]
    fn stateful_sgd_retains_state_writes_as_hidden_roots() {
        fn stateful_loss(cx: &mut Cx) -> Result<Tensor> {
            let input = cx.input(&[1, 1])?;
            let steps = cx.state("steps", &[], DType::F32)?;
            steps.add_(cx, &cx.constant(&[], &[1.0])?)?;
            let prediction = cx.layer("linear")?.linear(1).bias(false).apply(&input)?;
            Ok(prediction.mul(&prediction)?.sum(&[0, 1], false)?)
        }

        let model = Model::new(stateful_loss).trace().unwrap();
        let schema = model.schema().clone();
        let step =
            prepare_model_sgd(&model, &schema.select_all(), &model.outputs()[0], 0.1).unwrap();

        assert!(step.prepare(&model).is_err());
        let prepared = step.prepare_stateful(&model).unwrap();
        assert_eq!(prepared.output_spec(0).unwrap().shape, [1, 1]);
        let slot = model.states().next().unwrap().1;
        assert_eq!(prepared.state_type(slot).unwrap(), (DType::F32, vec![]));
    }

    #[test]
    fn resident_sgd_hides_parameter_replacements_from_the_result_abi() {
        let definition = Model::new(linear_loss);
        let (selection, mut model) = definition.trace_resident_under("linear").unwrap();
        let loss = model.outputs()[0].clone();

        apply_model_sgd(&mut model, &selection, &loss, 0.1).unwrap();

        let prepared = model.prepare_stateful().unwrap();
        assert!(prepared.output_spec(0).is_some());
        assert!(prepared.output_spec(1).is_none());
        assert_eq!(prepared.input_indices(), [0]);
        assert_eq!(model.resident_parameters().count(), 1);
    }

    #[test]
    fn resident_sgd_rejects_an_input_backed_selection() {
        let mut model = Model::new(linear_loss).trace().unwrap();
        let schema = model.schema().clone();
        let selection = schema.select_all();
        let loss = model.outputs()[0].clone();
        assert!(matches!(
            apply_model_sgd(&mut model, &selection, &loss, 0.1),
            Err(ModelSgdError::Model {
                source: rxla_nn::Error::ParameterNotResident { .. }
            })
        ));
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn stateful_sgd_executes_update_and_state_transition_together() {
        fn stateful_loss(cx: &mut Cx) -> Result<Tensor> {
            let input = cx.input(&[1, 1])?;
            let steps = cx.state("steps", &[], DType::F32)?;
            steps.add_(cx, &cx.constant(&[], &[1.0])?)?;
            let prediction = cx.layer("linear")?.linear(1).bias(false).apply(&input)?;
            Ok(prediction.mul(&prediction)?.sum(&[0, 1], false)?)
        }

        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .expect("load CPU plugin");
        let model = Model::new(stateful_loss).trace().unwrap();
        let schema = model.schema().clone();
        let step =
            prepare_model_sgd(&model, &schema.select_all(), &model.outputs()[0], 0.1).unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let program = step
            .compile_stateful(&model, &mut compiler)
            .expect("compile stateful SGD");
        let mut session = model.session(&program).build().unwrap();
        let input = client.buffer(&[1, 1], &[2.0]).unwrap();
        let weight = client.buffer(&[1, 1], &[1.0]).unwrap();

        let outputs = session.run(&[&input, &weight]).unwrap();
        let updated_weight = outputs[0].to_vec::<f32>().unwrap()[0];
        assert!((updated_weight - 0.2).abs() < 1e-6, "{updated_weight}");
        let slot = model.states().next().unwrap().1;
        assert_eq!(
            session.state(slot).unwrap().to_vec::<f32>().unwrap(),
            vec![1.0]
        );
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn linear_sgd_executes_as_one_device_program() {
        fn regression_loss(cx: &mut Cx) -> Result<Tensor> {
            let input = cx.input(&[1, 1])?;
            let target = cx.input(&[1, 1])?;
            let prediction = cx.layer("linear")?.linear(1).bias(false).apply(&input)?;
            let error = prediction.sub(&target)?;
            Ok(error.mul(&error)?.sum(&[0, 1], false)?)
        }

        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .expect("load CPU plugin");
        let model = Model::new(regression_loss).trace().unwrap();
        let schema = model.schema().clone();
        let selection = schema.select_under("linear");
        let step = prepare_model_sgd(&model, &selection, &model.outputs()[0], 0.1).unwrap();
        let output_plan = step.outputs_with(&[]);
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let loss_model = model.compile(&mut compiler).unwrap();
        let step_executable = output_plan.compile(&model, &mut compiler).unwrap();
        let input = client.buffer(&[1, 1], &[2.0]).unwrap();
        let target = client.buffer(&[1, 1], &[4.0]).unwrap();
        let mut parameters = BTreeMap::from([(
            "linear.weight".to_owned(),
            client.buffer(&[1, 1], &[0.0]).unwrap(),
        )]);

        let initial_loss: Buffer = loss_model
            .run(
                [&input, &target],
                parameters
                    .iter()
                    .map(|(name, value)| (name.as_str(), value)),
            )
            .unwrap();
        let initial_loss = initial_loss.to_vec::<f32>().unwrap()[0];

        for _ in 0..8 {
            let arguments = model
                .bind(
                    [&input, &target],
                    parameters
                        .iter()
                        .map(|(name, value)| (name.as_str(), value)),
                )
                .unwrap();
            let outputs = step_executable.execute(arguments.as_slice()).unwrap();
            assert!(
                output_plan
                    .commit(outputs, &mut parameters)
                    .unwrap()
                    .is_empty()
            );
        }

        let final_loss: Buffer = loss_model
            .run(
                [&input, &target],
                parameters
                    .iter()
                    .map(|(name, value)| (name.as_str(), value)),
            )
            .unwrap();
        let final_loss = final_loss.to_vec::<f32>().unwrap()[0];
        assert!(
            final_loss < initial_loss * 1e-6,
            "{initial_loss} -> {final_loss}"
        );
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn resident_linear_sgd_initializes_once_and_commits_in_session() {
        fn regression_loss(cx: &mut Cx) -> Result<Tensor> {
            let input = cx.input(&[1, 1])?;
            let prediction = cx.layer("linear")?.linear(1).bias(false).apply(&input)?;
            Ok(prediction.mul(&prediction)?.sum(&[0, 1], false)?)
        }

        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .expect("load CPU plugin");
        let definition = Model::new(regression_loss);
        let (selection, mut model) = definition.trace_resident_under("linear").unwrap();
        let schema = model.schema().clone();
        let slot = model.resident_parameters().next().unwrap().2.clone();
        assert_eq!(
            model.prepare_stateful().unwrap().state_type(&slot).unwrap(),
            (DType::F32, vec![1, 1])
        );
        let loss = model.outputs()[0].clone();
        apply_model_sgd(&mut model, &selection, &loss, 0.1).unwrap();
        assert_eq!(
            model.prepare_stateful().unwrap().state_type(&slot).unwrap(),
            (DType::F32, vec![1, 1])
        );

        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let program = model.compile_stateful(&mut compiler).unwrap();
        let weight = client.buffer(&[1, 1], &[1.0]).unwrap();
        let (_, _, weight_slot) = model.resident_parameters().next().unwrap();
        assert_eq!(
            program.state_type(weight_slot).unwrap(),
            (weight.dtype().unwrap(), weight.dimensions().unwrap())
        );
        let mut session = program
            .session()
            .parameter("linear.weight", weight)
            .unwrap()
            .build()
            .unwrap();
        let input = client.buffer(&[1, 1], &[2.0]).unwrap();

        let first = session
            .run::<_, Buffer>(&input)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[0];
        let mut last = first;
        for _ in 0..20 {
            last = session
                .run::<_, Buffer>(&input)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()[0];
        }
        assert!(last < first * 1e-6, "{first} -> {last}");
        let (_, _, slot) = model.resident_parameters().next().unwrap();
        let trained = session.raw().state(slot).unwrap().to_vec::<f32>().unwrap();
        assert!(
            trained.iter().all(|value| value.abs() < 1e-4),
            "{trained:?}"
        );

        // A separately traced inference model can take ownership of the
        // trained device buffer by canonical parameter path. No host download,
        // re-upload, or optimizer-shaped public ABI is involved.
        let inference = definition.apply_resident(&schema, &selection).unwrap();
        let inference_program = inference.compile_stateful(&mut compiler).unwrap();
        let snapshot = model.take_session(session.into_raw()).unwrap();
        let mut inference_session = snapshot
            .restore_model(inference_program.session().into_raw_builder())
            .unwrap()
            .build()
            .unwrap();
        let inference_loss = inference_session.run(&[&input]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(inference_loss < first * 1e-6, "{inference_loss}");
    }
}
