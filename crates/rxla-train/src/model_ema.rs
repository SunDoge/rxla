//! Device-resident exponential moving averages for effect-based models.

use rxla_core::{Buffer, DType, Session, Tensor};
use rxla_nn::{AppliedModel, ParameterSelection, TransformState};
use snafu::{Snafu, ensure};

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum ModelEmaError {
    #[snafu(display("model EMA requires at least one selected parameter"))]
    EmptySelection,
    #[snafu(display("model EMA decay must be finite and in [0, 1)"))]
    InvalidDecay,
    #[snafu(display("model EMA update mask must be a scalar F32 tensor"))]
    InvalidUpdateMask,
    #[snafu(display("model EMA can only track F32 parameter {path:?}, found {dtype:?}"))]
    UnsupportedParameterDType { path: String, dtype: DType },
    #[snafu(display("model EMA parameter layout overflows for {path:?}"))]
    ParameterSizeOverflow { path: String },
    #[snafu(transparent)]
    Model { source: rxla_nn::Error },
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
}

pub type ModelEmaResult<T, E = ModelEmaError> = std::result::Result<T, E>;

pub struct ModelEmaParameter {
    path: String,
    shape: Vec<i64>,
    offset: usize,
    elements: usize,
}

pub struct ModelEmaState {
    path: String,
    shape: Vec<i64>,
    offset: usize,
    elements: usize,
}

impl ModelEmaState {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.elements
    }
}

impl ModelEmaParameter {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.elements
    }
}

/// Device-resident EMA state produced by [`apply_model_ema`].
///
/// Its resident storage is transform-private; [`Self::parameters`] and
/// [`Self::states`] describe canonical slices for inference checkpoint export.
pub struct ModelEma {
    parameters: Vec<ModelEmaParameter>,
    states: Vec<ModelEmaState>,
    values: TransformState,
    updates: TransformState,
}

impl ModelEma {
    pub fn parameters(&self) -> &[ModelEmaParameter] {
        &self.parameters
    }

    pub fn states(&self) -> &[ModelEmaState] {
        &self.states
    }

    pub fn updates_state(&self) -> &TransformState {
        &self.updates
    }

    /// Borrow the single packed EMA buffer. [`Self::parameters`] describes the
    /// canonical parameter slices; packing avoids one state/fusion per tensor
    /// in the hot training step.
    pub fn packed_buffer<'session>(
        &self,
        session: &'session Session,
    ) -> ModelEmaResult<&'session Buffer> {
        Ok(session.state(self.values.slot())?)
    }
}

/// Append a YOLOX-style model EMA update to an already constructed training
/// step. Call this after the optimizer so it observes the new parameter values.
///
/// The effective decay ramps as `decay * (1 - exp(-updates / 2000))`, matching
/// the official YOLOX `ModelEMA`. The first update uses the step-start model
/// parameter as its exact initial EMA value, so no host-side checkpoint copy is
/// required and optimizer plus EMA remain one compiled device program.
pub fn apply_model_ema(
    model: &mut AppliedModel,
    selection: &ParameterSelection,
    decay: f32,
) -> ModelEmaResult<ModelEma> {
    apply_model_ema_inner(model, selection, decay, None)
}

/// Append a YOLOX-style EMA that advances only when `update_mask` is one.
///
/// This composes with gradient accumulation: microbatches may update model
/// state, but the EMA counter and values follow completed optimizer steps just
/// like a host-side `optimizer.step(); ema.update(model)` sequence.
pub fn apply_model_ema_when(
    model: &mut AppliedModel,
    selection: &ParameterSelection,
    decay: f32,
    update_mask: &Tensor,
) -> ModelEmaResult<ModelEma> {
    ensure!(
        update_mask.shape().is_empty() && update_mask.dtype() == DType::F32,
        InvalidUpdateMaskSnafu
    );
    apply_model_ema_inner(model, selection, decay, Some(update_mask))
}

fn apply_model_ema_inner(
    model: &mut AppliedModel,
    selection: &ParameterSelection,
    decay: f32,
    update_mask: Option<&Tensor>,
) -> ModelEmaResult<ModelEma> {
    ensure!(!selection.is_empty(), EmptySelectionSnafu);
    ensure!(
        decay.is_finite() && (0.0..1.0).contains(&decay),
        InvalidDecaySnafu
    );
    model.validate_resident_parameters(selection)?;
    for (_, spec) in selection.parameters() {
        ensure!(
            spec.dtype() == DType::F32,
            UnsupportedParameterDTypeSnafu {
                path: spec.path(),
                dtype: spec.dtype(),
            }
        );
    }

    let initial_parameters = model.parameter_tensors(selection)?;
    let current_parameters = model.current_resident_parameter_tensors(selection)?;
    let initial_states = model.initial_model_state_tensors().to_vec();
    let current_states = model.current_model_state_tensors()?;
    let updates = model.transform_state("__model_ema.updates", &[], DType::F32)?;
    let increment = match update_mask {
        Some(mask) => mask.clone(),
        None => updates.value().scalar(1.0)?,
    };
    let next_updates = updates.value().add(&increment)?;
    let effective_decay = next_updates
        .mul_scalar(-1.0 / 2000.0)?
        .exp()?
        .neg()?
        .add_scalar(1.0)?
        .mul_scalar(decay)?;
    let first_update = updates.value().eq_mask(&updates.value().scalar(0.0)?)?;

    let mut initial_flat = Vec::with_capacity(selection.len());
    let mut current_flat = Vec::with_capacity(selection.len());
    let mut parameters = Vec::with_capacity(selection.len());
    let mut states = Vec::new();
    let mut offset = 0usize;
    for (((_, spec), initial), current) in selection
        .parameters()
        .zip(initial_parameters)
        .zip(current_parameters)
    {
        let elements_i64 = spec
            .shape()
            .iter()
            .try_fold(1_i64, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| ModelEmaError::ParameterSizeOverflow {
                path: spec.path().to_owned(),
            })?;
        let elements =
            usize::try_from(elements_i64).map_err(|_| ModelEmaError::ParameterSizeOverflow {
                path: spec.path().to_owned(),
            })?;
        let next_offset =
            offset
                .checked_add(elements)
                .ok_or_else(|| ModelEmaError::ParameterSizeOverflow {
                    path: spec.path().to_owned(),
                })?;
        initial_flat.push(initial.reshape([elements_i64])?);
        current_flat.push(current.reshape([elements_i64])?);
        parameters.push(ModelEmaParameter {
            path: spec.path().to_owned(),
            shape: spec.shape().to_vec(),
            offset,
            elements,
        });
        offset = next_offset;
    }
    for ((spec, initial), current) in model
        .schema()
        .states()
        .iter()
        .zip(initial_states)
        .zip(current_states)
    {
        // Official ModelEMA updates floating entries from state_dict; integer
        // counters and RNG state retain their ordinary model semantics.
        if spec.dtype() != DType::F32 {
            continue;
        }
        let elements_i64 = spec
            .shape()
            .iter()
            .try_fold(1_i64, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| ModelEmaError::ParameterSizeOverflow {
                path: spec.path().to_owned(),
            })?;
        let elements =
            usize::try_from(elements_i64).map_err(|_| ModelEmaError::ParameterSizeOverflow {
                path: spec.path().to_owned(),
            })?;
        let next_offset =
            offset
                .checked_add(elements)
                .ok_or_else(|| ModelEmaError::ParameterSizeOverflow {
                    path: spec.path().to_owned(),
                })?;
        initial_flat.push(initial.reshape([elements_i64])?);
        current_flat.push(current.reshape([elements_i64])?);
        states.push(ModelEmaState {
            path: spec.path().to_owned(),
            shape: spec.shape().to_vec(),
            offset,
            elements,
        });
        offset = next_offset;
    }
    // Express EMA arithmetic once over one logical parameter vector. XLA can
    // fuse this into a large bandwidth-efficient update instead of launching a
    // separate tiny elementwise fusion for every bias and BatchNorm vector.
    let initial = Tensor::concatenate(&initial_flat, 0)?;
    let current = Tensor::concatenate(&current_flat, 0)?;
    let packed_elements =
        i64::try_from(offset).map_err(|_| ModelEmaError::ParameterSizeOverflow {
            path: "<packed>".to_owned(),
        })?;
    let values = model.transform_state("__model_ema.parameters", &[packed_elements], DType::F32)?;
    let previous = values.value();
    let first = first_update.broadcast_to(previous.shape())?;
    let previous = first.select(&initial, previous)?;
    let packed_decay = effective_decay.broadcast_to(previous.shape())?;
    let candidate = previous.add(
        &current
            .sub(&previous)?
            .mul(&packed_decay.neg()?.add_scalar(1.0)?)?,
    )?;
    let next_packed = match update_mask {
        Some(mask) => mask
            .broadcast_to(previous.shape())?
            .select(&candidate, &previous)?,
        None => candidate,
    };
    model.write_transform_states(&[(&values, &next_packed), (&updates, &next_updates)])?;

    Ok(ModelEma {
        parameters,
        states,
        values,
        updates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apply_model_sgd;
    use rxla_core::{CacheLimits, Client, Compiler, Tensor};
    use rxla_nn::{Cx, Linear, Model, Result, TensorApply};

    fn linear_loss(cx: Cx) -> Result<Tensor> {
        let input = cx.input(&[1, 1])?;
        let prediction = input.apply(&cx.named_layer("linear", Linear::new(1).bias(false))?)?;
        Ok(prediction.square()?.sum(&[0, 1], false)?)
    }

    #[test]
    fn ema_lowers_after_optimizer_as_hidden_resident_state() {
        let (selection, mut model) = Model::new(linear_loss).trace_resident_all().unwrap();
        let loss = model.outputs()[0].clone();
        apply_model_sgd(&mut model, &selection, &loss, 0.1).unwrap();
        let ema = apply_model_ema(&mut model, &selection, 0.9998).unwrap();

        assert_eq!(ema.parameters.len(), 1);
        assert_eq!(ema.parameters()[0].path(), "linear.weight");
        assert_eq!(model.states().count(), 2);
        let prepared = model.prepare_stateful().unwrap();
        let stablehlo = std::str::from_utf8(prepared.lowered_program().code()).unwrap();
        assert!(stablehlo.contains("stablehlo.exponential"));
        assert!(stablehlo.contains("stablehlo.select"));
    }

    #[test]
    fn ema_rejects_empty_selection_and_invalid_decay() {
        let (_, mut model) = Model::new(linear_loss).trace_resident_all().unwrap();
        let empty = model.schema().select_under("missing");
        assert!(matches!(
            apply_model_ema(&mut model, &empty, 0.9998),
            Err(ModelEmaError::EmptySelection)
        ));
        let all = model.schema().select_all();
        assert!(matches!(
            apply_model_ema(&mut model, &all, 1.0),
            Err(ModelEmaError::InvalidDecay)
        ));
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn ema_consumes_post_optimizer_parameter_and_persists_between_steps() {
        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .expect("load CPU plugin");
        let (selection, mut model) = Model::new(linear_loss).trace_resident_all().unwrap();
        let loss = model.outputs()[0].clone();
        apply_model_sgd(&mut model, &selection, &loss, 0.1).unwrap();
        let ema = apply_model_ema(&mut model, &selection, 0.9998).unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let program = model.compile_stateful(&mut compiler).unwrap();
        let weight = client.buffer(&[1, 1], &[1.0]).unwrap();
        let mut session = program
            .session()
            .parameter("linear.weight", weight)
            .unwrap()
            .build()
            .unwrap();
        let input = client.buffer(&[1, 1], &[2.0]).unwrap();

        let _: Buffer = session.run(&input).unwrap();
        let first = ema
            .packed_buffer(session.raw())
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[0];
        let decay1 = 0.9998 * (1.0 - (-1.0_f32 / 2000.0).exp());
        let expected1 = 1.0 * decay1 + 0.2 * (1.0 - decay1);
        assert!((first - expected1).abs() < 1e-6, "{first} != {expected1}");

        let _: Buffer = session.run(&input).unwrap();
        let second = ema
            .packed_buffer(session.raw())
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[0];
        let decay2 = 0.9998 * (1.0 - (-2.0_f32 / 2000.0).exp());
        let expected2 = expected1 * decay2 + 0.04 * (1.0 - decay2);
        assert!((second - expected2).abs() < 1e-6, "{second} != {expected2}");
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn ema_tracks_model_state_from_its_pre_forward_value() {
        fn stateful_loss(cx: Cx) -> Result<Tensor> {
            let input = cx.input([1, 1])?;
            let running = cx.state("running", [1], DType::F32)?;
            running.add_(&cx, &cx.constant([1], [2.0])?)?;
            let prediction = input.apply(&cx.named_layer("linear", Linear::new(1).bias(false))?)?;
            Ok(prediction.square()?.sum([0, 1], false)?)
        }

        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .expect("load CPU plugin");
        let (selection, mut model) = Model::new(stateful_loss).trace_resident_all().unwrap();
        let loss = model.outputs()[0].clone();
        apply_model_sgd(&mut model, &selection, &loss, 0.0).unwrap();
        let ema = apply_model_ema(&mut model, &selection, 0.9998).unwrap();
        assert_eq!(ema.states().len(), 1);
        assert_eq!(ema.states()[0].path(), "running");
        let state_range = ema.states()[0].range();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let program = model.compile_stateful(&mut compiler).unwrap();
        let weight = client.buffer(&[1, 1], &[1.0]).unwrap();
        let mut session = program
            .session()
            .parameter("linear.weight", weight)
            .unwrap()
            .build()
            .unwrap();
        let input = client.buffer(&[1, 1], &[1.0]).unwrap();

        let _: Buffer = session.run(&input).unwrap();
        let packed = ema
            .packed_buffer(session.raw())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let first = packed[state_range.clone()][0];
        let decay1 = 0.9998 * (1.0 - (-1.0_f32 / 2000.0).exp());
        let expected1 = 2.0 * (1.0 - decay1);
        assert!((first - expected1).abs() < 1e-6, "{first} != {expected1}");

        let _: Buffer = session.run(&input).unwrap();
        let packed = ema
            .packed_buffer(session.raw())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let second = packed[state_range][0];
        let decay2 = 0.9998 * (1.0 - (-2.0_f32 / 2000.0).exp());
        let expected2 = expected1 * decay2 + 4.0 * (1.0 - decay2);
        assert!((second - expected2).abs() < 1e-6, "{second} != {expected2}");
    }
}
