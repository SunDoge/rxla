//! IR transforms for training parameter-effect models.
//!
//! Training is functional: transforms consume an [`rxla_nn::AppliedModel`] and
//! emit replacement parameter tensors as part of the compiled program.

mod data_rng;
mod model_adam;
pub use data_rng::{DataRng, SampleRng};
mod lora;
pub use lora::select_lora;
pub use model_adam::{
    AdamOptions, ModelAdamError, ModelAdamOutputPlan, ModelAdamResult, ModelAdamState,
    ModelAdamStep, ModelAdamUpdate, apply_model_adam, prepare_model_adam,
};
mod model_ema;
pub use model_ema::{
    ModelEma, ModelEmaError, ModelEmaParameter, ModelEmaResult, ModelEmaState, apply_model_ema,
    apply_model_ema_when,
};
mod model_sgd;
pub use model_sgd::{
    ModelSgdAccumulation, ModelSgdError, ModelSgdOutputPlan, ModelSgdResult, ModelSgdStep,
    ModelSgdUpdate, SgdOptions, apply_model_sgd, apply_model_sgd_accumulating_with_learning_rate,
    apply_model_sgd_with_learning_rate, apply_model_sgd_with_options, prepare_model_sgd,
};
mod pipeline;
pub use pipeline::{BoundedPipeline, PipelineError, PipelineResult, ShufflePool};
mod rl;
pub use rl::{RlError, RlResult, clipped_policy_gradient_loss, policy_gradient_loss};
mod yolo;
pub use yolo::{
    DecodedYoloBatch, DetectionTarget, SimOtaAssignments, SimOtaConfig, SimOtaError, SimOtaInput,
    SimOtaResult, YoloHead, YoloLrSchedule,
};
