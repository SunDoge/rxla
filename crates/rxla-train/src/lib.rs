//! IR transforms for training parameter-effect models.
//!
//! Training is functional: transforms consume an [`rxla_nn::AppliedModel`] and
//! emit replacement parameter tensors as part of the compiled program.

mod data_rng;
mod model_adam;
pub use data_rng::{DataRng, SampleRng};
pub use model_adam::{
    AdamOptions, ModelAdamError, ModelAdamOutputPlan, ModelAdamResult, ModelAdamState,
    ModelAdamStep, ModelAdamUpdate, apply_model_adam, prepare_model_adam,
};
mod model_sgd;
pub use model_sgd::{
    ModelSgdError, ModelSgdOutputPlan, ModelSgdResult, ModelSgdStep, ModelSgdUpdate,
    apply_model_sgd, prepare_model_sgd,
};
mod pipeline;
pub use pipeline::{BoundedPipeline, PipelineError, PipelineResult};
