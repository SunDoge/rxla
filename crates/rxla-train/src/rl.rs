//! Tensor objectives shared by environment RL and language-model fine-tuning.

use rxla_core::Tensor;
use snafu::{Snafu, ensure};

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum RlError {
    #[snafu(display("PPO clip epsilon must be finite and in [0, 1)"))]
    InvalidClip,
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
}

pub type RlResult<T, E = RlError> = std::result::Result<T, E>;

/// Mean negative log-probability weighted by detached returns or advantages.
///
/// `actions` and `advantages` have the shape of `logits` with `class_axis`
/// removed. Zero advantages therefore provide ordinary padding/masking.
pub fn policy_gradient_loss(
    logits: &Tensor,
    actions: &Tensor,
    advantages: &Tensor,
    class_axis: usize,
) -> RlResult<Tensor> {
    let log_probabilities = logits
        .cross_entropy_with_indices(actions, class_axis)?
        .neg()?;
    let axes = (0..advantages.shape().len()).collect::<Vec<_>>();
    Ok(log_probabilities
        .mul(&advantages.detach()?)?
        .mean(axes, false)?
        .neg()?)
}

/// PPO clipped surrogate objective over arbitrary leading token/rollout axes.
///
/// `old_log_probabilities` and `advantages` match `actions`. Padding positions
/// can be represented by zero advantages. The old policy and advantages are
/// detached explicitly, keeping gradients in the new policy only.
pub fn clipped_policy_gradient_loss(
    logits: &Tensor,
    actions: &Tensor,
    old_log_probabilities: &Tensor,
    advantages: &Tensor,
    class_axis: usize,
    clip_epsilon: f32,
) -> RlResult<Tensor> {
    ensure!(
        clip_epsilon.is_finite() && (0.0..1.0).contains(&clip_epsilon),
        InvalidClipSnafu
    );
    let new_log_probabilities = logits
        .cross_entropy_with_indices(actions, class_axis)?
        .neg()?;
    let ratio = new_log_probabilities
        .sub(&old_log_probabilities.detach()?)?
        .exp()?;
    let advantages = advantages.detach()?;
    let unclipped = ratio.mul(&advantages)?;
    let clipped = ratio
        .clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)?
        .mul(&advantages)?;
    let axes = (0..advantages.shape().len()).collect::<Vec<_>>();
    Ok(unclipped.minimum(&clipped)?.mean(axes, false)?.neg()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{DType, Tracer};

    #[test]
    fn objectives_support_token_shaped_actions_and_masks() {
        let graph = Tracer::default();
        let logits = graph.input([2, 3, 7]).unwrap();
        let actions = graph.input_dtype([2, 3], DType::I32).unwrap();
        let old_log_probabilities = graph.input([2, 3]).unwrap();
        let advantages = graph.input([2, 3]).unwrap();
        assert_eq!(
            policy_gradient_loss(&logits, &actions, &advantages, 2)
                .unwrap()
                .shape(),
            []
        );
        let loss = clipped_policy_gradient_loss(
            &logits,
            &actions,
            &old_log_probabilities,
            &advantages,
            2,
            0.2,
        )
        .unwrap();
        assert_eq!(loss.shape(), []);
        assert_eq!(loss.grad(&[logits]).unwrap()[0].shape(), [2, 3, 7]);
    }

    #[test]
    fn clipped_objective_rejects_invalid_epsilon() {
        let graph = Tracer::default();
        let logits = graph.input([1, 2]).unwrap();
        let actions = graph.input_dtype([1], DType::I32).unwrap();
        let old = graph.input([1]).unwrap();
        let advantages = graph.input([1]).unwrap();
        assert!(matches!(
            clipped_policy_gradient_loss(&logits, &actions, &old, &advantages, 1, 1.0),
            Err(RlError::InvalidClip)
        ));
    }
}
