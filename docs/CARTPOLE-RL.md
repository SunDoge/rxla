# CartPole policy-gradient experiment

`rxla-train` contains a small end-to-end REINFORCE experiment. CartPole physics,
trajectory collection, and return normalization run on the Rust host. Policy
inference, the policy-gradient objective, autodiff, and Adam execute through
RXLA/PJRT.

```console
cargo run --release -p rxla-train --example cartpole_reinforce -- \
  --plugin "$PJRT_CPU_PLUGIN_PATH" \
  --iterations 100
```

The default run uses 32 environments, a 32-step rollout, a two-layer policy,
and batches all 1,024 transitions into one Adam update. Inference and training
are separately compiled static-shape programs sharing one named parameter map;
Adam moments remain resident in the training session.

On the development CPU plugin with seed 42, the recent-100-episode mean return
started at 16.1, reached 210.3 at iteration 80, and ended at 165.6 at iteration
99. This is evidence of a working learning loop, not a stable benchmark or a
claim that plain REINFORCE is the preferred production algorithm.

The example intentionally keeps environment and rollout policy outside the
tensor library. Its reusable boundary is the named policy parameter map plus
observations, sampled actions, and scalar training weights. PPO, preference
optimization, and language-model RL need richer loss construction and rollout
systems, but should not require a second optimizer or autodiff architecture.
