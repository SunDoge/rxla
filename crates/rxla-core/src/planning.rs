use crate::{LoweredProgram, Result, Sharding, err};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PlanningPolicy {
    #[default]
    SingleDevice,
    Auto(AutoShardingOptions),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoShardingOptions {
    mesh: crate::Mesh,
    /// Portable monotonic effort hint. Backends may quantize it rather than
    /// treating it as an exact count of explored states.
    max_search_states: usize,
}

impl AutoShardingOptions {
    pub fn new(mesh: crate::Mesh, max_search_states: usize) -> Result<Self> {
        if max_search_states == 0 {
            return Err(err("auto-sharding search budget must be nonzero"));
        }
        Ok(Self {
            mesh,
            max_search_states,
        })
    }

    pub fn mesh(&self) -> &crate::Mesh {
        &self.mesh
    }

    /// Requested portable search effort. The current XLA backend maps ranges to
    /// O0–O3; this is not a promise of an exact number of explored states.
    pub fn max_search_states(&self) -> usize {
        self.max_search_states
    }

    pub(crate) fn effort_level(&self) -> rxla_xla_proto::xla::execution_options::EffortLevel {
        use rxla_xla_proto::xla::execution_options::EffortLevel;
        match self.max_search_states {
            1..=16 => EffortLevel::EffortO0,
            17..=256 => EffortLevel::EffortO1,
            257..=4096 => EffortLevel::EffortO2,
            _ => EffortLevel::EffortO3,
        }
    }
}

/// A validated planner result. The baseline is deliberately represented as a
/// one-stage plan so local and future partitioned execution share one boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlan {
    policy: PlanningPolicy,
    stages: Vec<ExecutionStage>,
    shardings: Vec<PlannedSharding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionStage {
    index: usize,
    input_count: usize,
    output_count: usize,
    device_count: usize,
}

/// The planner's placement decision for one zero-based semantic SSA value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedSharding {
    value: usize,
    decision: ShardingDecision,
}

/// Whether placement is fixed by the user or delegated to the backend's
/// auto-SPMD search. An `Auto` entry is intent, not a fabricated final layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardingDecision {
    Explicit(Sharding),
    Auto { mesh: crate::Mesh },
}

impl ExecutionPlan {
    pub fn policy(&self) -> &PlanningPolicy {
        &self.policy
    }

    pub fn stages(&self) -> &[ExecutionStage] {
        &self.stages
    }

    pub fn required_devices(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.device_count)
            .max()
            .unwrap_or(0)
    }

    pub fn shardings(&self) -> &[PlannedSharding] {
        &self.shardings
    }

    /// True when this plan must pass through SPMD partitioning before PJRT
    /// compilation. Callers must not silently execute it as a local program.
    pub fn requires_spmd_partitioning(&self) -> bool {
        self.required_devices() > 1
    }
}

impl PlannedSharding {
    pub fn value(&self) -> usize {
        self.value
    }

    pub fn decision(&self) -> &ShardingDecision {
        &self.decision
    }

    pub fn sharding(&self) -> Option<&Sharding> {
        match &self.decision {
            ShardingDecision::Explicit(sharding) => Some(sharding),
            ShardingDecision::Auto { .. } => None,
        }
    }

    pub fn is_constraint(&self) -> bool {
        matches!(self.decision, ShardingDecision::Explicit(_))
    }

    pub fn is_auto(&self) -> bool {
        matches!(self.decision, ShardingDecision::Auto { .. })
    }
}

impl ExecutionStage {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    pub fn device_count(&self) -> usize {
        self.device_count
    }
}

pub use rxla_ir::ShardingConstraint;

/// Backend-neutral planning facts extracted from semantic SSA.
#[derive(Clone)]
pub(crate) struct PlanningSnapshot(rxla_ir::PlanningFacts);

impl From<rxla_ir::PlanningFacts> for PlanningSnapshot {
    fn from(facts: rxla_ir::PlanningFacts) -> Self {
        Self(facts)
    }
}

impl PlanningSnapshot {
    #[cfg(test)]
    pub(crate) fn sharding_constraints(&self) -> &[ShardingConstraint] {
        self.0.sharding_constraints()
    }

    pub(crate) fn plan(&self, policy: PlanningPolicy) -> Result<ExecutionPlan> {
        let constraints = self.0.sharding_constraints().to_vec();
        let (device_count, shardings) = match &policy {
            PlanningPolicy::SingleDevice => {
                if constraints.iter().any(|constraint| {
                    !constraint
                        .sharding
                        .mesh()
                        .device_count()
                        .is_ok_and(|count| count == 1)
                }) {
                    return Err(err(
                        "multi-device sharding constraint requires an auto-sharding planner",
                    ));
                }
                (
                    1,
                    constraints
                        .into_iter()
                        .map(|constraint| PlannedSharding {
                            value: constraint.value,
                            decision: ShardingDecision::Explicit(constraint.sharding),
                        })
                        .collect(),
                )
            }
            PlanningPolicy::Auto(options) => {
                if constraints
                    .iter()
                    .any(|constraint| constraint.sharding.mesh() != options.mesh())
                {
                    return Err(err(
                        "sharding constraint mesh does not match the auto-sharding mesh",
                    ));
                }
                let constrained = constraints
                    .into_iter()
                    .map(|constraint| (constraint.value, constraint.sharding))
                    .collect::<std::collections::HashMap<_, _>>();
                let shardings = (0..self.0.value_count())
                    .map(|value| PlannedSharding {
                        value,
                        decision: constrained.get(&value).cloned().map_or_else(
                            || ShardingDecision::Auto {
                                mesh: options.mesh().clone(),
                            },
                            ShardingDecision::Explicit,
                        ),
                    })
                    .collect();
                (options.mesh().device_count()?, shardings)
            }
        };
        Ok(ExecutionPlan {
            policy,
            stages: vec![ExecutionStage {
                index: 0,
                input_count: self.0.input_count(),
                output_count: self.0.output_count(),
                device_count,
            }],
            shardings,
        })
    }

    pub(crate) fn lower_spmd(
        &self,
        lowered: &LoweredProgram,
        plan: &ExecutionPlan,
        device_ids: &[i64],
    ) -> Result<LoweredProgram> {
        if device_ids.len() != plan.required_devices() {
            return Err(err("SPMD device assignment does not match execution plan"));
        }
        // StableHLO already carries SDY constraints and replicated boundaries.
        // Compile options select XLA's partitioner; planning does not rewrite a
        // second HLO-proto representation.
        Ok(lowered.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mesh, PartitionSpec, Sharding, Tracer};

    #[test]
    fn single_device_is_a_validated_one_stage_plan() {
        let program = Tracer::trace(|tracer| {
            let x = tracer.input(&[4])?;
            Ok(vec![x.exp()?])
        })
        .unwrap();
        let plan = program.plan(PlanningPolicy::SingleDevice).unwrap();
        assert_eq!(plan.required_devices(), 1);
        assert_eq!(plan.stages().len(), 1);
        assert_eq!(plan.stages()[0].input_count(), 1);
        assert_eq!(plan.stages()[0].output_count(), 1);
    }

    #[test]
    fn auto_sharding_generates_a_complete_multi_device_placement() {
        let tracer = Tracer::new();
        let mesh = Mesh::new([("data", 2)]).unwrap();
        let x = tracer
            .input(&[4])
            .unwrap()
            .with_sharding(Sharding::partitioned(
                mesh.clone(),
                PartitionSpec::new([Some("data")]).unwrap(),
            ))
            .unwrap();
        let program = tracer.program(vec![x.exp().unwrap()]).unwrap();
        assert_eq!(program.lowered.format(), "mlir");
        assert!(program.plan(PlanningPolicy::SingleDevice).is_err());
        let plan = program
            .plan(PlanningPolicy::Auto(
                AutoShardingOptions::new(mesh, 64).unwrap(),
            ))
            .unwrap();
        assert_eq!(plan.required_devices(), 2);
        assert!(plan.requires_spmd_partitioning());
        assert_eq!(plan.shardings().len(), 2);
        assert!(plan.shardings()[0].is_constraint());
        assert!(plan.shardings()[1].is_auto());

        let prepared = program
            .planning
            .lower_spmd(&program.lowered, &plan, &[0, 1])
            .unwrap();
        let mlir = str::from_utf8(prepared.code()).unwrap();
        assert!(mlir.contains("mhlo.num_partitions = 2 : i32"));
        assert!(mlir.contains("sdy.mesh @mesh0 = <[\"data\"=2]>"));
        assert!(mlir.contains("sdy.sharding_constraint %arg0 <@mesh0, [{\"data\"}]>"));
    }

    #[test]
    fn portable_search_budget_maps_monotonically_to_backend_effort() {
        use rxla_xla_proto::xla::execution_options::EffortLevel;
        let mesh = Mesh::new([("data", 1)]).unwrap();
        for (budget, expected) in [
            (1, EffortLevel::EffortO0),
            (17, EffortLevel::EffortO1),
            (257, EffortLevel::EffortO2),
            (4097, EffortLevel::EffortO3),
        ] {
            assert_eq!(
                AutoShardingOptions::new(mesh.clone(), budget)
                    .unwrap()
                    .effort_level(),
                expected
            );
        }
    }
}
