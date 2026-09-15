//! Canonical Pliron graph ownership, traversal, and structural queries.

use super::*;

impl Default for IrGraph {
    fn default() -> Self {
        let mut ctx = Context::new();
        let module = ModuleOp::new(
            &mut ctx,
            "rxla_program"
                .try_into()
                .expect("static module name is valid"),
        );
        Self { ctx, module }
    }
}

impl IrGraph {
    pub(super) fn verify(&self, stage: &'static str) -> Result<()> {
        verify_op(&self.module, &self.ctx).map_err(|error| IrError::Verification {
            stage,
            message: error.to_string(),
        })
    }

    pub(super) fn reachable_values(&self, outputs: &[Value]) -> Result<HashSet<Value>> {
        let mut reachable = HashSet::new();
        let mut worklist = outputs.to_vec();
        while let Some(value) = worklist.pop() {
            if !reachable.insert(value) {
                continue;
            }
            let operation_ptr = value.defining_op().ok_or(IrError::InvalidValue {
                operation: "walking source IR reachability",
            })?;
            let operation = operation_ptr.deref(&self.ctx);
            let op = Operation::get_op_dyn(operation_ptr, &self.ctx);
            if op.as_ref().is::<WithGradientOp>() || op.as_ref().is::<WithElementwiseDerivativeOp>()
            {
                // Custom derivative operands are AD metadata, not forward data
                // dependencies. When a derivative is itself requested, its
                // generated SSA expression is reached through the output root.
                worklist.push(operation.get_operand(0));
            } else {
                worklist.extend(operation.operands());
            }
            // Region operations capture outer SSA values through their nested
            // blocks. Those captures are data dependencies even though they
            // are not operands of the enclosing operation (a nested result
            // cannot dominate its parent).
            for region_index in 0..operation.num_regions() {
                let region = operation.get_region(region_index);
                for block in region.deref(&self.ctx).iter(&self.ctx) {
                    for nested in block.deref(&self.ctx).iter(&self.ctx) {
                        worklist.extend(nested.deref(&self.ctx).operands());
                    }
                }
            }
        }
        Ok(reachable)
    }

    pub(super) fn planning_snapshot(&self, outputs: &[Value]) -> Result<PlanningFacts> {
        let reachable = self.reachable_values(outputs)?;
        let mut compact_values = HashMap::new();
        let mut parameter_values = Vec::new();
        let mut constraints = Vec::new();
        for operation_ptr in self.operations() {
            let operation = operation_ptr.deref(&self.ctx);
            let op = Operation::get_op_dyn(operation_ptr, &self.ctx);
            for result in operation
                .results()
                .filter(|result| reachable.contains(result))
            {
                let value = compact_values.len();
                compact_values.insert(result, value as i64 + 1);
                if op.as_ref().is::<ParameterOp>() {
                    parameter_values.push(value);
                }
                if let Some(attribute) = operation
                    .attributes
                    .get::<ShardingAttr>(&sharding_attr_key())
                {
                    constraints.push(ShardingConstraint {
                        value,
                        sharding: attribute.sharding(),
                    });
                }
            }
        }
        Ok(PlanningFacts::new(
            parameter_values.len(),
            compact_values.len(),
            outputs.len(),
            constraints,
        ))
    }

    pub(super) fn tensor_type(&self, dims: &[i64], dtype: DType) -> TypeHandle {
        supported_dtype(dtype).expect("unsupported dtype must be rejected before IR construction");
        RankedTensorType::get(&self.ctx, ShapeAttr::new(dims), ElementTypeAttr::new(dtype)).into()
    }

    pub(super) fn push<O: PlironOp + OneResultInterface + Clone + 'static>(
        &mut self,
        op: O,
    ) -> Value {
        let result = op.get_result(&self.ctx);
        self.push_results(op);
        result
    }

    /// Insert an operation in the root block and return all of its SSA
    /// results. Zero- and multi-result operations are first-class internally.
    pub(super) fn push_results<O: PlironOp + Clone + 'static>(&mut self, op: O) -> Vec<Value> {
        let results = op.get_operation().deref(&self.ctx).results().collect();
        self.module
            .append_operation(&mut self.ctx, op.get_operation(), 0);
        results
    }

    pub(super) fn operations(&self) -> impl Iterator<Item = pliron::context::Ptr<Operation>> + '_ {
        self.module
            .get_body(&self.ctx, 0)
            .deref(&self.ctx)
            .iter(&self.ctx)
    }

    pub(super) fn value_type(&self, value: Value) -> Result<TensorType> {
        let ty = value.get_type(&self.ctx);
        let ty = ty.deref(&self.ctx);
        let ranked = ty
            .downcast_ref::<RankedTensorType>()
            .ok_or(IrError::ExpectedTensorType {
                operation: "reading an IR value type",
            })?;
        ranked
            .verify(&self.ctx)
            .map_err(|error| IrError::Verification {
                stage: "IR value type query",
                message: error.to_string(),
            })?;
        Ok(TensorType {
            dims: ranked.shape.values(),
            dtype: ranked.element.value(),
        })
    }
}
