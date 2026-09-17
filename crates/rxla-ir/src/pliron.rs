//! Pliron-backed tensor IR construction and transformation layer.
//!
//! The mutable Pliron [`Context`] intentionally lives only while IR is built,
//! transformed, and lowered. Executable program snapshots own a compact
//! immutable result, so they remain cheap to clone and can cross the execution
//! boundary without leaking arena handles or context lifetimes.

mod analysis;
mod attributes;
mod dialect;
mod graph;
mod stablehlo;
#[cfg(test)]
mod tests;
mod verification;

pub use crate::{Binary, Comparison, IntegerBinary, Op, Reduction, SliceAxis, TensorType, Unary};
pub use crate::{Conv2dOptions, ConvTranspose2dOptions, Pool2dOptions};
use attributes::{sharding_attr_key, supported_dtype};
use dialect::*;

use crate::{IrError, PlanningFacts, Result, Sharding, ShardingConstraint};
use pliron::basic_block::BasicBlock;
use pliron::{
    builtin::{
        attributes::StringAttr,
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NResultsInterface, OneOpdInterface,
            OneResultInterface, SameOperandsAndResultType, SameOperandsType, SameResultsType,
            SingleBlockRegionInterface,
        },
        ops::ModuleOp,
    },
    common_traits::Verify,
    context::Context,
    derive::{pliron_attr, pliron_op, pliron_type},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::{Op as PlironOp, verify_op},
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use rxla_pjrt::DType;
use std::collections::{HashMap, HashSet};

macro_rules! construct_op {
    ($name:ident, $ctx:expr, $results:expr, $operands:expr) => {{
        <$name as PlironOp>::from_operation(Operation::new(
            $ctx,
            $name::get_concrete_op_info(),
            $results,
            $operands,
            vec![],
            0,
        ))
    }};
}

mod builder;
mod program;
pub use analysis::SemanticProgram;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringTarget {
    Portable,
    CudaF16Attention,
    CudaF16Compute,
    CudaBf16Compute,
}
pub use program::{ProgramIr, SsaId, StableHloProgram};

pub struct IrGraph {
    ctx: Context,
    module: ModuleOp,
}

impl IrGraph {
    #[cfg(test)]
    fn parameter(&mut self, dims: &[i64]) -> Value {
        let number = self
            .operations()
            .filter(|&op| Operation::is_op::<ParameterOp>(op, &self.ctx))
            .count();
        self.parameter_number(number, dims, DType::F32)
    }

    #[cfg(test)]
    fn parameter_number(&mut self, number: usize, dims: &[i64], dtype: DType) -> Value {
        self.parameter_typed(number, &TensorType::static_shape(dims.to_vec(), dtype))
    }

    fn parameter_typed(&mut self, number: usize, result: &TensorType) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(ParameterOp, &mut self.ctx, vec![ty], vec![]);
        op.set_attr_number(&self.ctx, StringAttr::new(number.to_string()));
        self.push(op)
    }

    fn constant_f32(&mut self, dims: &[i64], values: Vec<f32>) -> Value {
        self.constant_bytes(
            dims,
            DType::F32,
            values.into_iter().flat_map(f32::to_le_bytes).collect(),
        )
    }

    fn constant_i32(&mut self, dims: &[i64], values: Vec<i32>) -> Value {
        self.constant_bytes(
            dims,
            DType::I32,
            values.into_iter().flat_map(i32::to_le_bytes).collect(),
        )
    }

    fn constant_bytes(&mut self, dims: &[i64], dtype: DType, bytes: Vec<u8>) -> Value {
        let ty = self.tensor_type(dims, dtype);
        let op = construct_op!(ConstantOp, &mut self.ctx, vec![ty], vec![]);
        op.set_attr_value(&self.ctx, BytesAttr::new(bytes));
        self.push(op)
    }

    fn iota_typed(&mut self, result: &TensorType, axis: usize) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(IotaOp, &mut self.ctx, vec![ty], vec![]);
        op.set_attr_iota_axis(&self.ctx, AxisAttr::new(axis));
        self.push(op)
    }

    fn integer_binary_typed(
        &mut self,
        lhs: Value,
        rhs: Value,
        result: &TensorType,
        operation: IntegerBinary,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(IntegerBinaryOp, &mut self.ctx, vec![ty], vec![lhs, rhs]);
        op.set_attr_integer_binary(&self.ctx, IntegerBinaryAttr::new(operation));
        self.push(op)
    }

    fn add_typed(&mut self, lhs: Value, rhs: Value, result: &TensorType) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(AddOp, &mut self.ctx, vec![ty], vec![lhs, rhs]);
        self.push(op)
    }

    fn multiply_typed(&mut self, lhs: Value, rhs: Value, result: &TensorType) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(MultiplyOp, &mut self.ctx, vec![ty], vec![lhs, rhs]);
        self.push(op)
    }

    fn gradient_wrapper_typed(
        &mut self,
        operands: Vec<Value>,
        result: &TensorType,
        elementwise: bool,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        if elementwise {
            let op = construct_op!(
                WithElementwiseDerivativeOp,
                &mut self.ctx,
                vec![ty],
                operands
            );
            self.push(op)
        } else {
            let op = construct_op!(WithGradientOp, &mut self.ctx, vec![ty], operands);
            self.push(op)
        }
    }

    fn matmul_typed(
        &mut self,
        lhs: Value,
        rhs: Value,
        result: &TensorType,
        batch_rank: usize,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(MatmulOp, &mut self.ctx, vec![ty], vec![lhs, rhs]);
        op.set_attr_batch_rank(&self.ctx, BatchRankAttr::new(batch_rank));
        self.push(op)
    }

    fn conv2d_typed(
        &mut self,
        input: Value,
        kernel: Value,
        result: &TensorType,
        options: Conv2dOptions,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(Conv2dOp, &mut self.ctx, vec![ty], vec![input, kernel]);
        op.set_attr_options(&self.ctx, Conv2dOptionsAttr::new(options));
        self.push(op)
    }

    fn conv2d_oihw_typed(
        &mut self,
        input: Value,
        kernel: Value,
        result: &TensorType,
        options: Conv2dOptions,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(Conv2dOihwOp, &mut self.ctx, vec![ty], vec![input, kernel]);
        op.set_attr_oihw_options(&self.ctx, Conv2dOptionsAttr::new(options));
        self.push(op)
    }

    fn conv2d_kernel_gradient_typed(
        &mut self,
        input: Value,
        output_gradient: Value,
        result: &TensorType,
        options: Conv2dOptions,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(
            Conv2dKernelGradientOp,
            &mut self.ctx,
            vec![ty],
            vec![input, output_gradient]
        );
        op.set_attr_kernel_gradient_options(&self.ctx, Conv2dOptionsAttr::new(options));
        self.push(op)
    }

    fn conv_transpose2d_typed(
        &mut self,
        input: Value,
        kernel: Value,
        result: &TensorType,
        options: ConvTranspose2dOptions,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(
            ConvTranspose2dOp,
            &mut self.ctx,
            vec![ty],
            vec![input, kernel]
        );
        op.set_attr_transpose_options(&self.ctx, ConvTranspose2dOptionsAttr::new(options));
        self.push(op)
    }

    fn pool2d_typed(
        &mut self,
        input: Value,
        result: &TensorType,
        options: Pool2dOptions,
        maximum: bool,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        if maximum {
            let op = construct_op!(MaxPool2dOp, &mut self.ctx, vec![ty], vec![input]);
            op.set_attr_max_pool_options(&self.ctx, Pool2dOptionsAttr::new(options));
            self.push(op)
        } else {
            let op = construct_op!(SumPool2dOp, &mut self.ctx, vec![ty], vec![input]);
            op.set_attr_sum_pool_options(&self.ctx, Pool2dOptionsAttr::new(options));
            self.push(op)
        }
    }

    fn broadcast_typed(&mut self, input: Value, result: &TensorType, axes: Vec<usize>) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(BroadcastOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_broadcast_axes(&self.ctx, AxesAttr::new(&axes));
        self.push(op)
    }

    fn transpose_typed(
        &mut self,
        input: Value,
        result: &TensorType,
        permutation: Vec<usize>,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(TransposeOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_permutation(&self.ctx, AxesAttr::new(&permutation));
        self.push(op)
    }

    fn reverse_typed(&mut self, input: Value, result: &TensorType, axes: Vec<usize>) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(ReverseOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_reverse_axes(&self.ctx, AxesAttr::new(&axes));
        self.push(op)
    }

    fn cumsum_typed(&mut self, input: Value, result: &TensorType, axis: usize) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(CumsumOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_axis(&self.ctx, AxisAttr::new(axis));
        self.push(op)
    }

    fn slice_typed(&mut self, input: Value, result: &TensorType, spec: Vec<SliceAxis>) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(SliceOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_spec(&self.ctx, SliceSpecAttr::new(&spec));
        self.push(op)
    }

    fn slice_gradient_typed(
        &mut self,
        input: Value,
        result: &TensorType,
        spec: Vec<SliceAxis>,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(SliceGradientOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_gradient_spec(&self.ctx, SliceSpecAttr::new(&spec));
        self.push(op)
    }

    fn pad_typed(
        &mut self,
        input: Value,
        fill: Value,
        result: &TensorType,
        padding: Vec<[i64; 2]>,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(PadOp, &mut self.ctx, vec![ty], vec![input, fill]);
        op.set_attr_padding(&self.ctx, PaddingAttr::new(&padding));
        self.push(op)
    }

    fn dynamic_slice_typed(&mut self, operands: Vec<Value>, result: &TensorType) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(DynamicSliceOp, &mut self.ctx, vec![ty], operands);
        self.push(op)
    }

    fn dynamic_update_slice_typed(&mut self, operands: Vec<Value>, result: &TensorType) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(DynamicUpdateSliceOp, &mut self.ctx, vec![ty], operands);
        self.push(op)
    }

    fn take_typed(
        &mut self,
        input: Value,
        indices: Value,
        result: &TensorType,
        axis: usize,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(TakeOp, &mut self.ctx, vec![ty], vec![input, indices]);
        op.set_attr_take_axis(&self.ctx, AxisAttr::new(axis));
        self.push(op)
    }

    fn take_along_axis_typed(
        &mut self,
        input: Value,
        indices: Value,
        result: &TensorType,
        axis: usize,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(
            TakeAlongAxisOp,
            &mut self.ctx,
            vec![ty],
            vec![input, indices]
        );
        op.set_attr_take_along_axis(&self.ctx, AxisAttr::new(axis));
        self.push(op)
    }

    fn gather_gradient_typed(
        &mut self,
        gradient: Value,
        indices: Value,
        result: &TensorType,
        axis: usize,
        batched: bool,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(
            GatherGradientOp,
            &mut self.ctx,
            vec![ty],
            vec![gradient, indices]
        );
        op.set_attr_gather_gradient(&self.ctx, GatherGradientAttr::new(axis, batched));
        self.push(op)
    }

    fn concatenate_typed(
        &mut self,
        operands: Vec<Value>,
        result: &TensorType,
        axis: usize,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(ConcatenateOp, &mut self.ctx, vec![ty], operands);
        op.set_attr_concatenate_axis(&self.ctx, AxisAttr::new(axis));
        self.push(op)
    }

    fn compare_mask_typed(
        &mut self,
        lhs: Value,
        rhs: Value,
        result: &TensorType,
        comparison: Comparison,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(CompareMaskOp, &mut self.ctx, vec![ty], vec![lhs, rhs]);
        op.set_attr_comparison(&self.ctx, ComparisonAttr::new(comparison));
        self.push(op)
    }

    fn is_finite_mask_typed(&mut self, input: Value, result: &TensorType) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(IsFiniteMaskOp, &mut self.ctx, vec![ty], vec![input]);
        self.push(op)
    }

    fn select_typed(
        &mut self,
        mask: Value,
        on_true: Value,
        on_false: Value,
        result: &TensorType,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(
            SelectOp,
            &mut self.ctx,
            vec![ty],
            vec![mask, on_true, on_false]
        );
        self.push(op)
    }

    fn reduce_sum_typed(&mut self, input: Value, result: &TensorType, axes: Vec<usize>) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(ReduceSumOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_reduction_axes(&self.ctx, AxesAttr::new(&axes));
        self.push(op)
    }

    fn reduce_maximum_typed(
        &mut self,
        input: Value,
        result: &TensorType,
        axes: Vec<usize>,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(ReduceMaximumOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_maximum_axes(&self.ctx, AxesAttr::new(&axes));
        self.push(op)
    }

    fn argmax_typed(&mut self, input: Value, result: &TensorType, axis: usize) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(ArgMaxOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_argmax_axis(&self.ctx, AxisAttr::new(axis));
        self.push(op)
    }

    fn sorted_indices_typed(
        &mut self,
        input: Value,
        result: &TensorType,
        axis: usize,
        descending: bool,
    ) -> Value {
        let ty = self.tensor_type_for(result);
        let op = construct_op!(SortedIndicesOp, &mut self.ctx, vec![ty], vec![input]);
        op.set_attr_sort(&self.ctx, SortAttr::new(axis, descending));
        self.push(op)
    }
}
