use super::*;
use crate::{Mesh, PartitionSpec};

fn vector_type(size: i64, dtype: DType) -> TensorType {
    TensorType {
        dims: vec![size],
        dtype,
    }
}

#[test]
fn semantic_ssa_lowers_to_verified_stablehlo() {
    let mut program = ProgramIr::default();
    let ty = vector_type(4, DType::F32);
    let lhs = program.append(&Op::Parameter(0), &[], &ty).unwrap();
    let rhs = program.append(&Op::Parameter(1), &[], &ty).unwrap();
    let sum = program
        .append(&Op::Binary(Binary::Add), &[lhs, rhs], &ty)
        .unwrap();
    let exponential = program.append(&Op::Unary(Unary::Exp), &[sum], &ty).unwrap();

    let lowered = program.stablehlo_program(&[exponential], true).unwrap();
    assert_eq!(lowered.parameters, [0, 1]);
    assert_eq!(lowered.inputs, [ty.clone(), ty.clone()]);
    assert_eq!(lowered.outputs, [ty]);
    assert!(lowered.code.contains("stablehlo.add"));
    assert!(lowered.code.contains("stablehlo.exponential"));
}

#[test]
fn invalid_ssa_identity_keeps_a_typed_error() {
    let program = ProgramIr::default();
    assert!(matches!(
        program.value_type(SsaId::from_index(0)),
        Err(IrError::InvalidValue {
            operation: "resolving a tensor ID"
        })
    ));
}

#[test]
fn parameter_abi_must_be_dense_and_unique() {
    let mut program = ProgramIr::default();
    let ty = vector_type(2, DType::F32);
    program.append(&Op::Parameter(0), &[], &ty).unwrap();
    program.append(&Op::Parameter(2), &[], &ty).unwrap();

    assert!(matches!(
        program.parameter_count(),
        Err(IrError::MalformedAttribute {
            attribute: "dense parameter ABI numbering"
        })
    ));
}

#[test]
fn planning_facts_retain_explicit_sharding() {
    let mut program = ProgramIr::default();
    let ty = vector_type(8, DType::F32);
    let input = program.append(&Op::Parameter(0), &[], &ty).unwrap();
    let mesh = Mesh::new([("data", 2)]).unwrap();
    let sharding = Sharding::partitioned(mesh, PartitionSpec::new([Some("data")]).unwrap());
    program.set_sharding(input, &sharding).unwrap();

    let facts = program.planning_snapshot(&[input]).unwrap();
    assert_eq!(facts.input_count(), 1);
    assert_eq!(facts.output_count(), 1);
    assert_eq!(facts.value_count(), 1);
    assert_eq!(facts.sharding_constraints().len(), 1);
    assert_eq!(facts.sharding_constraints()[0].sharding, sharding);
}

#[test]
fn semantic_snapshot_can_be_lowered_more_than_once() {
    let mut program = ProgramIr::default();
    let ty = vector_type(2, DType::F32);
    let input = program.append(&Op::Parameter(0), &[], &ty).unwrap();
    let snapshot = SemanticProgram::capture(&program, &[input], true).unwrap();

    let first = snapshot.lower(LoweringTarget::Portable).unwrap();
    let second = snapshot.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(first.code, second.code);
    assert_eq!(first.inputs, second.inputs);
    assert_eq!(first.outputs, second.outputs);
}

#[test]
fn state_effects_survive_semantic_snapshots_and_discharge_to_pure_stablehlo() {
    let mut program = ProgramIr::default();
    let ty = vector_type(2, DType::F32);
    let state = program.state_input(0, 7, "decoder.cache", &ty).unwrap();
    let read = program.state_read(state, 7).unwrap();
    let next = program
        .append(&Op::Unary(Unary::Exp), &[read], &ty)
        .unwrap();
    let written = program.state_write(next, 7).unwrap();

    let nodes = program.semantic_nodes().unwrap();
    assert!(matches!(
        &nodes[state.index()].op,
        Op::StateInput {
            number: 0,
            state_id: 7,
            path,
        } if path == "decoder.cache"
    ));
    assert!(matches!(
        nodes[read.index()].op,
        Op::StateRead { state_id: 7 }
    ));
    assert!(matches!(
        nodes[written.index()].op,
        Op::StateWrite { state_id: 7 }
    ));

    let snapshot = SemanticProgram::capture(&program, &[written], true).unwrap();
    let lowered = snapshot.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(lowered.parameters, [0]);
    assert!(!lowered.code.contains("rxla.state_"));
    assert!(lowered.code.contains("stablehlo.exponential"));
}
fn conditional(graph: &mut IrGraph, branch_add: bool) -> Value {
    let predicate = graph.parameter_number(0, &[], DType::I32);
    let then_value = graph.parameter_number(1, &[2], DType::F32);
    let else_value = graph.parameter_number(2, &[2], DType::F32);
    let result_type = graph.tensor_type(&[2], DType::F32);
    let branch_values = [then_value, else_value].map(|value| {
        if !branch_add {
            return (value, None);
        }
        let ty = graph.tensor_type(&[2], DType::F32);
        let add = <AddOp as PlironOp>::from_operation(Operation::new(
            &mut graph.ctx,
            AddOp::get_concrete_op_info(),
            vec![ty],
            vec![value, value],
            vec![],
            0,
        ));
        (add.get_result(&graph.ctx), Some(add))
    });
    let conditional = <IfOp as PlironOp>::from_operation(Operation::new(
        &mut graph.ctx,
        IfOp::get_concrete_op_info(),
        vec![result_type],
        vec![predicate],
        vec![],
        2,
    ));
    for (region_index, (branch_value, add)) in branch_values.into_iter().enumerate() {
        let block = BasicBlock::new(&mut graph.ctx, None, vec![]);
        block.insert_at_back(
            conditional
                .get_operation()
                .deref(&graph.ctx)
                .get_region(region_index),
            &graph.ctx,
        );
        if let Some(add) = add {
            add.get_operation().insert_at_back(block, &graph.ctx);
        }
        let yield_op = <YieldOp as PlironOp>::from_operation(Operation::new(
            &mut graph.ctx,
            YieldOp::get_concrete_op_info(),
            vec![],
            vec![branch_value],
            vec![],
            0,
        ));
        yield_op.get_operation().insert_at_back(block, &graph.ctx);
    }
    graph.push_results(conditional)[0]
}

fn nested_conditional(graph: &mut IrGraph) -> Value {
    let predicate = graph.parameter_number(0, &[], DType::I32);
    let then_value = graph.parameter_number(1, &[2], DType::F32);
    let else_value = graph.parameter_number(2, &[2], DType::F32);
    let result_type = graph.tensor_type(&[2], DType::F32);
    let inner = <IfOp as PlironOp>::from_operation(Operation::new(
        &mut graph.ctx,
        IfOp::get_concrete_op_info(),
        vec![result_type],
        vec![predicate],
        vec![],
        2,
    ));
    for (region_index, value) in [then_value, else_value].into_iter().enumerate() {
        let block = BasicBlock::new(&mut graph.ctx, None, vec![]);
        block.insert_at_back(
            inner
                .get_operation()
                .deref(&graph.ctx)
                .get_region(region_index),
            &graph.ctx,
        );
        let yield_op = <YieldOp as PlironOp>::from_operation(Operation::new(
            &mut graph.ctx,
            YieldOp::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        ));
        yield_op.get_operation().insert_at_back(block, &graph.ctx);
    }
    let inner_result = inner.get_operation().deref(&graph.ctx).get_result(0);
    let outer = <IfOp as PlironOp>::from_operation(Operation::new(
        &mut graph.ctx,
        IfOp::get_concrete_op_info(),
        vec![result_type],
        vec![predicate],
        vec![],
        2,
    ));
    let then_block = BasicBlock::new(&mut graph.ctx, None, vec![]);
    then_block.insert_at_back(
        outer.get_operation().deref(&graph.ctx).get_region(0),
        &graph.ctx,
    );
    inner.get_operation().insert_at_back(then_block, &graph.ctx);
    let then_yield = <YieldOp as PlironOp>::from_operation(Operation::new(
        &mut graph.ctx,
        YieldOp::get_concrete_op_info(),
        vec![],
        vec![inner_result],
        vec![],
        0,
    ));
    then_yield
        .get_operation()
        .insert_at_back(then_block, &graph.ctx);
    let else_block = BasicBlock::new(&mut graph.ctx, None, vec![]);
    else_block.insert_at_back(
        outer.get_operation().deref(&graph.ctx).get_region(1),
        &graph.ctx,
    );
    let else_yield = <YieldOp as PlironOp>::from_operation(Operation::new(
        &mut graph.ctx,
        YieldOp::get_concrete_op_info(),
        vec![],
        vec![else_value],
        vec![],
        0,
    ));
    else_yield
        .get_operation()
        .insert_at_back(else_block, &graph.ctx);
    graph.push_results(outer)[0]
}

#[test]
fn structured_conditionals_verify_with_explicit_branch_dependencies() {
    let mut graph = IrGraph::default();
    let output = conditional(&mut graph, false);
    assert_eq!(graph.value_type(output).unwrap().dims, [2]);
    graph.verify("structured conditional").unwrap();
    let stablehlo = graph.export_stablehlo(&[output], false).unwrap();
    assert!(stablehlo.contains("stablehlo.if"));
    assert!(stablehlo.contains("stablehlo.return %arg1 : tensor<2xf32>"));
    assert!(stablehlo.contains("stablehlo.return %arg2 : tensor<2xf32>"));
}

#[test]
fn conditional_regions_emit_branch_computation_before_return() {
    let mut graph = IrGraph::default();
    let output = conditional(&mut graph, true);
    let stablehlo = graph.export_stablehlo(&[output], false).unwrap();
    assert_eq!(stablehlo.matches("stablehlo.add").count(), 2);
    assert!(stablehlo.contains("stablehlo.return %v"));
}

#[test]
fn nested_conditionals_retain_captures_and_emit_recursively() {
    let mut graph = IrGraph::default();
    let output = nested_conditional(&mut graph);
    let stablehlo = graph.export_stablehlo(&[output], false).unwrap();
    assert_eq!(stablehlo.matches("stablehlo.if").count(), 2);
    assert!(stablehlo.contains("%arg1: tensor<2xf32>"));
    assert!(stablehlo.contains("%arg2: tensor<2xf32>"));
}

#[test]
fn stablehlo_arguments_follow_abi_numbers_not_block_order() {
    let mut program = ProgramIr::default();
    let first = program
        .append(
            &Op::Parameter(1),
            &[],
            &TensorType {
                dims: vec![1],
                dtype: DType::F32,
            },
        )
        .unwrap();
    let second = program
        .append(
            &Op::Parameter(0),
            &[],
            &TensorType {
                dims: vec![2],
                dtype: DType::F32,
            },
        )
        .unwrap();

    let lowered = program.stablehlo_program(&[first, second], true).unwrap();
    assert_eq!(lowered.parameters, [0, 1]);
    assert_eq!(lowered.inputs[0].dims, [2]);
    assert_eq!(lowered.inputs[1].dims, [1]);
    assert!(lowered.code.contains("%arg0: tensor<2xf32>"));
    assert!(lowered.code.contains("%arg1: tensor<1xf32>"));
    assert!(lowered.code.contains("return %arg1, %arg0"));
}

#[test]
fn direct_builder_distinguishes_invalid_operands_from_unsupported_ops() {
    let mut direct = ProgramIr::default();
    let ty = TensorType {
        dims: vec![1],
        dtype: DType::F32,
    };
    assert!(
        direct
            .append(&Op::Reshape, &[SsaId::from_index(0)], &ty)
            .is_err()
    );
    assert_eq!(
        direct.append(&Op::Parameter(0), &[], &ty).unwrap(),
        SsaId::from_index(0)
    );
    assert_eq!(
        direct.append(&Op::Parameter(1), &[], &ty).unwrap(),
        SsaId::from_index(1)
    );
    assert_eq!(
        direct
            .append(
                &Op::WithGradient,
                &[SsaId::from_index(0), SsaId::from_index(1)],
                &ty,
            )
            .unwrap(),
        SsaId::from_index(2)
    );
    assert_eq!(
        direct
            .append(&Op::Reshape, &[SsaId::from_index(0)], &ty)
            .unwrap(),
        SsaId::from_index(3)
    );
}

#[test]
fn pliron_interfaces_reject_mismatched_add_types() {
    let mut graph = IrGraph::default();
    let lhs = graph.parameter(&[2]);
    let rhs = graph.parameter(&[3]);
    let result = graph.tensor_type(&[2], DType::F32);
    let add = AddOp::from_operation(Operation::new(
        &mut graph.ctx,
        AddOp::get_concrete_op_info(),
        vec![result],
        vec![lhs, rhs],
        vec![],
        0,
    ));
    assert!(verify_op(&add, &graph.ctx).is_err());
}

#[test]
fn leaf_ops_verify_their_abi_and_payload_invariants() {
    let mut graph = IrGraph::default();
    let ty = graph.tensor_type(&[2], DType::F32);
    let parameter = ParameterOp::from_operation(Operation::new(
        &mut graph.ctx,
        ParameterOp::get_concrete_op_info(),
        vec![ty],
        vec![],
        vec![],
        0,
    ));
    assert!(verify_op(&parameter, &graph.ctx).is_err());
    parameter.set_attr_number(&graph.ctx, StringAttr::new("not-an-index".into()));
    assert!(verify_op(&parameter, &graph.ctx).is_err());

    let ty = graph.tensor_type(&[2], DType::F32);
    let constant = ConstantOp::from_operation(Operation::new(
        &mut graph.ctx,
        ConstantOp::get_concrete_op_info(),
        vec![ty],
        vec![],
        vec![],
        0,
    ));
    constant.set_attr_value(&graph.ctx, BytesAttr::new(vec![0; 4]));
    assert!(verify_op(&constant, &graph.ctx).is_err());
    constant.set_attr_value(&graph.ctx, BytesAttr::new(vec![0; 8]));
    assert!(verify_op(&constant, &graph.ctx).is_ok());
}

#[test]
fn binary_attributes_reject_malformed_encodings_before_access() {
    let ctx = Context::new();
    assert!(
        ShapeAttr {
            bytes: BytesAttr::new(vec![0; 7])
        }
        .verify(&ctx)
        .is_err()
    );
    assert!(
        ComparisonAttr {
            bytes: BytesAttr::new(vec![6])
        }
        .verify(&ctx)
        .is_err()
    );
    assert!(
        GatherGradientAttr {
            bytes: BytesAttr::new(vec![2; 9])
        }
        .verify(&ctx)
        .is_err()
    );
    assert!(
        ShardingAttr {
            bytes: BytesAttr::new(vec![0xff])
        }
        .verify(&ctx)
        .is_err()
    );
}

#[test]
fn operation_verifiers_reject_non_tensor_types_without_panicking() {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = Context::new();
    let integer = IntegerType::get(&ctx, 32, Signedness::Signed).into();
    let iota = IotaOp::from_operation(Operation::new(
        &mut ctx,
        IotaOp::get_concrete_op_info(),
        vec![integer],
        vec![],
        vec![],
        0,
    ));
    iota.set_attr_iota_axis(&ctx, AxisAttr::new(0));

    assert!(verify_op(&iota, &ctx).is_err());
}

#[test]
fn program_type_queries_reject_non_tensor_values_without_panicking() {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut program = ProgramIr::default();
    let integer = IntegerType::get(&program.graph.ctx, 32, Signedness::Signed).into();
    let iota = IotaOp::from_operation(Operation::new(
        &mut program.graph.ctx,
        IotaOp::get_concrete_op_info(),
        vec![integer],
        vec![],
        vec![],
        0,
    ));
    iota.set_attr_iota_axis(&program.graph.ctx, AxisAttr::new(0));
    let value = iota.get_result(&program.graph.ctx);
    program
        .graph
        .module
        .append_operation(&mut program.graph.ctx, iota.get_operation(), 0);
    let id = program.values.push(value);

    assert!(matches!(
        program.graph.value_type(value),
        Err(IrError::ExpectedTensorType {
            operation: "reading an IR value type"
        })
    ));
    assert!(matches!(
        program.value_type(id),
        Err(IrError::ExpectedTensorType {
            operation: "reading an SSA value type"
        })
    ));
}

#[test]
fn invalid_source_ir_is_reported_instead_of_panicking() {
    let mut program = ProgramIr::default();
    let lhs = program.graph.parameter(&[2]);
    let rhs = program.graph.parameter(&[3]);
    let result_type = program.graph.tensor_type(&[2], DType::F32);
    let add = AddOp::from_operation(Operation::new(
        &mut program.graph.ctx,
        AddOp::get_concrete_op_info(),
        vec![result_type],
        vec![lhs, rhs],
        vec![],
        0,
    ));
    let result = add.get_result(&program.graph.ctx);
    program
        .graph
        .module
        .append_operation(&mut program.graph.ctx, add.get_operation(), 0);
    program.values.push(lhs);
    program.values.push(rhs);
    program.values.push(result);

    assert!(matches!(
        program.semantic_nodes().unwrap_err(),
        IrError::Verification {
            stage: "source Pliron",
            ..
        }
    ));
    assert!(matches!(
        program.stablehlo(&[SsaId::from_index(2)]).unwrap_err(),
        IrError::Verification {
            stage: "source Pliron",
            ..
        }
    ));
    assert!(matches!(
        program.stablehlo_program(&[SsaId::from_index(2)], true),
        Err(IrError::Verification {
            stage: "source Pliron",
            ..
        })
    ));
    assert!(matches!(
        program.planning_snapshot(&[SsaId::from_index(2)]),
        Err(IrError::Verification {
            stage: "source Pliron planning",
            ..
        })
    ));
}

#[test]
fn construction_defers_whole_ir_verification_to_analysis_boundaries() {
    let mut graph = IrGraph::default();
    let lhs = graph.parameter(&[2]);
    let rhs = graph.parameter(&[3]);
    let malformed = graph.add_typed(
        lhs,
        rhs,
        &TensorType {
            dims: vec![2],
            dtype: DType::F32,
        },
    );

    assert!(matches!(
        graph.export_stablehlo(&[malformed], false),
        Err(IrError::Verification {
            stage: "source Pliron",
            ..
        })
    ));
}

#[test]
fn shape_operation_verifiers_reject_invalid_transformed_ir() {
    let mut graph = IrGraph::default();
    let input = graph.parameter(&[2, 3]);

    let result = graph.tensor_type(&[2, 4], DType::F32);
    let broadcast = BroadcastOp::from_operation(Operation::new(
        &mut graph.ctx,
        BroadcastOp::get_concrete_op_info(),
        vec![result],
        vec![input],
        vec![],
        0,
    ));
    broadcast.set_attr_broadcast_axes(&graph.ctx, AxesAttr::new(&[0, 1]));
    assert!(verify_op(&broadcast, &graph.ctx).is_err());

    let result = graph.tensor_type(&[3, 2], DType::F32);
    let transpose = TransposeOp::from_operation(Operation::new(
        &mut graph.ctx,
        TransposeOp::get_concrete_op_info(),
        vec![result],
        vec![input],
        vec![],
        0,
    ));
    transpose.set_attr_permutation(&graph.ctx, AxesAttr::new(&[0, 1]));
    assert!(verify_op(&transpose, &graph.ctx).is_err());

    let result = graph.tensor_type(&[5], DType::F32);
    let reshape = ReshapeOp::from_operation(Operation::new(
        &mut graph.ctx,
        ReshapeOp::get_concrete_op_info(),
        vec![result],
        vec![input],
        vec![],
        0,
    ));
    assert!(verify_op(&reshape, &graph.ctx).is_err());

    let result = graph.tensor_type(&[2, 3], DType::F32);
    let iota = IotaOp::from_operation(Operation::new(
        &mut graph.ctx,
        IotaOp::get_concrete_op_info(),
        vec![result],
        vec![],
        vec![],
        0,
    ));
    iota.set_attr_iota_axis(&graph.ctx, AxisAttr::new(0));
    assert!(verify_op(&iota, &graph.ctx).is_err());
}

#[test]
fn tensor_ir_preserves_runtime_dtypes() {
    let mut graph = IrGraph::default();
    let index = graph.constant_i32(&[], vec![7]);
    let module = graph.export_stablehlo(&[index], false).unwrap();
    assert!(module.contains("tensor<i32>"));
}

#[test]
fn tensor_type_attributes_preserve_open_dtype_and_shape_values() {
    let dtype = DType::from_raw(0xfeed);
    assert_eq!(ElementTypeAttr::new(dtype).value(), dtype);
    assert_eq!(ShapeAttr::new(&[-1, 0, 7]).values(), [-1, 0, 7]);
}

#[test]
fn tensor_type_verifier_accepts_dynamic_dims_and_rejects_invalid_types() {
    let ctx = Context::new();
    let dynamic = RankedTensorType {
        shape: ShapeAttr::new(&[-1, 0, 7]),
        element: ElementTypeAttr::new(DType::F32),
    };
    assert!(dynamic.verify(&ctx).is_ok());

    let invalid_shape = RankedTensorType {
        shape: ShapeAttr::new(&[-2, 7]),
        element: ElementTypeAttr::new(DType::F32),
    };
    assert!(invalid_shape.verify(&ctx).is_err());

    let invalid_dtype = RankedTensorType {
        shape: ShapeAttr::new(&[7]),
        element: ElementTypeAttr::new(DType::from_raw(u32::MAX)),
    };
    assert!(invalid_dtype.verify(&ctx).is_err());
}
