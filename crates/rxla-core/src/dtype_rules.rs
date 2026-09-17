//! Runtime dtype validation for Pliron operation construction.
use crate::{DType, Result, err};
use rxla_ir::{Binary, Op};

pub(crate) fn infer(op: &Op, types: &[DType]) -> Result<DType> {
    let first = types.first().copied().unwrap_or(DType::F32);
    let all = |dtype| types.iter().all(|&t| t == dtype);
    let (valid, output) = match op {
        Op::ConstantF32(_) => (types.is_empty(), DType::F32),
        Op::ConstantI32(_) | Op::Iota { .. } => (types.is_empty(), DType::I32),
        Op::IndexToFloat => (types == [DType::I32], DType::F32),
        Op::Bf16ToFloat => (types == [DType::BF16], DType::F32),
        Op::Convert { dtype } => (
            types.len() == 1
                && matches!(first, DType::U8 | DType::F16 | DType::BF16 | DType::F32)
                && matches!(*dtype, DType::F16 | DType::BF16 | DType::F32),
            *dtype,
        ),
        Op::IntegerBinary(_) => (types == [DType::I32, DType::I32], DType::I32),
        Op::IndexLessEqualMask => (types == [DType::I32, DType::I32], DType::F32),
        Op::CompareMask(_) => (
            types.len() == 2 && all(first) && first != DType::BF16,
            DType::F32,
        ),
        Op::ArgMax { .. } | Op::SortedIndices { .. } => (types == [DType::F32], DType::I32),
        Op::Attention { .. } => (
            matches!(
                types,
                [DType::F32, DType::F32, DType::F32]
                    | [DType::F32, DType::F32, DType::F32, DType::F32]
            ),
            DType::F32,
        ),
        Op::Reshape
        | Op::StopGradient
        | Op::OptimizationBarrier
        | Op::Broadcast { .. }
        | Op::Transpose { .. }
        | Op::Reverse { .. }
        | Op::Slice(_) => (types.len() == 1, first),
        Op::Concatenate { .. } => (!types.is_empty() && all(first), first),
        Op::Take { .. } | Op::TakeAlongAxis { .. } => {
            (types.len() == 2 && types[1] == DType::I32, first)
        }
        Op::DynamicSlice => (
            !types.is_empty() && types[1..].iter().all(|&t| t == DType::I32),
            first,
        ),
        Op::DynamicUpdateSlice => (
            types.len() >= 2 && types[1] == first && types[2..].iter().all(|&t| t == DType::I32),
            first,
        ),
        Op::Select => (
            types.len() == 3 && types[0] == DType::F32 && types[1] == types[2],
            types.get(1).copied().unwrap_or(DType::F32),
        ),
        // Structured conditionals are built by the region-aware ProgramIr API;
        // their result dtype comes from matching branch results, not operands.
        Op::If { .. } | Op::MultiResult { .. } => (false, DType::F32),
        Op::CustomCall { result_dtype, .. } => (!types.is_empty(), *result_dtype),
        Op::Binary(Binary::Add | Binary::Sub | Binary::Mul | Binary::Maximum | Binary::Minimum) => {
            (
                types.len() == 2 && all(first) && first != DType::BF16,
                first,
            )
        }
        Op::GatherGradient { .. } => (types == [DType::F32, DType::I32], DType::F32),
        // Current NN kernels/reducers and derivatives use F32 constants.
        // Reject unsupported dtypes instead of inserting implicit conversions.
        _ => (all(DType::F32), DType::F32),
    };
    if !valid {
        return Err(err(format!(
            "unsupported operand dtypes {types:?} for {op:?}"
        )));
    }
    Ok(output)
}
