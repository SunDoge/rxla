use super::*;

fn axes(rank: usize, contracting: &[usize], batch: &[usize]) -> Result<Vec<usize>> {
    let mut used = vec![false; rank];
    for &axis in contracting.iter().chain(batch) {
        if axis >= rank || used[axis] {
            return Err(err(
                "contraction axes must be in range, unique and disjoint",
            ));
        }
        used[axis] = true;
    }
    Ok((0..rank).filter(|&axis| !used[axis]).collect())
}

fn product(shape: &[i64], axes: &[usize]) -> Result<i64> {
    axes.iter().try_fold(1i64, |n, &axis| {
        n.checked_mul(shape[axis])
            .ok_or_else(|| err("flattened contraction dimension overflows i64"))
    })
}

impl Tensor {
    /// Two-operand einsum with an explicit output, e.g. `bmk,kbn->bmn`.
    /// Labels are case-sensitive ASCII letters; ASCII whitespace is ignored.
    /// Repeated input labels select diagonals, omitted labels are summed, and
    /// output labels must be unique and present in an input. Scalars use empty
    /// label lists. Equal labels require equal dimensions (no broadcasting).
    /// Ellipses, implicit outputs and more than two operands are unsupported.
    /// Composes existing diagonal/reduction/dot_general operations with their
    /// precision and AD contracts; this is not a contraction-order optimizer.
    pub fn einsum(&self, equation: &str, rhs: &Self) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &rhs.graph().0) {
            return Err(err("cross-graph einsum operands"));
        }
        let equation: String = equation
            .chars()
            .filter(|c| !c.is_ascii_whitespace())
            .collect();
        let (inputs, output) = equation
            .split_once("->")
            .ok_or_else(|| err("einsum requires explicit -> output"))?;
        let (left, right) = inputs
            .split_once(',')
            .ok_or_else(|| err("einsum requires two operands"))?;
        if !left
            .bytes()
            .chain(right.bytes())
            .chain(output.bytes())
            .all(|c| c.is_ascii_alphabetic())
        {
            return Err(err("einsum labels must be ASCII letters"));
        }
        if left.len() != self.shape.len() || right.len() != rhs.shape.len() {
            return Err(err("einsum label count must match operand rank"));
        }
        let mut dimensions = [None; 128];
        for (labels, shape) in [(left, self.shape()), (right, rhs.shape())] {
            for (label, &size) in labels.bytes().zip(shape) {
                if dimensions[label as usize].is_some_and(|old| old != size) {
                    return Err(err("einsum dimensions for equal labels must match"));
                }
                dimensions[label as usize] = Some(size);
            }
        }
        let mut seen = [false; 128];
        for label in output.bytes() {
            if dimensions[label as usize].is_none()
                || std::mem::replace(&mut seen[label as usize], true)
            {
                return Err(err(
                    "einsum output labels must be unique and present in inputs",
                ));
            }
        }
        // All syntax/rank/label checks precede graph mutations.
        // An empty summation has no products, even when another operand is
        // nonfinite. Reducing it first and multiplying 0*Inf would be wrong.
        if dimensions
            .iter()
            .enumerate()
            .any(|(label, size)| *size == Some(0) && !seen[label])
        {
            let shape: Vec<_> = output
                .bytes()
                .map(|label| dimensions[label as usize].unwrap())
                .collect();
            return self.graph().constant(&[], &[0.])?.broadcast_to(&shape);
        }
        let (left, ll) =
            einsum_operand(self, left.as_bytes(), right.as_bytes(), output.as_bytes())?;
        let (right, rl) = einsum_operand(rhs, right.as_bytes(), &ll, output.as_bytes())?;
        let mut lc = Vec::new();
        let mut rc = Vec::new();
        let mut lb = Vec::new();
        let mut rb = Vec::new();
        let mut result_labels = Vec::new();
        for (i, label) in ll.iter().enumerate() {
            if let Some(j) = rl.iter().position(|r| r == label) {
                if output.as_bytes().contains(label) {
                    lb.push(i);
                    rb.push(j);
                    result_labels.push(*label);
                } else {
                    lc.push(i);
                    rc.push(j);
                }
            }
        }
        result_labels.extend(ll.iter().filter(|l| !rl.contains(l)).copied());
        result_labels.extend(rl.iter().filter(|r| !ll.contains(r)).copied());
        let permutation = output
            .bytes()
            .map(|label| {
                result_labels
                    .iter()
                    .position(|&l| l == label)
                    .ok_or_else(|| err("einsum output label lost during lowering"))
            })
            .collect::<Result<Vec<_>>>()?;
        left.dot_general(&right, &lc, &rc, &lb, &rb)?
            .transpose(&permutation)
    }

    /// Embed the last dimension as a diagonal in two appended matrix axes.
    /// `[..., N]` becomes `[..., M, M]`, where `M = N + abs(offset)`.
    /// Positive offsets lie above the main diagonal, negative offsets below.
    /// All other entries are zero, including when diagonal values are NaN/Inf.
    /// Scalars and overflowing shapes fail; empty vectors are supported.
    /// Uses padding/reshape/slice, retaining their higher-order AD rules.
    pub fn diag_embed(&self, offset: i64) -> Result<Self> {
        let rank = self.shape.len();
        let &n = self
            .shape
            .last()
            .ok_or_else(|| err("diag_embed requires a vector axis"))?;
        let shift = offset
            .checked_abs()
            .ok_or_else(|| err("diagonal offset overflow"))?;
        let side = n
            .checked_add(shift)
            .ok_or_else(|| err("diagonal dimension overflow"))?;
        let padded = n
            .checked_add(1)
            .and_then(|v| n.checked_mul(v))
            .ok_or_else(|| err("diagonal staging dimension overflow"))?;
        let mut output_shape = self.shape[..rank - 1].to_vec();
        output_shape.extend([side, side]);
        elements(&output_shape)?;
        let mut padding = vec![[0, 0]; rank + 1];
        padding[rank] = [0, n];
        let mut flat_shape = self.shape.to_vec();
        flat_shape[rank - 1] = padded;
        elements(&flat_shape)?;
        let mut limits = flat_shape.clone();
        limits[rank - 1] = n * n; // bounded by the checked n*(n+1)
        let mut square_shape = self.shape[..rank - 1].to_vec();
        square_shape.extend([n, n]);
        let square = self
            .unsqueeze(rank)?
            .pad(&padding, 0.)?
            .reshape(&flat_shape)?
            .slice(&vec![0; rank], &limits, &vec![1; rank])?
            .reshape(&square_shape)?;
        padding[rank] = if offset >= 0 { [shift, 0] } else { [0, shift] };
        padding[rank - 1] = if offset >= 0 { [0, shift] } else { [shift, 0] };
        square.pad(&padding, 0.)
    }

    /// Extract a diagonal across two distinct axes, appending its dimension
    /// after the remaining axes in their original order. Positive offsets move
    /// along axis2, negative offsets along axis1. Out-of-range offsets return
    /// an empty diagonal. This is a symbolic operation, not an aliased view.
    /// Composes transpose/reshape/strided slice and supports their AD rules.
    pub fn diagonal(&self, offset: i64, axis1: usize, axis2: usize) -> Result<Self> {
        let rank = self.shape.len();
        if axis1 >= rank || axis2 >= rank || axis1 == axis2 {
            return Err(err("diagonal requires two distinct valid axes"));
        }
        let rows = self.shape[axis1];
        let cols = self.shape[axis2];
        let row = if offset < 0 {
            offset.unsigned_abs().min(rows as u64) as i64
        } else {
            0
        };
        let col = if offset > 0 { offset.min(cols) } else { 0 };
        let len = (rows - row).min(cols - col);
        let flat = rows
            .checked_mul(cols)
            .ok_or_else(|| err("diagonal dimension overflow"))?;
        let mut order: Vec<_> = (0..rank).filter(|&a| a != axis1 && a != axis2).collect();
        let mut shape: Vec<_> = order.iter().map(|&a| self.shape[a]).collect();
        order.extend([axis1, axis2]);
        shape.push(flat);
        let mut starts = vec![0; rank - 1];
        let mut limits = shape.clone();
        let mut strides = vec![1; rank - 1];
        // For empty diagonals, use a valid empty slice even when offset is MIN.
        let start = if len == 0 { 0 } else { row * cols + col };
        let stride = if len <= 1 { 1 } else { cols + 1 };
        starts[rank - 2] = start;
        limits[rank - 2] = if len == 0 {
            0
        } else {
            start + (len - 1) * stride + 1
        };
        strides[rank - 2] = stride;
        self.transpose(&order)?
            .reshape(&shape)?
            .slice(&starts, &limits, &strides)
    }

    /// Sum an offset diagonal, removing axis1 and axis2 and preserving all
    /// other axes. An empty diagonal sums to zero.
    pub fn trace(&self, offset: i64, axis1: usize, axis2: usize) -> Result<Self> {
        let diagonal = self.diagonal(offset, axis1, axis2)?;
        diagonal.sum(&[diagonal.shape.len() - 1], false)
    }

    /// General F32 contraction with explicit paired contraction and batch axes.
    /// Each paired dimension must match exactly; no implicit broadcasting.
    /// Output order is the listed batch axes, remaining lhs axes in original
    /// order, then remaining rhs axes in original order. Axis lists must be
    /// unique and batch/contracting axes disjoint on each operand.
    ///
    /// Implemented with transpose/reshape/batched matmul, reusing its precision
    /// and higher-order AD rules. These are symbolic graph operations, not host
    /// copies; final layout and fusion decisions belong to the backend.
    pub fn dot_general(
        &self,
        rhs: &Self,
        lhs_contract: &[usize],
        rhs_contract: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &rhs.graph().0) {
            return Err(err("cross-graph contraction operands"));
        }
        if lhs_contract.len() != rhs_contract.len() || lhs_batch.len() != rhs_batch.len() {
            return Err(err("contraction axis lists must pair one-to-one"));
        }
        let lhs_free = axes(self.shape.len(), lhs_contract, lhs_batch)?;
        let rhs_free = axes(rhs.shape.len(), rhs_contract, rhs_batch)?;
        for (&l, &r) in lhs_contract
            .iter()
            .zip(rhs_contract)
            .chain(lhs_batch.iter().zip(rhs_batch))
        {
            if self.shape[l] != rhs.shape[r] {
                return Err(err("paired contraction/batch dimensions differ"));
            }
        }
        let m = product(self.shape(), &lhs_free)?;
        let k = product(self.shape(), lhs_contract)?;
        let n = product(rhs.shape(), &rhs_free)?;
        let batch: Vec<_> = lhs_batch.iter().map(|&a| self.shape[a]).collect();
        let mut left_shape = batch.clone();
        left_shape.extend([m, k]);
        let mut right_shape = batch.clone();
        right_shape.extend([k, n]);
        let left_order: Vec<_> = lhs_batch
            .iter()
            .chain(&lhs_free)
            .chain(lhs_contract)
            .copied()
            .collect();
        let right_order: Vec<_> = rhs_batch
            .iter()
            .chain(rhs_contract)
            .chain(&rhs_free)
            .copied()
            .collect();
        let mut output = batch;
        output.extend(lhs_free.iter().map(|&a| self.shape[a]));
        output.extend(rhs_free.iter().map(|&a| rhs.shape[a]));
        self.transpose(&left_order)?
            .reshape(&left_shape)?
            .matmul(&rhs.transpose(&right_order)?.reshape(&right_shape)?)?
            .reshape(&output)
    }

    /// Contract paired axes without batch axes; empty lists form an outer product.
    /// Remaining lhs axes precede remaining rhs axes. See dot_general.
    pub fn tensordot(&self, rhs: &Self, lhs_axes: &[usize], rhs_axes: &[usize]) -> Result<Self> {
        self.dot_general(rhs, lhs_axes, rhs_axes, &[], &[])
    }
}

fn einsum_operand(
    input: &Tensor,
    labels: &[u8],
    other: &[u8],
    output: &[u8],
) -> Result<(Tensor, Vec<u8>)> {
    let mut value = input.clone();
    let mut labels = labels.to_vec();
    loop {
        let pair = labels.iter().enumerate().find_map(|(i, label)| {
            labels[i + 1..]
                .iter()
                .position(|v| v == label)
                .map(|j| (i, i + 1 + j))
        });
        let Some((a, b)) = pair else { break };
        let label = labels[a];
        value = value.diagonal(0, a, b)?;
        labels.remove(b);
        labels.remove(a);
        labels.push(label);
    }
    let reduce: Vec<_> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, label)| (!other.contains(label) && !output.contains(label)).then_some(i))
        .collect();
    if !reduce.is_empty() {
        value = value.sum(&reduce, false)?;
        labels.retain(|label| other.contains(label) || output.contains(label));
    }
    Ok((value, labels))
}
