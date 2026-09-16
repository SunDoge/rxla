use crate::{Result, Tensor, elements, err};

fn flattened_shape(shape: &[i64], start: usize, end: usize) -> Result<Vec<i64>> {
    if shape.is_empty() && start == 0 && end == 0 {
        return Ok(vec![1]);
    }
    if start > end || end >= shape.len() {
        return Err(err(
            "flatten requires an ordered inclusive axis range within rank",
        ));
    }
    let selected = &shape[start..=end];
    let size = if selected.contains(&0) {
        0
    } else {
        selected
            .iter()
            .try_fold(1i64, |n, &d| n.checked_mul(d))
            .ok_or_else(|| err("flattened dimension exceeds i64"))?
    };
    let mut result = Vec::with_capacity(shape.len() - (end - start));
    result.extend_from_slice(&shape[..start]);
    result.push(size);
    result.extend_from_slice(&shape[end + 1..]);
    Ok(result)
}

fn unflattened_shape(shape: &[i64], axis: usize, sizes: &[i64]) -> Result<Vec<i64>> {
    let dimension = shape
        .get(axis)
        .ok_or_else(|| err("unflatten axis out of range"))?;
    if sizes.is_empty() || elements(sizes)? != *dimension as usize {
        return Err(err(
            "unflatten sizes must be nonempty and multiply to the selected dimension",
        ));
    }
    let mut result = shape[..axis].to_vec();
    result.extend_from_slice(sizes);
    result.extend_from_slice(&shape[axis + 1..]);
    Ok(result)
}

macro_rules! reshape_axes {
    ($ty:ty) => {
        impl $ty {
            /// Move one axis to its final destination position, preserving the
            /// relative order of all other axes. Both axes must be within rank;
            /// scalars have no movable axis. Records a transpose, with no host
            /// transfer. Dtype and Tensor gradients are preserved.
            pub fn move_axis(&self, source: usize, destination: usize) -> Result<Self> {
                let rank = self.shape().len();
                if source >= rank || destination >= rank {
                    return Err(err("move_axis axis out of range"));
                }
                let mut permutation: Vec<_> = (0..rank).collect();
                permutation.remove(source);
                permutation.insert(destination, source);
                self.transpose(&permutation)
            }

            /// Exchange two axes, leaving all other axes in place. Both axes
            /// must be within rank, including when equal. Records a transpose
            /// and preserves dtype and Tensor gradients without a host transfer.
            pub fn swap_axes(&self, first: usize, second: usize) -> Result<Self> {
                let rank = self.shape().len();
                if first >= rank || second >= rank {
                    return Err(err("swap_axes axis out of range"));
                }
                let mut permutation: Vec<_> = (0..rank).collect();
                permutation.swap(first, second);
                self.transpose(&permutation)
            }

            /// Merge axes start..=end, preserving row-major element order and dtype.
            /// A scalar supports only `flatten(0, 0)`, producing shape `[1]`. Other
            /// ranges must be ordered and within rank; a flattened dimension
            /// must fit i64. Empty dimensions remain empty. Records a reshape,
            /// not a host copy or synchronization; Tensor gradients follow reshape.
            pub fn flatten(&self, start: usize, end: usize) -> Result<Self> {
                self.reshape(&flattened_shape(self.shape(), start, end)?)
            }

            /// Replace one axis with explicit sizes whose product equals that
            /// axis, preserving every other dimension, dtype and element order.
            /// Sizes must be nonempty and nonnegative; -1 inference is not used.
            /// Scalars have no axis to split. Zero sizes may split a zero axis.
            /// Uses reshape without host transfer; Tensor gradients are retained.
            pub fn unflatten(&self, axis: usize, sizes: &[i64]) -> Result<Self> {
                self.reshape(&unflattened_shape(self.shape(), axis, sizes)?)
            }
        }
    };
}
reshape_axes!(Tensor);
