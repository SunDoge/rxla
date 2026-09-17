//! Static metadata and runtime shape values for symbolic tensors.
use super::{DType, Error, Op, Result, Tensor, elements};

macro_rules! shape_queries {
    ($($ty:ty),+ $(,)?) => { $(
        impl $ty {
            /// Number of axes; a scalar has rank zero. Reads symbolic metadata
            /// only, without adding graph nodes or contacting a device.
            pub fn ndim(&self) -> usize {
                self.shape().len()
            }

            /// A statically known dimension size. Dynamic dimensions return
            /// `None`; an invalid axis also returns `None`.
            pub fn static_dim(&self, axis: usize) -> Option<usize> {
                self.shape()
                    .get(axis)
                    .and_then(|&size| usize::try_from(size).ok())
            }

            /// Compile-time upper bound for a dynamic axis. Static axes and
            /// invalid axes return `None`.
            pub fn dim_bound(&self, axis: usize) -> Option<usize> {
                self.dynamic_bounds
                    .get(axis)
                    .and_then(|&bound| usize::try_from(bound).ok())
            }

            /// Static element count, or `None` when any extent is dynamic.
            /// A scalar has one element and any static zero extent makes the
            /// tensor empty. This does not add IR or contact a device.
            pub fn static_numel(&self) -> Option<usize> {
                elements(self.shape()).ok()
            }

            /// Whether any axis has extent zero. Scalars are not empty.
            pub fn is_empty(&self) -> bool {
                self.shape().contains(&0)
            }
        }
    )+ };
}
shape_queries!(Tensor);

impl Tensor {
    /// Record the runtime element count as a scalar I32 SSA value.
    ///
    /// Unlike [`Self::static_numel`], this also works for bounded dynamic
    /// dimensions and therefore remains part of the compiled program.
    pub fn numel(&self) -> Result<Tensor> {
        let mut dimensions = (0..self.ndim()).map(|axis| self.dim(axis));
        let Some(first) = dimensions.next() else {
            return self.graph().constant_i32(&[], &[1]);
        };
        dimensions.try_fold(first?, |product, dimension| product.mul(&dimension?))
    }

    /// Record the runtime size of one statically selected axis as a scalar I32
    /// SSA value. Unlike `static_dim`, this remains in the generated IR.
    pub fn dim(&self, axis: usize) -> Result<Tensor> {
        if axis >= self.ndim() {
            return Err(Error::AxisOutOfRange {
                operation: "dim",
                axis,
                rank: self.ndim(),
            });
        }
        self.graph()
            .node(Op::GetDimensionSize { axis }, vec![self.node_id()], &[])
    }

    /// Record all runtime dimension sizes as a rank-one I32 tensor. Tensor rank
    /// remains static, so a scalar produces an empty `[0]` shape tensor.
    pub fn shape_tensor(&self) -> Result<Tensor> {
        if self.ndim() == 0 {
            return self.graph().constant_i32(&[0], &[]);
        }
        let dimensions = (0..self.ndim())
            .map(|axis| self.dim(axis)?.reshape(&[1]))
            .collect::<Result<Vec<_>>>()?;
        let shape = Tensor::concatenate(&dimensions, 0)?;
        debug_assert_eq!(shape.dtype(), DType::I32);
        Ok(shape)
    }
}
