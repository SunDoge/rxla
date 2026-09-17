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

            /// Static element count; a scalar has one element and any zero
            /// extent makes the tensor empty. This is not a byte or memory-usage
            /// estimate and does not compile, execute or synchronize the graph.
            pub fn numel(&self) -> usize {
                // All constructors validate this same product before exposing
                // a handle; callers cannot mutate its shape metadata.
                elements(self.shape()).expect("symbolic shape validated at construction")
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
