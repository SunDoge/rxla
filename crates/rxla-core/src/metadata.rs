//! Infallible queries over already validated, static symbolic shapes.
use super::{Tensor, elements};

macro_rules! shape_queries {
    ($($ty:ty),+ $(,)?) => { $(
        impl $ty {
            /// Number of axes; a scalar has rank zero. Reads symbolic metadata
            /// only, without adding graph nodes or contacting a device.
            pub fn ndim(&self) -> usize {
                self.shape().len()
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
