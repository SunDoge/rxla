//! Typed model-input declarations.

use crate::{Cx, Result, TraceOutputs};
use rxla_core::{DType, Tensor};

/// One tensor input specification for a [`crate::Model`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInput {
    shape: Vec<i64>,
    dtype: DType,
}

impl ModelInput {
    pub fn new(shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::F32,
        }
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn element_type(&self) -> DType {
        self.dtype
    }
}

/// Declare a structured input specification as equally structured lazy tensors.
///
/// Implementations may compose existing implementations, allowing applications
/// to use domain-specific input structs without teaching [`crate::Model`] about
/// every possible model signature.
pub trait ModelInputs {
    type Tensors;

    fn declare(&self, cx: &mut Cx) -> Result<Self::Tensors>;
}

/// Invoke a model function after extracting its typed lazy inputs.
///
/// Implementations are provided for one structured argument and for ordinary
/// Rust functions with two through six extracted arguments. The `Marker` type
/// disambiguates function arities in the same way as web-framework handler
/// traits and is normally inferred.
pub trait ModelHandler<I, Marker> {
    type Outputs: TraceOutputs;

    fn invoke(&self, cx: &mut Cx, inputs: &I) -> Result<Self::Outputs>;
}

impl<F, I, T> ModelHandler<I, fn(I) -> T> for F
where
    I: ModelInputs,
    F: Fn(&mut Cx, I::Tensors) -> Result<T>,
    T: TraceOutputs,
{
    type Outputs = T;

    fn invoke(&self, cx: &mut Cx, inputs: &I) -> Result<Self::Outputs> {
        let inputs = inputs.declare(cx)?;
        self(cx, inputs)
    }
}

impl ModelInputs for ModelInput {
    type Tensors = Tensor;

    fn declare(&self, cx: &mut Cx) -> Result<Self::Tensors> {
        cx.input_dtype(&self.shape, self.dtype)
    }
}

impl<I: ModelInputs, const N: usize> ModelInputs for [I; N] {
    type Tensors = [I::Tensors; N];

    fn declare(&self, cx: &mut Cx) -> Result<Self::Tensors> {
        let tensors = self
            .iter()
            .map(|input| input.declare(cx))
            .collect::<Result<Vec<_>>>()?;
        match tensors.try_into() {
            Ok(tensors) => Ok(tensors),
            Err(_) => unreachable!("array input arity is preserved"),
        }
    }
}

impl<I: ModelInputs> ModelInputs for Vec<I> {
    type Tensors = Vec<I::Tensors>;

    fn declare(&self, cx: &mut Cx) -> Result<Self::Tensors> {
        self.iter().map(|input| input.declare(cx)).collect()
    }
}

macro_rules! impl_tuple_inputs {
    ($(($($name:ident),+)),+ $(,)?) => {
        $ (
            impl<$($name: ModelInputs),+> ModelInputs for ($($name,)+) {
                type Tensors = ($($name::Tensors,)+);

                #[allow(non_snake_case)]
                fn declare(&self, cx: &mut Cx) -> Result<Self::Tensors> {
                    let ($($name,)+) = self;
                    Ok(($($name.declare(cx)?,)+))
                }
            }
        )+
    };
}

impl_tuple_inputs!(
    (A, B),
    (A, B, C),
    (A, B, C, D),
    (A, B, C, D, E),
    (A, B, C, D, E, F),
    (A, B, C, D, E, F, G),
    (A, B, C, D, E, F, G, H),
    (A, B, C, D, E, F, G, H, I),
    (A, B, C, D, E, F, G, H, I, J),
    (A, B, C, D, E, F, G, H, I, J, K),
    (A, B, C, D, E, F, G, H, I, J, K, L),
    (A, B, C, D, E, F, G, H, I, J, K, L, M),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
);

macro_rules! impl_tuple_handlers {
    ($(($($name:ident),+)),+ $(,)?) => {
        $(
            impl<Func, Output, $($name: ModelInputs),+>
                ModelHandler<($($name,)+), fn($($name),+) -> Output> for Func
            where
                Func: Fn(&mut Cx, $($name::Tensors),+) -> Result<Output>,
                Output: TraceOutputs,
            {
                type Outputs = Output;

                #[allow(non_snake_case)]
                fn invoke(&self, cx: &mut Cx, inputs: &($($name,)+)) -> Result<Self::Outputs> {
                    let ($($name,)+) = inputs;
                    $(let $name = $name.declare(cx)?;)+
                    self(cx, $($name),+)
                }
            }
        )+
    };
}

impl_tuple_handlers!(
    (A, B),
    (A, B, C),
    (A, B, C, D),
    (A, B, C, D, E),
    (A, B, C, D, E, F),
    (A, B, C, D, E, F, G),
    (A, B, C, D, E, F, G, H),
    (A, B, C, D, E, F, G, H, I),
    (A, B, C, D, E, F, G, H, I, J),
    (A, B, C, D, E, F, G, H, I, J, K),
    (A, B, C, D, E, F, G, H, I, J, K, L),
    (A, B, C, D, E, F, G, H, I, J, K, L, M),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;

    struct Batch {
        images: ModelInput,
        labels: ModelInput,
    }

    struct BatchTensors {
        images: Tensor,
        labels: Tensor,
    }

    impl ModelInputs for Batch {
        type Tensors = BatchTensors;

        fn declare(&self, cx: &mut Cx) -> Result<Self::Tensors> {
            Ok(BatchTensors {
                images: self.images.declare(cx)?,
                labels: self.labels.declare(cx)?,
            })
        }
    }

    #[test]
    fn single_tuple_array_vector_and_custom_inputs_preserve_structure() {
        let single = Model::new(|_: &mut Cx, x: Tensor| Ok(x)).inputs(ModelInput::new(vec![2, 3]));
        assert_eq!(single.trace().unwrap().1.outputs()[0].shape(), [2, 3]);

        let tuple = Model::new(|_: &mut Cx, x: Tensor, y: Tensor| Ok(x.add(&y)?))
            .inputs((ModelInput::new(vec![2]), ModelInput::new(vec![2])));
        assert_eq!(tuple.trace().unwrap().0.inputs().len(), 2);

        let array = Model::new(|_: &mut Cx, [x, y]: [Tensor; 2]| Ok(x.add(&y)?))
            .inputs([ModelInput::new(vec![2]), ModelInput::new(vec![2])]);
        assert_eq!(array.trace().unwrap().0.inputs().len(), 2);

        let vector = Model::new(|_: &mut Cx, xs: Vec<Tensor>| Ok(xs[0].add(&xs[1])?))
            .inputs(vec![ModelInput::new(vec![2]), ModelInput::new(vec![2])]);
        assert_eq!(vector.trace().unwrap().0.inputs().len(), 2);

        let custom =
            Model::new(|_: &mut Cx, batch: BatchTensors| Ok(batch.images.add(&batch.labels)?))
                .inputs(Batch {
                    images: ModelInput::new(vec![2]),
                    labels: ModelInput::new(vec![2]),
                });
        assert_eq!(custom.trace().unwrap().0.inputs().len(), 2);
    }
}
