//! Typed model-input declarations and function-handler adaptation.
//!
//! Input specifications stay outside the mathematical model body, while the
//! handler receives ordinary lazy tensors as separate Rust arguments:
//!
//! ```
//! use rxla_core::{DType, Tensor};
//! use rxla_nn::{Cx, Linear, Model, ModelInput, Result, TensorApply};
//!
//! fn apply(cx: Cx, image: Tensor, label: Tensor) -> Result<(Tensor, Tensor)> {
//!     let logits = image.apply(&cx.layer("head", Linear::new(10))?)?;
//!     let loss = logits.cross_entropy_with_indices(&label, 1)?.mean(&[0], false)?;
//!     Ok((loss, logits))
//! }
//!
//! let applied = Model::new(apply)
//!     .inputs((
//!         ModelInput::new([4, 32]),
//!         ModelInput::new([4]).with_dtype(DType::I32),
//!     ))
//!     .trace()?;
//! assert_eq!(applied.schema().inputs().len(), 2);
//! assert_eq!(applied.outputs().len(), 2);
//! # Ok::<(), rxla_nn::Error>(())
//! ```

use crate::{Cx, ModelOutputs, NoModelInputs, Result};
use rxla_core::{Buffer, DType, Tensor};

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

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
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

    fn declare(&self, cx: &Cx) -> Result<Self::Tensors>;
}

/// Flatten structured runtime input buffers in the same deterministic order as
/// their [`ModelInputs`] declarations.
///
/// This trait is intentionally independent from a particular model so domain
/// structs can implement it once and be passed directly to
/// [`crate::AppliedModel::bind`]. Shape, dtype, count and client ownership are
/// still validated against the applied model schema before execution.
pub trait ModelInputValues<'a> {
    fn append_to(self, values: &mut Vec<&'a Buffer>);

    fn into_values(self) -> Vec<&'a Buffer>
    where
        Self: Sized,
    {
        let mut values = Vec::new();
        self.append_to(&mut values);
        values
    }
}

impl<'a> ModelInputValues<'a> for &'a Buffer {
    fn append_to(self, values: &mut Vec<&'a Buffer>) {
        values.push(self);
    }
}

impl<'a> ModelInputValues<'a> for &[&'a Buffer] {
    fn append_to(self, values: &mut Vec<&'a Buffer>) {
        values.extend_from_slice(self);
    }
}

impl<'a, const N: usize> ModelInputValues<'a> for &[&'a Buffer; N] {
    fn append_to(self, values: &mut Vec<&'a Buffer>) {
        values.extend_from_slice(self);
    }
}

impl<'a, I: ModelInputValues<'a>, const N: usize> ModelInputValues<'a> for [I; N] {
    fn append_to(self, values: &mut Vec<&'a Buffer>) {
        for input in self {
            input.append_to(values);
        }
    }
}

impl<'a, I: ModelInputValues<'a>> ModelInputValues<'a> for Vec<I> {
    fn append_to(self, values: &mut Vec<&'a Buffer>) {
        for input in self {
            input.append_to(values);
        }
    }
}

/// Invoke a model function after extracting its typed lazy inputs.
///
/// Implementations are provided for one structured argument and for ordinary
/// Rust functions with two through sixteen extracted arguments. The `Marker` type
/// disambiguates function arities in the same way as web-framework handler
/// traits and is normally inferred.
pub trait ModelHandler<I, Marker> {
    type Outputs: ModelOutputs;
    type Error: From<crate::Error>;

    fn invoke(&self, cx: Cx, inputs: &I) -> std::result::Result<Self::Outputs, Self::Error>;
}

impl<F, T, E> ModelHandler<NoModelInputs, fn() -> std::result::Result<T, E>> for F
where
    F: Fn(Cx) -> std::result::Result<T, E>,
    T: ModelOutputs,
    E: From<crate::Error>,
{
    type Outputs = T;
    type Error = E;

    fn invoke(&self, cx: Cx, _: &NoModelInputs) -> std::result::Result<Self::Outputs, Self::Error> {
        self(cx)
    }
}

impl<F, I, T, E> ModelHandler<I, fn(I) -> std::result::Result<T, E>> for F
where
    I: ModelInputs,
    F: Fn(Cx, I::Tensors) -> std::result::Result<T, E>,
    T: ModelOutputs,
    E: From<crate::Error>,
{
    type Outputs = T;
    type Error = E;

    fn invoke(&self, cx: Cx, inputs: &I) -> std::result::Result<Self::Outputs, Self::Error> {
        let inputs = inputs.declare(&cx).map_err(E::from)?;
        self(cx, inputs)
    }
}

impl ModelInputs for ModelInput {
    type Tensors = Tensor;

    fn declare(&self, cx: &Cx) -> Result<Self::Tensors> {
        cx.input_dtype(&self.shape, self.dtype)
    }
}

impl ModelInputs for NoModelInputs {
    type Tensors = ();

    fn declare(&self, _: &Cx) -> Result<Self::Tensors> {
        Ok(())
    }
}

impl<I: ModelInputs, const N: usize> ModelInputs for [I; N] {
    type Tensors = [I::Tensors; N];

    fn declare(&self, cx: &Cx) -> Result<Self::Tensors> {
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

    fn declare(&self, cx: &Cx) -> Result<Self::Tensors> {
        self.iter().map(|input| input.declare(cx)).collect()
    }
}

macro_rules! impl_tuple_inputs {
    ($(($($name:ident),+)),+ $(,)?) => {
        $ (
            impl<$($name: ModelInputs),+> ModelInputs for ($($name,)+) {
                type Tensors = ($($name::Tensors,)+);

                #[allow(non_snake_case)]
                fn declare(&self, cx: &Cx) -> Result<Self::Tensors> {
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

macro_rules! impl_tuple_input_values {
    ($(($($name:ident),+)),+ $(,)?) => {
        $(
            impl<'a, $($name: ModelInputValues<'a>),+> ModelInputValues<'a> for ($($name,)+) {
                #[allow(non_snake_case)]
                fn append_to(self, values: &mut Vec<&'a Buffer>) {
                    let ($($name,)+) = self;
                    $($name.append_to(values);)+
                }
            }
        )+
    };
}

impl_tuple_input_values!(
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
            impl<Func, Output, HandlerError, $($name: ModelInputs),+>
                ModelHandler<
                    ($($name,)+),
                    fn($($name),+) -> std::result::Result<Output, HandlerError>,
                > for Func
            where
                Func: Fn(
                    Cx,
                    $($name::Tensors),+
                ) -> std::result::Result<Output, HandlerError>,
                Output: ModelOutputs,
                HandlerError: From<crate::Error>,
            {
                type Outputs = Output;
                type Error = HandlerError;

                #[allow(non_snake_case)]
                fn invoke(
                    &self,
                    cx: Cx,
                    inputs: &($($name,)+),
                ) -> std::result::Result<Self::Outputs, Self::Error> {
                    let ($($name,)+) = inputs;
                    $(let $name = $name.declare(&cx).map_err(HandlerError::from)?;)+
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
    use crate::{Error, Model};

    #[derive(Debug)]
    enum ModelError {
        Framework(#[allow(dead_code)] Error),
        Rejected,
    }

    impl From<Error> for ModelError {
        fn from(error: Error) -> Self {
            Self::Framework(error)
        }
    }

    struct Batch {
        images: ModelInput,
        labels: ModelInput,
    }

    struct BatchTensors {
        images: Tensor,
        labels: Tensor,
    }

    struct Predictions {
        logits: Tensor,
        auxiliary: Tensor,
    }

    impl ModelOutputs for Predictions {
        fn into_tensors(self) -> Vec<Tensor> {
            vec![self.logits, self.auxiliary]
        }
    }

    impl ModelInputs for Batch {
        type Tensors = BatchTensors;

        fn declare(&self, cx: &Cx) -> Result<Self::Tensors> {
            Ok(BatchTensors {
                images: self.images.declare(cx)?,
                labels: self.labels.declare(cx)?,
            })
        }
    }

    #[test]
    fn single_tuple_array_vector_and_custom_inputs_preserve_structure() {
        let input = ModelInput::new([2, 3]);
        assert_eq!(input.shape(), [2, 3]);
        assert_eq!(input.dtype(), DType::F32);
        let single = Model::new(|_: Cx, x: Tensor| Ok::<_, Error>(x)).inputs(input);
        assert_eq!(single.trace().unwrap().outputs()[0].shape(), [2, 3]);

        let tuple = Model::new(|_: Cx, x: Tensor, y: Tensor| -> Result<_> { Ok(x.add(&y)?) })
            .inputs((ModelInput::new(vec![2]), ModelInput::new(vec![2])));
        assert_eq!(tuple.trace().unwrap().schema().inputs().len(), 2);

        let array = Model::new(|_: Cx, [x, y]: [Tensor; 2]| -> Result<_> { Ok(x.add(&y)?) })
            .inputs([ModelInput::new(vec![2]), ModelInput::new(vec![2])]);
        assert_eq!(array.trace().unwrap().schema().inputs().len(), 2);

        let vector = Model::new(|_: Cx, xs: Vec<Tensor>| -> Result<_> { Ok(xs[0].add(&xs[1])?) })
            .inputs(vec![ModelInput::new(vec![2]), ModelInput::new(vec![2])]);
        assert_eq!(vector.trace().unwrap().schema().inputs().len(), 2);

        let custom = Model::new(|_: Cx, batch: BatchTensors| -> Result<_> {
            Ok(Predictions {
                logits: batch.images.add(&batch.labels)?,
                auxiliary: batch.images,
            })
        })
        .inputs(Batch {
            images: ModelInput::new(vec![2]),
            labels: ModelInput::new(vec![2]),
        });
        let applied = custom.trace().unwrap();
        assert_eq!(applied.schema().inputs().len(), 2);
        assert_eq!(applied.outputs().len(), 2);

        let nested =
            Model::new(|_: Cx, x: Tensor| -> Result<_> { Ok((x.clone(), [x.clone(), x])) })
                .inputs(ModelInput::new(vec![2]));
        assert_eq!(nested.trace().unwrap().outputs().len(), 3);
    }

    #[test]
    fn model_handlers_preserve_downstream_errors() {
        let model = Model::new(|_: Cx| -> std::result::Result<Tensor, ModelError> {
            Err(ModelError::Rejected)
        });

        assert!(matches!(model.trace(), Err(ModelError::Rejected)));
    }
}
