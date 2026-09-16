//! Typed symbolic model outputs and concrete execution results.

use rxla_core::{Buffer, Tensor};

/// Structured values a model `apply` function may return.
///
/// Containers are flattened deterministically into the executable output ABI.
/// Applications may implement this for domain structs to keep model signatures
/// typed without exposing their storage structure to the compiler boundary.
pub trait ModelOutputs {
    fn into_tensors(self) -> Vec<Tensor>;
}

impl ModelOutputs for () {
    fn into_tensors(self) -> Vec<Tensor> {
        Vec::new()
    }
}

impl ModelOutputs for Tensor {
    fn into_tensors(self) -> Vec<Tensor> {
        vec![self]
    }
}

impl<T: ModelOutputs> ModelOutputs for Vec<T> {
    fn into_tensors(self) -> Vec<Tensor> {
        self.into_iter()
            .flat_map(ModelOutputs::into_tensors)
            .collect()
    }
}

impl<T: ModelOutputs, const N: usize> ModelOutputs for [T; N] {
    fn into_tensors(self) -> Vec<Tensor> {
        self.into_iter()
            .flat_map(ModelOutputs::into_tensors)
            .collect()
    }
}

/// Reconstruct a typed host-side result from ordered model output buffers.
///
/// Implementations must consume exactly the buffers belonging to one value.
/// [`crate::AppliedModel::decode_outputs`] checks both the model ABI count and
/// that the requested structure leaves no buffers behind.
pub trait ModelOutputValues: Sized {
    fn take_from(buffers: &mut std::vec::IntoIter<Buffer>) -> Option<Self>;
}

impl ModelOutputValues for () {
    fn take_from(_buffers: &mut std::vec::IntoIter<Buffer>) -> Option<Self> {
        Some(())
    }
}

impl ModelOutputValues for Buffer {
    fn take_from(buffers: &mut std::vec::IntoIter<Buffer>) -> Option<Self> {
        buffers.next()
    }
}

impl ModelOutputValues for Vec<Buffer> {
    fn take_from(buffers: &mut std::vec::IntoIter<Buffer>) -> Option<Self> {
        Some(buffers.by_ref().collect())
    }
}

impl<const N: usize> ModelOutputValues for [Buffer; N] {
    fn take_from(buffers: &mut std::vec::IntoIter<Buffer>) -> Option<Self> {
        buffers.by_ref().take(N).collect::<Vec<_>>().try_into().ok()
    }
}

macro_rules! impl_tuple_outputs {
    ($(($($name:ident),+)),+ $(,)?) => {
        $(
            impl<$($name: ModelOutputs),+> ModelOutputs for ($($name,)+) {
                #[allow(non_snake_case)]
                fn into_tensors(self) -> Vec<Tensor> {
                    let ($($name,)+) = self;
                    let mut outputs = Vec::new();
                    $(outputs.extend($name.into_tensors());)+
                    outputs
                }
            }

            impl<$($name: ModelOutputValues),+> ModelOutputValues for ($($name,)+) {
                #[allow(non_snake_case)]
                fn take_from(buffers: &mut std::vec::IntoIter<Buffer>) -> Option<Self> {
                    Some(($($name::take_from(buffers)?,)+))
                }
            }
        )+
    };
}

impl_tuple_outputs!(
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
    use crate::{Cx, Error, Model, Result};
    use rxla_core::{CacheLimits, Client, Compiler};

    struct EmptyResult;

    impl ModelOutputValues for EmptyResult {
        fn take_from(_buffers: &mut std::vec::IntoIter<Buffer>) -> Option<Self> {
            Some(Self)
        }
    }

    #[test]
    fn applied_models_validate_and_decode_output_structure() {
        let (_, empty) = Model::new(|_: &mut Cx| Ok::<_, Error>(())).trace().unwrap();
        let _: EmptyResult = empty.decode_outputs(Vec::new()).unwrap();
        assert!(matches!(
            empty.decode_outputs::<Buffer>(Vec::new()),
            Err(Error::OutputStructure)
        ));

        let (_, one) = Model::new(|cx: &mut Cx| -> Result<Tensor> { cx.constant(&[], &[1.0]) })
            .trace()
            .unwrap();
        assert!(matches!(
            one.decode_outputs::<()>(Vec::new()),
            Err(Error::OutputCount {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn typed_output_buffers_execute_on_cpu() {
        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .unwrap();
        let (_, applied) = Model::new(|cx: &mut Cx| {
            Ok::<_, Error>((cx.constant(&[], &[3.0])?, cx.constant(&[2], &[4.0, 5.0])?))
        })
        .trace()
        .unwrap();
        let mut compiler = Compiler::new(client, CacheLimits::default());
        let executable = applied.compile(&mut compiler).unwrap();
        let outputs = executable.execute(&[]).unwrap();
        let (scalar, vector): (Buffer, Buffer) = applied.decode_outputs(outputs).unwrap();
        assert_eq!(scalar.to_vec::<f32>().unwrap(), [3.0]);
        assert_eq!(vector.to_vec::<f32>().unwrap(), [4.0, 5.0]);
    }
}
