//! Fallible Rust operators for lazy Tensor expressions.
use crate::{Result, Tensor};
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! binary_operator {
    ($trait:ident, $method:ident, $tensor_method:ident) => {
        impl $trait<&Tensor> for &Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: &Tensor) -> Self::Output {
                Tensor::$tensor_method(self, rhs)
            }
        }
        impl $trait<Tensor> for Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: Tensor) -> Self::Output {
                Tensor::$tensor_method(&self, &rhs)
            }
        }
        impl $trait<&Tensor> for Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: &Tensor) -> Self::Output {
                Tensor::$tensor_method(&self, rhs)
            }
        }
        impl $trait<Tensor> for &Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: Tensor) -> Self::Output {
                Tensor::$tensor_method(self, &rhs)
            }
        }
    };
}

binary_operator!(Add, add, add);
binary_operator!(Sub, sub, sub);
binary_operator!(Mul, mul, mul);
binary_operator!(Div, div, div);

impl Neg for &Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}

impl Add<f32> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: f32) -> Self::Output {
        Tensor::add_scalar(self, rhs)
    }
}

impl Mul<f32> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: f32) -> Self::Output {
        Tensor::mul_scalar(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operators_build_fallible_lazy_expressions_without_consuming_inputs() {
        let x = Tensor::from_slice([2], crate::DType::F32, [1.0, 2.0]).unwrap();
        let y = Tensor::from_slice([2], crate::DType::F32, [3.0, 4.0]).unwrap();
        let z = ((&x + &y).unwrap() * &x).unwrap();
        assert_eq!(z.shape(), [2]);
        assert!(!z.is_materialized());
        assert!((&x + 1.0).is_ok());
        assert!((-&y).is_ok());
    }
}
