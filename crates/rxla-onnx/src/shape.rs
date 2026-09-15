use std::collections::BTreeMap;

use onnx_ir::node::{
    padding::{AutoPad, PaddingConfig2d},
    reduce::ReduceAxes,
    resize::{ResizeScales, ResizeSizes},
};
use onnx_ir::{ArgType, Node};

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub value: String,
    pub inferred: Vec<usize>,
    pub declared: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeReport {
    pub values: BTreeMap<String, Vec<usize>>,
    pub outputs: Vec<Vec<usize>>,
    pub declared_mismatches: Vec<ShapeMismatch>,
}

pub(super) fn infer(graph: &onnx_ir::OnnxGraph) -> Result<ShapeReport> {
    let mut values = BTreeMap::new();
    for input in &graph.inputs {
        values.insert(
            input.name.clone(),
            concrete_shape(input).ok_or_else(|| Error::MissingShape {
                node: "graph input".into(),
                operator: "Input".into(),
                value: input.name.clone(),
            })?,
        );
    }
    let mut mismatches = Vec::new();
    for node in &graph.nodes {
        let operator = node.node_type().to_string();
        let inputs = node
            .inputs()
            .iter()
            .filter(|input| !input.is_optional())
            .map(|input| {
                if let Some(shape) = concrete_shape(input).filter(|_| input.value().is_some()) {
                    Ok(shape)
                } else {
                    values
                        .get(&input.name)
                        .cloned()
                        .ok_or_else(|| Error::MissingShape {
                            node: node.name().to_owned(),
                            operator: operator.clone(),
                            value: input.name.clone(),
                        })
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let output_shapes = infer_node(node, &inputs)?;
        if output_shapes.len() != node.outputs().len() {
            return Err(shape_error(
                node,
                "inferred output count does not match ONNX",
            ));
        }
        for (output, inferred) in node.outputs().iter().zip(output_shapes) {
            if let Some(declared) = concrete_shape(output)
                && declared != inferred
            {
                mismatches.push(ShapeMismatch {
                    value: output.name.clone(),
                    inferred: inferred.clone(),
                    declared,
                });
            }
            values.insert(output.name.clone(), inferred);
        }
    }
    let outputs = graph
        .outputs
        .iter()
        .map(|output| {
            values
                .get(&output.name)
                .cloned()
                .ok_or_else(|| Error::MissingShape {
                    node: "graph output".into(),
                    operator: "Output".into(),
                    value: output.name.clone(),
                })
        })
        .collect::<Result<_>>()?;
    Ok(ShapeReport {
        values,
        outputs,
        declared_mismatches: mismatches,
    })
}

fn infer_node(node: &Node, inputs: &[Vec<usize>]) -> Result<Vec<Vec<usize>>> {
    let one = |shape: Vec<usize>| Ok(vec![shape]);
    match node {
        Node::Constant(_) => one(concrete_shape(&node.outputs()[0])
            .ok_or_else(|| shape_error(node, "constant has no concrete shape"))?),
        Node::Relu(_) | Node::Erf(_) | Node::HardSigmoid(_) | Node::Sigmoid(_) => {
            one(inputs[0].clone())
        }
        Node::Add(_) | Node::Mul(_) | Node::Div(_) => one(broadcast(&inputs[0], &inputs[1])
            .ok_or_else(|| shape_error(node, "inputs are not broadcast-compatible"))?),
        Node::Conv2d(conv) => {
            let [n, _, h, w] = as_4d(node, &inputs[0])?;
            let [out_channels, _, kh, kw] = as_4d(node, &inputs[1])?;
            let [top, left, bottom, right] = padding(&conv.config.padding);
            let oh = spatial_output(
                h,
                kh,
                [top, bottom],
                conv.config.stride[0],
                conv.config.dilation[0],
                false,
                &conv.config.auto_pad,
            );
            let ow = spatial_output(
                w,
                kw,
                [left, right],
                conv.config.stride[1],
                conv.config.dilation[1],
                false,
                &conv.config.auto_pad,
            );
            one(vec![n, out_channels, oh, ow])
        }
        Node::ConvTranspose2d(conv) => {
            let [n, _, h, w] = as_4d(node, &inputs[0])?;
            let [_, output_per_group, kh, kw] = as_4d(node, &inputs[1])?;
            let output = |size, kernel, axis| {
                (size - 1) * conv.config.stride[axis] - 2 * conv.config.padding[axis]
                    + conv.config.dilation[axis] * (kernel - 1)
                    + conv.config.padding_out[axis]
                    + 1
            };
            one(vec![
                n,
                output_per_group * conv.config.groups,
                output(h, kh, 0),
                output(w, kw, 1),
            ])
        }
        Node::MaxPool2d(pool) => {
            let [n, c, h, w] = as_4d(node, &inputs[0])?;
            let [top, left, bottom, right] = padding(&pool.config.padding);
            one(vec![
                n,
                c,
                spatial_output(
                    h,
                    pool.config.kernel_size[0],
                    [top, bottom],
                    pool.config.strides[0],
                    pool.config.dilation[0],
                    pool.config.ceil_mode,
                    &pool.config.auto_pad,
                ),
                spatial_output(
                    w,
                    pool.config.kernel_size[1],
                    [left, right],
                    pool.config.strides[1],
                    pool.config.dilation[1],
                    pool.config.ceil_mode,
                    &pool.config.auto_pad,
                ),
            ])
        }
        Node::ReduceMean(reduce) => {
            let axes = match &reduce.config.axes {
                ReduceAxes::Static(axes) => axes,
                ReduceAxes::Runtime(_) => return Err(shape_error(node, "runtime reduction axes")),
            };
            let axes: Vec<_> = if axes.is_empty() {
                (0..inputs[0].len()).collect()
            } else {
                axes.clone()
            };
            let mut shape = inputs[0].clone();
            if reduce.config.keepdims {
                for axis in axes {
                    shape[axis] = 1;
                }
            } else {
                for axis in axes.into_iter().rev() {
                    shape.remove(axis);
                }
            }
            one(shape)
        }
        Node::Concat(concat) => {
            let mut shape = inputs[0].clone();
            shape[concat.config.axis] = inputs.iter().map(|s| s[concat.config.axis]).sum();
            one(shape)
        }
        Node::Resize(resize) => {
            let [n, c, h, w] = as_4d(node, &inputs[0])?;
            let spatial = match (&resize.config.sizes, &resize.config.scales) {
                (Some(ResizeSizes::Static(sizes)), _) => sizes.clone(),
                (_, Some(ResizeScales::Static(scales))) => vec![
                    (h as f32 * scales[0]).floor() as usize,
                    (w as f32 * scales[1]).floor() as usize,
                ],
                _ => return Err(shape_error(node, "runtime resize size/scale")),
            };
            if spatial.len() != 2 {
                return Err(shape_error(node, "resize is not spatial 2D"));
            }
            one(vec![n, c, spatial[0], spatial[1]])
        }
        _ => Err(Error::UnsupportedShapeOperator {
            operator: node.node_type().to_string(),
        }),
    }
}

fn concrete_shape(argument: &onnx_ir::Argument) -> Option<Vec<usize>> {
    match &argument.ty {
        ArgType::Tensor(tensor) => tensor.static_shape_known(),
        ArgType::ScalarTensor(_) => Some(vec![1]),
        ArgType::ScalarNative(_) => Some(vec![]),
        ArgType::Shape(rank) => Some(vec![*rank]),
    }
}

fn broadcast(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    (0..rank)
        .map(|offset| {
            let a = lhs
                .get(lhs.len().wrapping_sub(1 + offset))
                .copied()
                .unwrap_or(1);
            let b = rhs
                .get(rhs.len().wrapping_sub(1 + offset))
                .copied()
                .unwrap_or(1);
            (a == b || a == 1 || b == 1).then_some(a.max(b))
        })
        .collect::<Option<Vec<_>>>()
        .map(|mut shape| {
            shape.reverse();
            shape
        })
}

fn padding(value: &PaddingConfig2d) -> [usize; 4] {
    match value {
        PaddingConfig2d::Valid => [0; 4],
        PaddingConfig2d::Explicit(top, left, bottom, right) => [*top, *left, *bottom, *right],
    }
}

fn conv_output(
    input: usize,
    kernel: usize,
    low: usize,
    high: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    (input + low + high - dilation * (kernel - 1) - 1) / stride + 1
}

fn pool_output(
    input: usize,
    kernel: usize,
    low: usize,
    high: usize,
    stride: usize,
    dilation: usize,
    ceil: bool,
) -> usize {
    let numerator = input + low + high - dilation * (kernel - 1) - 1;
    if ceil {
        numerator.div_ceil(stride) + 1
    } else {
        numerator / stride + 1
    }
}

fn spatial_output(
    input: usize,
    kernel: usize,
    padding: [usize; 2],
    stride: usize,
    dilation: usize,
    ceil: bool,
    auto: &AutoPad,
) -> usize {
    if matches!(auto, AutoPad::SameUpper | AutoPad::SameLower) {
        input.div_ceil(stride)
    } else if ceil {
        pool_output(
            input, kernel, padding[0], padding[1], stride, dilation, true,
        )
    } else {
        conv_output(input, kernel, padding[0], padding[1], stride, dilation)
    }
}

fn as_4d(node: &Node, shape: &[usize]) -> Result<[usize; 4]> {
    shape
        .try_into()
        .map_err(|_| shape_error(node, "expected rank four"))
}

fn shape_error(node: &Node, message: impl Into<String>) -> Error {
    Error::ShapeInference {
        node: node.name().to_owned(),
        operator: node.node_type().to_string(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{broadcast, conv_output, pool_output};

    #[test]
    fn numpy_broadcast_shape() {
        assert_eq!(
            broadcast(&[1, 32, 20, 30], &[32, 1, 1]),
            Some(vec![1, 32, 20, 30])
        );
        assert_eq!(broadcast(&[2, 3], &[4, 3]), None);
    }

    #[test]
    fn spatial_output_shape() {
        assert_eq!(conv_output(640, 3, 1, 1, 2, 1), 320);
        assert_eq!(pool_output(5, 2, 0, 0, 2, 1, false), 2);
        assert_eq!(pool_output(5, 2, 0, 0, 2, 1, true), 3);
    }
}
