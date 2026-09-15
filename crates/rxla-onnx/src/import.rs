use std::{collections::HashMap, sync::Arc};

use onnx_ir::node::{
    padding::{AutoPad, PaddingConfig2d},
    reduce::ReduceAxes,
    resize::{CoordinateTransformMode, NearestMode, ResizeMode, ResizeScales, ResizeSizes},
};
use onnx_ir::{Node, TensorDataExt, ValueSource};
use rxla_ir::{
    Binary, Conv2dOptions, ConvTranspose2dOptions, Op, Pool2dOptions, ProgramIr, Reduction, SsaId,
    TensorType, Unary,
};
use rxla_pjrt::DType;
use snafu::ResultExt;

use crate::{Error, Result, shape};

pub struct ImportedProgram {
    pub program: ProgramIr,
    pub outputs: Vec<SsaId>,
    pub inputs: Vec<String>,
    pub parameters: Vec<Parameter>,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Arc<[f32]>,
}

pub(super) fn import(graph: onnx_ir::OnnxGraph) -> Result<ImportedProgram> {
    let shapes = shape::infer(&graph)?;
    let mut cx = Importer {
        program: ProgramIr::default(),
        values: HashMap::new(),
        static_values: HashMap::new(),
        shapes: &shapes.values,
        inputs: Vec::new(),
        parameters: Vec::new(),
    };
    for input in &graph.inputs {
        let shape = cx.shape(&input.name)?.clone();
        let parameter = cx.parameter(&input.name, &shape, None)?;
        let id = cx.transpose_to_nhwc(parameter, &shape)?;
        cx.inputs.push(input.name.clone());
        cx.values.insert(input.name.clone(), id);
    }
    for node in &graph.nodes {
        cx.node(node)?;
    }
    let outputs = graph
        .outputs
        .iter()
        .map(|output| {
            let id = cx.value(&output.name, node_label("graph output", "Output"))?;
            cx.transpose_to_nchw(id, cx.shape(&output.name)?.clone())
        })
        .collect::<Result<_>>()?;
    Ok(ImportedProgram {
        program: cx.program,
        outputs,
        inputs: cx.inputs,
        parameters: cx.parameters,
    })
}

struct Importer<'a> {
    program: ProgramIr,
    values: HashMap<String, SsaId>,
    static_values: HashMap<onnx_ir::DataId, SsaId>,
    shapes: &'a std::collections::BTreeMap<String, Vec<usize>>,
    inputs: Vec<String>,
    parameters: Vec<Parameter>,
}

impl Importer<'_> {
    fn node(&mut self, node: &Node) -> Result<()> {
        if matches!(node, Node::Constant(_)) {
            self.resolve(&node.outputs()[0], node)?;
            return Ok(());
        }
        let args = node
            .inputs()
            .iter()
            .filter(|arg| !arg.is_optional())
            .collect::<Vec<_>>();
        let ids = args
            .iter()
            .map(|arg| self.resolve(arg, node))
            .collect::<Result<Vec<_>>>()?;
        let output = &node.outputs()[0];
        let result = self.physical_ty(&output.name)?;
        let id = match node {
            Node::Relu(_) => self.append(Op::Relu, &ids, &result)?,
            Node::Erf(_) => self.append(Op::Unary(Unary::Erf), &ids, &result)?,
            Node::Sigmoid(_) => self.append(Op::Sigmoid, &ids, &result)?,
            Node::Add(_) => self.binary(Binary::Add, &args, &ids, &result, node)?,
            Node::Mul(_) => self.binary(Binary::Mul, &args, &ids, &result, node)?,
            Node::Div(_) => self.binary(Binary::Div, &args, &ids, &result, node)?,
            Node::Concat(concat) => self.append(
                Op::Concatenate {
                    axis: physical_axis(concat.config.axis, result.dims.len()),
                },
                &ids,
                &result,
            )?,
            Node::Conv2d(conv) => self.conv2d(conv, &args, &ids, &result, node)?,
            Node::ConvTranspose2d(conv) => self.conv_transpose(conv, &args, &ids, &result, node)?,
            Node::ReduceMean(reduce) => {
                let axes = match &reduce.config.axes {
                    ReduceAxes::Static(axes) => axes.clone(),
                    ReduceAxes::Runtime(_) => {
                        return Err(import_error(node, "runtime reduction axes"));
                    }
                };
                let input_shape = self.physical_shape(&args[0].name)?;
                let axes: Vec<_> = axes
                    .into_iter()
                    .map(|axis| physical_axis(axis, input_shape.len()))
                    .collect();
                let mut reduced_shape = input_shape.clone();
                for &axis in axes.iter().rev() {
                    reduced_shape.remove(axis);
                }
                let reduced_ty = TensorType {
                    dims: dims(&reduced_shape),
                    dtype: DType::F32,
                };
                let sum = self.append(
                    Op::Reduce {
                        kind: Reduction::Sum,
                        axes: axes.clone(),
                    },
                    &[ids[0]],
                    &reduced_ty,
                )?;
                let count = axes
                    .iter()
                    .map(|&axis| input_shape[axis])
                    .product::<usize>() as f32;
                let scalar = self.constant_scalar(count)?;
                let divisor = self.broadcast_id(scalar, &[], &reduced_ty, node)?;
                let mean = self.append(Op::Binary(Binary::Div), &[sum, divisor], &reduced_ty)?;
                if reduce.config.keepdims {
                    self.append(Op::Reshape, &[mean], &result)?
                } else {
                    mean
                }
            }
            Node::HardSigmoid(hard) => self.hard_sigmoid(
                hard.config.alpha as f32,
                hard.config.beta as f32,
                ids[0],
                &result,
                node,
            )?,
            Node::MaxPool2d(pool) => {
                let input = self.shape(&args[0].name)?;
                let padding = pool_padding(
                    input,
                    pool.config.kernel_size,
                    pool.config.strides,
                    pool.config.dilation,
                    &pool.config.padding,
                    &pool.config.auto_pad,
                );
                self.append(
                    Op::MaxPool2d(Pool2dOptions {
                        window: cast2(pool.config.kernel_size),
                        strides: cast2(pool.config.strides),
                        padding,
                    }),
                    &[ids[0]],
                    &result,
                )?
            }
            Node::Resize(resize) => self.resize(resize, ids[0], &args[0].name, &result, node)?,
            _ => {
                return Err(Error::UnsupportedOperator {
                    operator: node.node_type().to_string(),
                });
            }
        };
        self.values.insert(output.name.clone(), id);
        Ok(())
    }

    fn conv2d(
        &mut self,
        conv: &onnx_ir::node::conv2d::Conv2dNode,
        args: &[&onnx_ir::Argument],
        ids: &[SsaId],
        result: &TensorType,
        node: &Node,
    ) -> Result<SsaId> {
        let input = self.shape(&args[0].name)?;
        let padding = pool_padding(
            input,
            conv.config.kernel_size,
            conv.config.stride,
            conv.config.dilation,
            &conv.config.padding,
            &conv.config.auto_pad,
        );
        let value = self.append(
            Op::Conv2dOihw(Conv2dOptions {
                strides: cast2(conv.config.stride),
                padding,
                dilation: cast2(conv.config.dilation),
                groups: conv.config.groups as i64,
            }),
            &[ids[0], ids[1]],
            result,
        )?;
        self.optional_bias(value, args.get(2), ids.get(2), result, 3, node)
    }

    fn conv_transpose(
        &mut self,
        conv: &onnx_ir::node::conv_transpose2d::ConvTranspose2dNode,
        args: &[&onnx_ir::Argument],
        ids: &[SsaId],
        result: &TensorType,
        node: &Node,
    ) -> Result<SsaId> {
        if conv.config.groups != 1 {
            return Err(import_error(node, "grouped ConvTranspose2d"));
        }
        let weight_shape = self.shape_of_arg(args[1])?;
        let transposed_shape = vec![
            weight_shape[2],
            weight_shape[3],
            weight_shape[1],
            weight_shape[0],
        ];
        let weight = self.append(
            Op::Transpose {
                permutation: vec![2, 3, 1, 0],
            },
            &[ids[1]],
            &TensorType {
                dims: dims(&transposed_shape),
                dtype: DType::F32,
            },
        )?;
        let padding = [
            [conv.config.padding[0] as i64; 2],
            [conv.config.padding[1] as i64; 2],
        ];
        let value = self.append(
            Op::ConvTranspose2d(ConvTranspose2dOptions {
                strides: cast2(conv.config.stride),
                padding,
                dilation: cast2(conv.config.dilation),
                output_padding: cast2(conv.config.padding_out),
            }),
            &[ids[0], weight],
            result,
        )?;
        self.optional_bias(value, args.get(2), ids.get(2), result, 3, node)
    }

    fn resize(
        &mut self,
        resize: &onnx_ir::node::resize::ResizeNode,
        input: SsaId,
        input_name: &str,
        result: &TensorType,
        node: &Node,
    ) -> Result<SsaId> {
        if resize.config.mode != ResizeMode::Nearest
            || resize.config.coordinate_transformation_mode != CoordinateTransformMode::Asymmetric
            || resize.config.nearest_mode != NearestMode::Floor
        {
            return Err(import_error(
                node,
                "only asymmetric/floor nearest resize is currently canonicalized",
            ));
        }
        let old = self.shape(input_name)?.clone();
        let new: Vec<usize> = self.shape(&node.outputs()[0].name)?.clone();
        let scales = match (&resize.config.sizes, &resize.config.scales) {
            (Some(ResizeSizes::Static(sizes)), _) => [sizes[0] / old[2], sizes[1] / old[3]],
            (_, Some(ResizeScales::Static(scales))) => [scales[0] as usize, scales[1] as usize],
            _ => return Err(import_error(node, "runtime resize")),
        };
        if new[2] != old[2] * scales[0] || new[3] != old[3] * scales[1] {
            return Err(import_error(node, "non-integral nearest resize"));
        }
        let expanded = vec![old[0], old[2], 1, old[3], 1, old[1]];
        let reshaped = self.append(
            Op::Reshape,
            &[input],
            &TensorType {
                dims: dims(&expanded),
                dtype: DType::F32,
            },
        )?;
        let broadcast = vec![old[0], old[2], scales[0], old[3], scales[1], old[1]];
        let expanded = self.append(
            Op::Broadcast {
                axes: vec![0, 1, 2, 3, 4, 5],
            },
            &[reshaped],
            &TensorType {
                dims: dims(&broadcast),
                dtype: DType::F32,
            },
        )?;
        self.append(Op::Reshape, &[expanded], result)
    }

    fn hard_sigmoid(
        &mut self,
        alpha: f32,
        beta: f32,
        input: SsaId,
        result: &TensorType,
        node: &Node,
    ) -> Result<SsaId> {
        let alpha_scalar = self.constant_scalar(alpha)?;
        let beta_scalar = self.constant_scalar(beta)?;
        let zero_scalar = self.constant_scalar(0.0)?;
        let one_scalar = self.constant_scalar(1.0)?;
        let alpha = self.broadcast_id(alpha_scalar, &[], result, node)?;
        let beta = self.broadcast_id(beta_scalar, &[], result, node)?;
        let zero = self.broadcast_id(zero_scalar, &[], result, node)?;
        let one = self.broadcast_id(one_scalar, &[], result, node)?;
        let scaled = self.append(Op::Binary(Binary::Mul), &[input, alpha], result)?;
        let shifted = self.append(Op::Binary(Binary::Add), &[scaled, beta], result)?;
        let low = self.append(Op::Binary(Binary::Maximum), &[shifted, zero], result)?;
        self.append(Op::Binary(Binary::Minimum), &[low, one], result)
    }

    fn binary(
        &mut self,
        op: Binary,
        args: &[&onnx_ir::Argument],
        ids: &[SsaId],
        result: &TensorType,
        node: &Node,
    ) -> Result<SsaId> {
        let (lhs, lhs_shape) = self.physical_arg(args[0], ids[0])?;
        let (rhs, rhs_shape) = self.physical_arg(args[1], ids[1])?;
        let lhs = self.broadcast_id(lhs, &lhs_shape, result, node)?;
        let rhs = self.broadcast_id(rhs, &rhs_shape, result, node)?;
        self.append(Op::Binary(op), &[lhs, rhs], result)
    }

    fn optional_bias(
        &mut self,
        value: SsaId,
        arg: Option<&&onnx_ir::Argument>,
        bias: Option<&SsaId>,
        result: &TensorType,
        axis: usize,
        node: &Node,
    ) -> Result<SsaId> {
        let (Some(arg), Some(&bias)) = (arg, bias) else {
            return Ok(value);
        };
        let shape = self.shape_of_arg(arg)?;
        let bias = self.broadcast_id_at(bias, &shape, result, axis, node)?;
        self.append(Op::Binary(Binary::Add), &[value, bias], result)
    }

    fn resolve(&mut self, arg: &onnx_ir::Argument, node: &Node) -> Result<SsaId> {
        if let ValueSource::Static(data_id) = arg.value_source
            && let Some(&id) = self.static_values.get(&data_id)
        {
            return Ok(id);
        }
        if let Some(&id) = self.values.get(&arg.name) {
            return Ok(id);
        }
        let data = arg
            .value()
            .ok_or_else(|| import_error(node, format!("unresolved value `{}`", arg.name)))?;
        let shape = data.shape.clone();
        let values = data.to_f32_vec().map_err(|error| {
            import_error(node, format!("constant `{}` is not F32: {error}", arg.name))
        })?;
        let name = if let ValueSource::Static(data_id) = arg.value_source {
            format!("static:{data_id}")
        } else if arg.name.is_empty() {
            format!("{}:constant", node.name())
        } else {
            arg.name.clone()
        };
        let id = self.parameter(&name, &shape, Some(values.into()))?;
        if let ValueSource::Static(data_id) = arg.value_source {
            self.static_values.insert(data_id, id);
        }
        if !arg.name.is_empty() {
            self.values.insert(arg.name.clone(), id);
        }
        Ok(id)
    }

    fn parameter(
        &mut self,
        name: &str,
        shape: &[usize],
        values: Option<Arc<[f32]>>,
    ) -> Result<SsaId> {
        let number = self.inputs.len() + self.parameters.len();
        let id = self.append(
            Op::Parameter(number),
            &[],
            &TensorType {
                dims: dims(shape),
                dtype: DType::F32,
            },
        )?;
        if let Some(values) = values {
            self.parameters.push(Parameter {
                name: name.to_owned(),
                shape: shape.to_vec(),
                values,
            });
        }
        Ok(id)
    }

    fn constant_scalar(&mut self, value: f32) -> Result<SsaId> {
        self.append(
            Op::ConstantF32(Arc::from([value])),
            &[],
            &TensorType {
                dims: vec![],
                dtype: DType::F32,
            },
        )
    }

    fn broadcast_id(
        &mut self,
        id: SsaId,
        source: &[usize],
        result: &TensorType,
        node: &Node,
    ) -> Result<SsaId> {
        let rank = result.dims.len();
        let axes = (rank - source.len()..rank).collect();
        self.broadcast_with_axes(id, source, result, axes, node)
    }

    fn broadcast_id_at(
        &mut self,
        id: SsaId,
        source: &[usize],
        result: &TensorType,
        axis: usize,
        node: &Node,
    ) -> Result<SsaId> {
        self.broadcast_with_axes(id, source, result, vec![axis], node)
    }

    fn broadcast_with_axes(
        &mut self,
        id: SsaId,
        source: &[usize],
        result: &TensorType,
        axes: Vec<usize>,
        node: &Node,
    ) -> Result<SsaId> {
        let target: Vec<_> = result.dims.iter().map(|&x| x as usize).collect();
        if source == target {
            return Ok(id);
        }
        if source.len() != axes.len() {
            return Err(import_error(node, "invalid broadcast rank"));
        }
        self.append(Op::Broadcast { axes }, &[id], result)
    }

    fn append(&mut self, op: Op, operands: &[SsaId], result: &TensorType) -> Result<SsaId> {
        self.program
            .append(&op, operands, result)
            .context(crate::error::IrSnafu)
    }
    fn physical_ty(&self, name: &str) -> Result<TensorType> {
        Ok(TensorType {
            dims: dims(&self.physical_shape(name)?),
            dtype: DType::F32,
        })
    }
    fn physical_shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(physical_shape(self.shape(name)?))
    }
    fn physical_arg(&mut self, arg: &onnx_ir::Argument, id: SsaId) -> Result<(SsaId, Vec<usize>)> {
        let shape = self.shape_of_arg(arg)?;
        if arg.value().is_some() && shape.len() == 4 {
            Ok((self.transpose_to_nhwc(id, &shape)?, physical_shape(&shape)))
        } else if arg.value().is_none() {
            Ok((id, physical_shape(&shape)))
        } else {
            Ok((id, shape))
        }
    }
    fn transpose_to_nhwc(&mut self, id: SsaId, shape: &[usize]) -> Result<SsaId> {
        if shape.len() != 4 {
            return Ok(id);
        }
        self.append(
            Op::Transpose {
                permutation: vec![0, 2, 3, 1],
            },
            &[id],
            &TensorType {
                dims: dims(&physical_shape(shape)),
                dtype: DType::F32,
            },
        )
    }
    fn transpose_to_nchw(&mut self, id: SsaId, shape: Vec<usize>) -> Result<SsaId> {
        if shape.len() != 4 {
            return Ok(id);
        }
        self.append(
            Op::Transpose {
                permutation: vec![0, 3, 1, 2],
            },
            &[id],
            &TensorType {
                dims: dims(&shape),
                dtype: DType::F32,
            },
        )
    }
    fn shape(&self, name: &str) -> Result<&Vec<usize>> {
        self.shapes.get(name).ok_or_else(|| Error::MissingShape {
            node: "IR import".into(),
            operator: "Value".into(),
            value: name.into(),
        })
    }
    fn shape_of_arg(&self, arg: &onnx_ir::Argument) -> Result<Vec<usize>> {
        if let Some(data) = arg.value() {
            Ok(data.shape.to_vec())
        } else {
            Ok(self.shape(&arg.name)?.clone())
        }
    }
    fn value(&self, name: &str, (node, operator): (&str, &str)) -> Result<SsaId> {
        self.values
            .get(name)
            .copied()
            .ok_or_else(|| Error::MissingShape {
                node: node.into(),
                operator: operator.into(),
                value: name.into(),
            })
    }
}

fn pool_padding(
    input: &[usize],
    kernel: [usize; 2],
    strides: [usize; 2],
    dilation: [usize; 2],
    explicit: &PaddingConfig2d,
    auto: &AutoPad,
) -> [[i64; 2]; 2] {
    let [mut top, mut left, mut bottom, mut right] = match explicit {
        PaddingConfig2d::Valid => [0; 4],
        PaddingConfig2d::Explicit(t, l, b, r) => [*t, *l, *b, *r],
    };
    if matches!(auto, AutoPad::SameUpper | AutoPad::SameLower) {
        let total = |size: usize, kernel: usize, stride: usize, dilation: usize| {
            ((size.div_ceil(stride) - 1) * stride + dilation * (kernel - 1) + 1)
                .saturating_sub(size)
        };
        let ph = total(input[2], kernel[0], strides[0], dilation[0]);
        let pw = total(input[3], kernel[1], strides[1], dilation[1]);
        if *auto == AutoPad::SameLower {
            top = ph.div_ceil(2);
            left = pw.div_ceil(2);
        } else {
            top = ph / 2;
            left = pw / 2;
        }
        bottom = ph - top;
        right = pw - left;
    }
    [[top as i64, bottom as i64], [left as i64, right as i64]]
}
fn cast2(value: [usize; 2]) -> [i64; 2] {
    [value[0] as i64, value[1] as i64]
}
fn dims(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|&x| x as i64).collect()
}
fn physical_shape(shape: &[usize]) -> Vec<usize> {
    if let [n, c, h, w] = shape {
        vec![*n, *h, *w, *c]
    } else {
        shape.to_vec()
    }
}
fn physical_axis(axis: usize, rank: usize) -> usize {
    if rank == 4 { [0, 3, 1, 2][axis] } else { axis }
}
fn import_error(node: &Node, message: impl Into<String>) -> Error {
    Error::Import {
        node: node.name().into(),
        operator: node.node_type().to_string(),
        message: message.into(),
    }
}
fn node_label<'a>(node: &'a str, operator: &'a str) -> (&'a str, &'a str) {
    (node, operator)
}
