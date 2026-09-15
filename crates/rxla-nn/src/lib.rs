//! Scoped parameter effects for tensor tracing.
//!
//! A [`Cx`] is deliberately not a global variable store. `init` and `apply`
//! interpret the same `param` calls differently, so model code is written once
//! without making parameter identity depend on call order.

use rxla_core::{DType, Graph, Tensor};
use snafu::{OptionExt, Snafu, ensure};
use std::collections::{BTreeMap, HashSet};

mod applied;
pub use applied::{AppliedModel, BoundParameters, ModelArguments};
mod layers;
pub use layers::{
    Conv2d, Embedding, GroupNorm, LayerNorm, Linear, Named, QuantizedLinear, RmsNorm,
};
mod schema;
pub use schema::{ModelArgument, ModelInputSpec, ParamSchema, ParameterSpec};
mod selection;
pub use selection::{ParameterId, ParameterSelection};

/// A reusable effect-based model definition.
///
/// It captures one ordinary Rust function so callers do not repeat closures
/// around schema discovery and application. Compiled execution artifacts use
/// [`rxla_core::Program`]; keeping the names distinct avoids import aliases in
/// applications that construct and run models in the same module.
pub struct Model<F> {
    build: F,
}

/// Recoverable parameter-effect and model-binding failures.
#[derive(Debug, Snafu)]
#[non_exhaustive]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
    #[snafu(transparent)]
    Pjrt { source: rxla_core::PjrtError },
    #[snafu(display("invalid model name segment {name:?}: expected nonempty and dot-free"))]
    InvalidName { name: String },
    #[snafu(display("parameter {path:?} has a negative dimension"))]
    NegativeParameterDimension { path: String },
    #[snafu(display("parameter dtype {dtype:?} is unsupported"))]
    UnsupportedParameterDType { dtype: DType },
    #[snafu(display("model input dtype {dtype:?} is unsupported"))]
    UnsupportedInputDType { dtype: DType },
    #[snafu(display("parameter declaration for {path:?} is incompatible with the schema"))]
    IncompatibleParameter { path: String },
    #[snafu(display("parameter {path:?} is absent from the schema"))]
    UnknownParameter { path: String },
    #[snafu(display("model effect at index {index} is incompatible with the schema"))]
    EffectMismatch { index: usize },
    #[snafu(display("model input {index} is absent from the schema"))]
    UnexpectedInput { index: usize },
    #[snafu(display("model input declaration at index {index} is incompatible with the schema"))]
    IncompatibleInput { index: usize },
    #[snafu(display("apply did not read schema parameter {path:?}"))]
    UnreadParameter { path: String },
    #[snafu(display("apply stopped before schema input {index}"))]
    UnreadInput { index: usize },
    #[snafu(display("apply stopped before schema effect {index}"))]
    UnreadEffect { index: usize },
    #[snafu(display("duplicate parameter binding {path:?}"))]
    DuplicateBinding { path: String },
    #[snafu(display("missing parameter binding {path:?}"))]
    MissingBinding { path: String },
    #[snafu(display("model expected {expected} inputs, received {actual}"))]
    InputCount { expected: usize, actual: usize },
    #[snafu(display("model expected {expected} bound parameters, received {actual}"))]
    ParameterCount { expected: usize, actual: usize },
    #[snafu(display("parameter selection belongs to a different model schema"))]
    SelectionSchemaMismatch,
    #[snafu(display("{kind} {identity}: shape does not match the schema"))]
    BufferShape {
        kind: &'static str,
        identity: String,
    },
    #[snafu(display("{kind} {identity}: dtype does not match the schema"))]
    BufferDType {
        kind: &'static str,
        identity: String,
    },
    #[snafu(display("{layer} requires {requirement}"))]
    InvalidLayerInput {
        layer: &'static str,
        requirement: &'static str,
    },
    #[snafu(display("invalid model definition: {message}"))]
    InvalidDefinition { message: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl<F> Model<F> {
    pub fn new(build: F) -> Self {
        Self { build }
    }
}

impl<F, T> Model<F>
where
    F: Fn(&mut Cx) -> Result<T>,
    T: TraceOutputs,
{
    /// Discover parameter effects and produce the executable model trace.
    ///
    /// This is the ordinary one-call path. [`Self::init`] and [`Self::apply`]
    /// remain available when callers need to inspect or restore a schema
    /// between the two interpretations.
    pub fn trace(&self) -> Result<(ParamSchema, AppliedModel)> {
        let schema = self.init()?;
        let applied = self.apply(&schema)?;
        Ok((schema, applied))
    }

    /// Discover the input/parameter effect schema from this model body.
    pub fn init(&self) -> Result<ParamSchema> {
        init(|cx| (self.build)(cx)).map(|(schema, _)| schema)
    }

    /// Trace this model body against a previously discovered schema.
    pub fn apply(&self, schema: &ParamSchema) -> Result<AppliedModel> {
        apply(schema, |cx| (self.build)(cx))
    }
}

enum ParamMode {
    Init {
        schema: ParamSchema,
        values: BTreeMap<String, Tensor>,
    },
    Apply {
        schema: ParamSchema,
        values: BTreeMap<String, Tensor>,
        read: HashSet<String>,
    },
}

/// The explicit interpreter for scoped parameter effects.
///
/// Model functions receive this same type during [`init`] and [`apply`]. The
/// mode is selected by the caller, never inferred from prior invocations.
pub struct Cx {
    graph: Graph,
    scope: Vec<String>,
    input_index: usize,
    effect_index: usize,
    mode: ParamMode,
}

/// Values a model trace may return from [`apply`].
///
/// A single [`Tensor`] is the common case; vectors and fixed arrays retain
/// multi-output programs without forcing every single-output model to allocate
/// or spell `vec![output]`.
pub trait TraceOutputs {
    fn into_outputs(self) -> Vec<Tensor>;
}

impl TraceOutputs for Tensor {
    fn into_outputs(self) -> Vec<Tensor> {
        vec![self]
    }
}

impl TraceOutputs for Vec<Tensor> {
    fn into_outputs(self) -> Vec<Tensor> {
        self
    }
}

impl<const N: usize> TraceOutputs for [Tensor; N] {
    fn into_outputs(self) -> Vec<Tensor> {
        self.into()
    }
}

impl Cx {
    fn init() -> Self {
        Self {
            graph: Graph::default(),
            scope: Vec::new(),
            input_index: 0,
            effect_index: 0,
            mode: ParamMode::Init {
                schema: ParamSchema::default(),
                values: BTreeMap::new(),
            },
        }
    }

    fn apply(schema: ParamSchema) -> Self {
        Self {
            graph: Graph::default(),
            scope: Vec::new(),
            input_index: 0,
            effect_index: 0,
            mode: ParamMode::Apply {
                schema,
                values: BTreeMap::new(),
                read: HashSet::new(),
            },
        }
    }

    /// Declare/read an F32 parameter at the current lexical scope.
    pub fn param(&mut self, name: &str, shape: &[i64]) -> Result<Tensor> {
        self.param_dtype(name, shape, DType::F32)
    }

    /// Declare/read a parameter with its storage dtype.
    ///
    /// The current symbolic tensor surface supports F32 parameters, frozen
    /// BF16 storage exposed as F32 computation values, and raw U8 storage for
    /// explicitly dequantized inference layers. Other dtypes are rejected.
    pub fn param_dtype(&mut self, name: &str, shape: &[i64], dtype: DType) -> Result<Tensor> {
        validate_name(name)?;
        let path = self.path(name);
        ensure!(
            shape.iter().all(|&dim| dim >= 0),
            NegativeParameterDimensionSnafu { path }
        );
        let requested = ParameterSpec {
            path: path.clone(),
            shape: shape.to_vec(),
            dtype,
        };
        let (graph, mode) = (&mut self.graph, &mut self.mode);
        match mode {
            ParamMode::Init { schema, values } => {
                if let Some(existing) = schema.get(&path) {
                    ensure!(existing == &requested, IncompatibleParameterSnafu { path });
                    return Ok(values
                        .get(&path)
                        .expect("schema and parameter value are inserted together")
                        .clone());
                }
                let value = parameter_tensor(graph, shape, dtype)?;
                let index = schema.parameters.len();
                schema.parameters.push(requested);
                schema.indices.insert(path.clone(), index);
                schema.arguments.push(ModelArgument::Parameter(index));
                self.effect_index += 1;
                values.insert(path, value.clone());
                Ok(value)
            }
            ParamMode::Apply {
                schema,
                values,
                read,
            } => {
                let expected = schema
                    .get(&path)
                    .with_context(|| UnknownParameterSnafu { path: path.clone() })?;
                ensure!(expected == &requested, IncompatibleParameterSnafu { path });
                read.insert(path.clone());
                if let Some(value) = values.get(&path) {
                    return Ok(value.clone());
                }
                let parameter_index = *schema
                    .indices
                    .get(&path)
                    .expect("every parameter schema entry has an index");
                ensure!(
                    schema.arguments.get(self.effect_index)
                        == Some(&ModelArgument::Parameter(parameter_index)),
                    EffectMismatchSnafu {
                        index: self.effect_index
                    }
                );
                let value = parameter_tensor(graph, shape, dtype)?;
                values.insert(path, value.clone());
                self.effect_index += 1;
                Ok(value)
            }
        }
    }

    /// Enter one lexical parameter scope for the duration of `build`.
    pub fn scope<T>(
        &mut self,
        name: &str,
        build: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        validate_name(name)?;
        self.scope.push(name.to_owned());
        let result = build(self);
        self.scope.pop();
        result
    }

    /// Create a visible F32 model input in this trace.
    pub fn input(&mut self, shape: &[i64]) -> Result<Tensor> {
        self.input_dtype(shape, DType::F32)
    }

    /// Create a visible input with an explicit dtype.
    pub fn input_dtype(&mut self, shape: &[i64], dtype: DType) -> Result<Tensor> {
        let requested = ModelInputSpec {
            shape: shape.to_vec(),
            dtype,
        };
        let input_index = self.input_index;
        match &mut self.mode {
            ParamMode::Init { schema, .. } => {
                schema.inputs.push(requested.clone());
                schema.arguments.push(ModelArgument::Input(input_index));
            }
            ParamMode::Apply { schema, .. } => {
                let expected = schema
                    .inputs
                    .get(input_index)
                    .context(UnexpectedInputSnafu { index: input_index })?;
                ensure!(
                    expected == &requested,
                    IncompatibleInputSnafu { index: input_index }
                );
                ensure!(
                    schema.arguments.get(self.effect_index)
                        == Some(&ModelArgument::Input(input_index)),
                    EffectMismatchSnafu {
                        index: self.effect_index
                    }
                );
            }
        }
        self.input_index += 1;
        self.effect_index += 1;
        match dtype {
            DType::F32 | DType::I32 => Ok(self.graph.input_dtype(shape, dtype)?),
            DType::BF16 => Ok(self.graph.input_bf16_as_f32(shape)?),
            _ => UnsupportedInputDTypeSnafu { dtype }.fail(),
        }
    }

    /// Create a graph-local I32 coordinate tensor without adding an ABI input.
    pub fn iota_i32(&self, shape: &[i64], axis: usize) -> Result<Tensor> {
        Ok(self.graph.iota_i32(shape, axis)?)
    }

    /// Create a graph-local F32 constant without adding an ABI input.
    pub fn constant(&self, shape: &[i64], values: &[f32]) -> Result<Tensor> {
        Ok(self.graph.constant(shape, values)?)
    }

    fn path(&self, name: &str) -> String {
        if self.scope.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{}", self.scope.join("."), name)
        }
    }

    fn finish_apply(&self) -> Result<()> {
        let ParamMode::Apply { schema, read, .. } = &self.mode else {
            return Ok(());
        };
        if let Some(missing) = schema
            .parameters()
            .iter()
            .find(|parameter| !read.contains(parameter.path()))
        {
            return UnreadParameterSnafu {
                path: missing.path(),
            }
            .fail();
        }
        ensure!(
            self.input_index == schema.inputs.len(),
            UnreadInputSnafu {
                index: self.input_index,
            }
        );
        ensure!(
            self.effect_index == schema.arguments.len(),
            UnreadEffectSnafu {
                index: self.effect_index,
            }
        );
        Ok(())
    }

    fn parameter_tensors(&self) -> Vec<Tensor> {
        let ParamMode::Apply { schema, values, .. } = &self.mode else {
            unreachable!("only apply contexts expose parameter tensors")
        };
        schema
            .parameters()
            .iter()
            .map(|parameter| {
                values
                    .get(parameter.path())
                    .expect("finish_apply verified every schema parameter")
                    .clone()
            })
            .collect()
    }

    fn into_schema(self) -> ParamSchema {
        match self.mode {
            ParamMode::Init { schema, .. } => schema,
            ParamMode::Apply { .. } => unreachable!("only init contexts produce schemas"),
        }
    }
}

fn parameter_tensor(graph: &Graph, shape: &[i64], dtype: DType) -> Result<Tensor> {
    match dtype {
        DType::F32 => Ok(graph.input(shape)?),
        DType::BF16 => Ok(graph.input_bf16_as_f32(shape)?),
        DType::U8 => Ok(graph.input_dtype(shape, DType::U8)?),
        _ => UnsupportedParameterDTypeSnafu { dtype }.fail(),
    }
}

fn validate_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty() && !name.contains('.'),
        InvalidNameSnafu { name }
    );
    Ok(())
}

/// Interpret parameter effects as declarations and return the frozen schema.
pub fn init<T>(build: impl FnOnce(&mut Cx) -> Result<T>) -> Result<(ParamSchema, T)> {
    let mut cx = Cx::init();
    let result = build(&mut cx)?;
    Ok((cx.into_schema(), result))
}

/// Interpret parameter effects as reads from `schema` and retain traced outputs.
pub fn apply<T: TraceOutputs>(
    schema: &ParamSchema,
    build: impl FnOnce(&mut Cx) -> Result<T>,
) -> Result<AppliedModel> {
    let mut cx = Cx::apply(schema.clone());
    let outputs = build(&mut cx)?.into_outputs();
    cx.finish_apply()?;
    let parameters = cx.parameter_tensors();
    Ok(AppliedModel::new(
        cx.graph,
        outputs,
        parameters,
        schema.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{Buffer, CacheLimits, Client, ClientOptions, Compiler, Conv2dOptions};

    fn classifier(cx: &mut Cx) -> Result<Tensor> {
        let input = cx.input(&[2, 4])?;
        cx.scope("head", |cx| {
            let weight = cx.param("weight", &[4, 3])?;
            Ok(input.matmul(&weight)?)
        })
    }

    #[test]
    fn init_and_apply_share_one_model_body_and_schema() {
        let (schema, init_output) = init(classifier).unwrap();
        assert_eq!(init_output.shape(), [2, 3]);
        assert_eq!(schema.inputs().len(), 1);
        assert_eq!(schema.inputs()[0].shape(), [2, 4]);
        assert_eq!(schema.inputs()[0].dtype(), DType::F32);
        assert_eq!(schema.parameters().len(), 1);
        assert_eq!(schema.parameters()[0].path(), "head.weight");
        assert_eq!(schema.parameters()[0].shape(), [4, 3]);
        assert_eq!(
            schema.arguments(),
            &[ModelArgument::Input(0), ModelArgument::Parameter(0)]
        );

        let applied = apply(&schema, classifier).unwrap();
        assert_eq!(applied.outputs()[0].shape(), [2, 3]);
        assert_eq!(applied.prepare().unwrap().input_count(), 2);
    }

    #[test]
    fn selected_parameter_tensors_drive_partial_autodiff() {
        fn product(cx: &mut Cx) -> Result<Tensor> {
            let body = cx.scope("body", |cx| cx.param("weight", &[2]))?;
            let head = cx.scope("head", |cx| cx.param("weight", &[2]))?;
            Ok(body.mul(&head)?.sum(&[0], false)?)
        }

        let (schema, _) = init(product).unwrap();
        let applied = apply(&schema, product).unwrap();
        let head = schema.select_under("head");
        let leaves = applied.parameter_tensors(&head).unwrap();
        let gradients = applied.outputs()[0].grad(&leaves).unwrap();

        assert_eq!(leaves.len(), 1);
        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), [2]);
    }

    #[test]
    fn selected_parameter_tensors_reject_another_schema() {
        let (schema, _) = init(|cx| cx.param("weight", &[2])).unwrap();
        let applied = apply(&schema, |cx| cx.param("weight", &[2])).unwrap();
        let (other, _) = init(|cx| cx.param("weight", &[3])).unwrap();
        let error = match applied.parameter_tensors(&other.select_all()) {
            Ok(_) => panic!("a structurally different schema was accepted"),
            Err(error) => error,
        };
        assert!(matches!(error, Error::SelectionSchemaMismatch));
    }

    #[test]
    fn apply_rejects_changed_or_missing_parameter_effects() {
        let (schema, _) = init(|cx| cx.param("weight", &[2, 3])).unwrap();
        let changed = match apply(&schema, |cx| {
            cx.param("weight", &[3, 2])?;
            Ok(vec![])
        }) {
            Ok(_) => panic!("changed parameter shape unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(changed, Error::IncompatibleParameter { .. }));

        let missing = match apply(&schema, |_cx| Ok(vec![])) {
            Ok(_) => panic!("missing parameter effect unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(missing, Error::UnreadParameter { .. }));
    }

    #[test]
    fn repeated_parameter_reads_share_one_input() {
        let (schema, _) = init(|cx| {
            let first = cx.param("weight", &[2, 2])?;
            let second = cx.param("weight", &[2, 2])?;
            Ok(first.add(&second)?)
        })
        .unwrap();
        let applied = apply(&schema, |cx| {
            let first = cx.param("weight", &[2, 2])?;
            let second = cx.param("weight", &[2, 2])?;
            Ok(vec![first.add(&second)?])
        })
        .unwrap();
        assert_eq!(applied.prepare().unwrap().input_count(), 1);
    }

    #[test]
    fn apply_rejects_changed_or_extra_input_effects() {
        let (schema, _) = init(classifier).unwrap();
        let changed = match apply(&schema, |cx| {
            let input = cx.input(&[3, 4])?;
            let weight = cx.scope("head", |cx| cx.param("weight", &[4, 3]))?;
            Ok(input.matmul(&weight)?)
        }) {
            Ok(_) => panic!("changed input ABI unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(changed, Error::IncompatibleInput { index: 0 }));

        let extra = match apply(&schema, |cx| {
            let input = cx.input(&[2, 4])?;
            let _unused = cx.input(&[1])?;
            let weight = cx.scope("head", |cx| cx.param("weight", &[4, 3]))?;
            Ok(input.matmul(&weight)?)
        }) {
            Ok(_) => panic!("extra input ABI unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(extra, Error::UnexpectedInput { index: 1 }));
    }

    #[test]
    fn apply_rejects_effect_reordering() {
        let (schema, _) = init(classifier).unwrap();
        let error = match apply(&schema, |cx| {
            let weight = cx.scope("head", |cx| cx.param("weight", &[4, 3]))?;
            let input = cx.input(&[2, 4])?;
            Ok(input.matmul(&weight)?)
        }) {
            Ok(_) => panic!("reordered effects unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(error, Error::EffectMismatch { index: 0 }));
    }

    #[test]
    fn bind_rejects_wrong_number_of_model_inputs_before_buffer_access() {
        let (schema, _) = init(classifier).unwrap();
        let applied = apply(&schema, classifier).unwrap();
        let error = match applied.bind(&[], std::iter::empty::<(&str, &Buffer)>()) {
            Ok(_) => panic!("missing model input unexpectedly bound"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            Error::InputCount {
                expected: 1,
                actual: 0
            }
        ));
    }

    #[test]
    fn linear_infers_input_features_at_its_use_site() {
        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 4])?;
            cx.named("head")?.linear(3).apply(&input)
        })
        .unwrap();
        assert_eq!(output.shape(), [2, 3]);
        assert_eq!(schema.parameters()[0].path(), "head.weight");
        assert_eq!(schema.parameters()[0].shape(), [3, 4]);
        assert_eq!(schema.parameters()[1].path(), "head.bias");
        assert_eq!(schema.parameters()[1].shape(), [3]);
    }

    #[test]
    fn program_owns_the_model_body_and_layer_builders_own_scopes() {
        let model = Model::new(|cx: &mut Cx| {
            let input = cx.input(&[2, 4])?;
            let hidden = cx
                .named("hidden")?
                .linear(8)
                .bias(false)
                .apply(&input)?
                .relu()?;
            cx.named("head")?.linear(3).apply(&hidden)
        });

        let (schema, applied) = model.trace().unwrap();

        assert_eq!(schema.parameters()[0].path(), "hidden.weight");
        assert_eq!(schema.parameters()[0].shape(), [8, 4]);
        assert_eq!(schema.parameters()[1].path(), "head.weight");
        assert_eq!(schema.parameters()[1].shape(), [3, 8]);
        assert_eq!(schema.parameters()[2].path(), "head.bias");
        assert_eq!(applied.outputs()[0].shape(), [2, 3]);
    }

    #[test]
    fn conv2d_infers_checkpoint_kernel_input_channels() {
        let options = Conv2dOptions {
            padding: [[1, 1], [1, 1]],
            groups: 2,
            ..Default::default()
        };
        let model = |cx: &mut Cx| {
            let input = cx.input(&[1, 8, 8, 4])?;
            cx.scope("conv_in", |cx| {
                cx.apply_conv2d(&input, 6, [3, 3], options, true)
            })
        };
        let (schema, output) = init(model).unwrap();
        assert_eq!(output.shape(), [1, 8, 8, 6]);
        assert_eq!(schema.parameters()[0].path(), "conv_in.weight");
        assert_eq!(schema.parameters()[0].shape(), [6, 2, 3, 3]);
        assert_eq!(schema.parameters()[1].path(), "conv_in.bias");
        assert_eq!(schema.parameters()[1].shape(), [6]);

        let applied = apply(&schema, model).unwrap();
        let lowered = applied.prepare().unwrap();
        assert!(
            std::str::from_utf8(lowered.code())
                .unwrap()
                .contains("stablehlo.convolution")
        );
    }

    #[test]
    fn group_norm_nhwc_infers_affine_channel_shape() {
        let model = |cx: &mut Cx| {
            let input = cx.input(&[1, 8, 8, 32])?;
            cx.scope("norm", |cx| cx.apply_group_norm_nhwc(&input, 8, 1e-5, true))
        };
        let (schema, output) = init(model).unwrap();
        assert_eq!(output.shape(), [1, 8, 8, 32]);
        assert_eq!(schema.parameters()[0].path(), "norm.weight");
        assert_eq!(schema.parameters()[0].shape(), [32]);
        assert_eq!(schema.parameters()[1].path(), "norm.bias");
        assert_eq!(schema.parameters()[1].shape(), [32]);
        apply(&schema, model).unwrap().prepare().unwrap();
    }

    #[test]
    fn layer_norm_infers_trailing_affine_shape() {
        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 7, 32])?;
            cx.scope("norm", |cx| cx.apply_layer_norm(&input, 1, 1e-5, true))
        })
        .unwrap();
        assert_eq!(output.shape(), [2, 7, 32]);
        assert_eq!(schema.parameters()[0].shape(), [32]);
        assert_eq!(schema.parameters()[1].shape(), [32]);
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH and PJRT_CUDA_PLUGIN_PATH"]
    fn parameter_effect_model_executes_on_cpu_and_cuda() {
        let cpu = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .expect("load CPU plugin");
        let cuda_options = ClientOptions::new().set("preallocate", false);
        let cuda = unsafe {
            Client::load_with_options(
                std::env::var("PJRT_CUDA_PLUGIN_PATH").expect("CUDA plugin path"),
                &cuda_options,
            )
        }
        .expect("load CUDA plugin");

        let model = |cx: &mut Cx| {
            let input = cx.input(&[2, 3])?;
            Ok(cx
                .scope("head", |cx| cx.apply_linear(&input, 2, false))?
                .relu()?)
        };
        let (schema, _) = init(model).expect("initialize model schema");
        let applied = apply(&schema, model).expect("trace model application");

        for client in [cpu, cuda] {
            let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
            let executable = applied.compile(&mut compiler).expect("compile model");
            let input = client
                .buffer(&[2, 3], &[1., -2., 3., -4., 5., -6.])
                .expect("upload input");
            let weight = client
                .buffer(&[2, 3], &[1., 3., 5., 2., 4., 6.])
                .expect("upload parameter");
            let arguments = applied
                .bind(&[&input], [("head.weight", &weight)])
                .expect("bind model arguments");
            let outputs = executable
                .execute(arguments.as_slice())
                .expect("execute model");
            assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [10., 12., 0., 0.]);
        }
    }
}
