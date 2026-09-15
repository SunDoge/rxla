use super::*;

macro_rules! verify_exact_bytes {
    ($ty:ty, $size:expr, $name:literal) => {
        impl Verify for $ty {
            fn verify(&self, _: &Context) -> pliron::result::Result<()> {
                if self.bytes.as_ref().len() != $size {
                    return pliron::verify_err_noloc!(
                        "{}",
                        concat!("malformed ", $name, " attribute")
                    );
                }
                Ok(())
            }
        }
    };
}

macro_rules! verify_multiple_bytes {
    ($ty:ty, $size:expr, $name:literal) => {
        impl Verify for $ty {
            fn verify(&self, _: &Context) -> pliron::result::Result<()> {
                if self.bytes.as_ref().len() % $size != 0 {
                    return pliron::verify_err_noloc!(
                        "{}",
                        concat!("malformed ", $name, " attribute")
                    );
                }
                Ok(())
            }
        }
    };
}

verify_multiple_bytes!(ShapeAttr, 8, "shape");
verify_exact_bytes!(ElementTypeAttr, 4, "element type");
verify_exact_bytes!(Conv2dOptionsAttr, 72, "conv2d options");
verify_exact_bytes!(ConvTranspose2dOptionsAttr, 80, "conv transpose options");
verify_exact_bytes!(Pool2dOptionsAttr, 64, "pool2d options");
verify_multiple_bytes!(AxesAttr, 8, "axes");
verify_exact_bytes!(AxisAttr, 8, "axis");
verify_multiple_bytes!(SliceSpecAttr, 24, "slice specification");
verify_multiple_bytes!(PaddingAttr, 16, "padding");
verify_exact_bytes!(BatchRankAttr, 8, "batch rank");
verify_exact_bytes!(AttentionScaleAttr, 4, "attention scale");

impl Verify for ComparisonAttr {
    fn verify(&self, _: &Context) -> pliron::result::Result<()> {
        if !matches!(self.bytes.as_ref().as_slice(), [0..=5]) {
            return pliron::verify_err_noloc!("malformed comparison attribute");
        }
        Ok(())
    }
}

impl Verify for IntegerBinaryAttr {
    fn verify(&self, _: &Context) -> pliron::result::Result<()> {
        if !matches!(self.bytes.as_ref().as_slice(), [0..=5]) {
            return pliron::verify_err_noloc!("malformed integer binary attribute");
        }
        Ok(())
    }
}

impl Verify for GatherGradientAttr {
    fn verify(&self, _: &Context) -> pliron::result::Result<()> {
        let bytes = self.bytes.as_ref();
        if bytes.len() != 9 || !matches!(bytes[8], 0 | 1) {
            return pliron::verify_err_noloc!("malformed gather gradient attribute");
        }
        Ok(())
    }
}

impl Verify for SortAttr {
    fn verify(&self, _: &Context) -> pliron::result::Result<()> {
        let bytes = self.bytes.as_ref();
        if bytes.len() != 9 || !matches!(bytes[8], 0 | 1) {
            return pliron::verify_err_noloc!("malformed sort attribute");
        }
        Ok(())
    }
}

impl Verify for ShardingAttr {
    fn verify(&self, _: &Context) -> pliron::result::Result<()> {
        if Sharding::decode_ir(self.bytes.as_ref()).is_err() {
            return pliron::verify_err_noloc!("malformed sharding attribute");
        }
        Ok(())
    }
}
pub(super) fn sharding_attr_key() -> Identifier {
    "rxla_sharding"
        .try_into()
        .expect("static sharding attribute name is valid")
}

pub(super) fn supported_dtype(dtype: DType) -> Option<()> {
    match dtype {
        DType::U8 | DType::F16 | DType::F32 | DType::I32 | DType::BF16 => Some(()),
        _ => None,
    }
}

impl AttentionScaleAttr {
    pub(super) fn new(scale: f32) -> Self {
        Self {
            bytes: BytesAttr::new(scale.to_le_bytes().to_vec()),
        }
    }

    pub(super) fn value(&self) -> f32 {
        f32::from_le_bytes(
            self.bytes
                .as_ref()
                .as_slice()
                .try_into()
                .expect("malformed attention scale attribute"),
        )
    }
}

impl AxesAttr {
    pub(super) fn new(axes: &[usize]) -> Self {
        Self {
            bytes: BytesAttr::new(
                axes.iter()
                    .flat_map(|&axis| (axis as u64).to_le_bytes())
                    .collect(),
            ),
        }
    }

    pub(super) fn values(&self) -> Vec<usize> {
        let (axes, remainder) = self.bytes.as_ref().as_chunks::<8>();
        assert!(remainder.is_empty(), "malformed axes attribute");
        axes.iter()
            .map(|&bytes| u64::from_le_bytes(bytes) as usize)
            .collect()
    }
}

impl AxisAttr {
    pub(super) fn new(axis: usize) -> Self {
        Self {
            bytes: BytesAttr::new((axis as u64).to_le_bytes().to_vec()),
        }
    }

    pub(super) fn value(&self) -> usize {
        u64::from_le_bytes(
            self.bytes
                .as_ref()
                .as_slice()
                .try_into()
                .expect("malformed axis attribute"),
        ) as usize
    }
}

impl SliceSpecAttr {
    pub(super) fn new(spec: &[SliceAxis]) -> Self {
        Self {
            bytes: BytesAttr::new(
                spec.iter()
                    .flat_map(|axis| [axis.start, axis.limit, axis.stride])
                    .flat_map(i64::to_le_bytes)
                    .collect(),
            ),
        }
    }

    pub(super) fn values(&self) -> Vec<SliceAxis> {
        let (values, remainder) = self.bytes.as_ref().as_chunks::<24>();
        assert!(remainder.is_empty(), "malformed slice attribute");
        values
            .iter()
            .map(|value| SliceAxis {
                start: i64::from_le_bytes(value[0..8].try_into().unwrap()),
                limit: i64::from_le_bytes(value[8..16].try_into().unwrap()),
                stride: i64::from_le_bytes(value[16..24].try_into().unwrap()),
            })
            .collect()
    }
}

impl PaddingAttr {
    pub(super) fn new(padding: &[[i64; 2]]) -> Self {
        Self {
            bytes: BytesAttr::new(
                padding
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
        }
    }

    pub(super) fn values(&self) -> Vec<[i64; 2]> {
        let (values, remainder) = self.bytes.as_ref().as_chunks::<16>();
        assert!(remainder.is_empty(), "malformed padding attribute");
        values
            .iter()
            .map(|value| {
                [
                    i64::from_le_bytes(value[0..8].try_into().unwrap()),
                    i64::from_le_bytes(value[8..16].try_into().unwrap()),
                ]
            })
            .collect()
    }
}

impl ComparisonAttr {
    pub(super) fn new(comparison: Comparison) -> Self {
        let value = match comparison {
            Comparison::Equal => 0,
            Comparison::NotEqual => 1,
            Comparison::Less => 2,
            Comparison::LessEqual => 3,
            Comparison::Greater => 4,
            Comparison::GreaterEqual => 5,
        };
        Self {
            bytes: BytesAttr::new(vec![value]),
        }
    }

    pub(super) fn direction(&self) -> &'static str {
        match self.bytes.as_ref().as_slice() {
            [0] => "EQ",
            [1] => "NE",
            [2] => "LT",
            [3] => "LE",
            [4] => "GT",
            [5] => "GE",
            _ => panic!("malformed comparison attribute"),
        }
    }

    pub(super) fn value(&self) -> Comparison {
        match self.bytes.as_ref().as_slice() {
            [0] => Comparison::Equal,
            [1] => Comparison::NotEqual,
            [2] => Comparison::Less,
            [3] => Comparison::LessEqual,
            [4] => Comparison::Greater,
            [5] => Comparison::GreaterEqual,
            _ => panic!("malformed comparison attribute"),
        }
    }
}

impl IntegerBinaryAttr {
    pub(super) fn new(operation: IntegerBinary) -> Self {
        let value = match operation {
            IntegerBinary::And => 0,
            IntegerBinary::Or => 1,
            IntegerBinary::Xor => 2,
            IntegerBinary::ShiftLeft => 3,
            IntegerBinary::ShiftRightLogical => 4,
            IntegerBinary::ShiftRightArithmetic => 5,
        };
        Self {
            bytes: BytesAttr::new(vec![value]),
        }
    }

    pub(super) fn opcode(&self) -> &'static str {
        match self.bytes.as_ref().as_slice() {
            [0] => "and",
            [1] => "or",
            [2] => "xor",
            [3] => "shift-left",
            [4] => "shift-right-logical",
            [5] => "shift-right-arithmetic",
            _ => panic!("malformed integer binary attribute"),
        }
    }

    pub(super) fn value(&self) -> IntegerBinary {
        match self.bytes.as_ref().as_slice() {
            [0] => IntegerBinary::And,
            [1] => IntegerBinary::Or,
            [2] => IntegerBinary::Xor,
            [3] => IntegerBinary::ShiftLeft,
            [4] => IntegerBinary::ShiftRightLogical,
            [5] => IntegerBinary::ShiftRightArithmetic,
            _ => panic!("malformed integer binary attribute"),
        }
    }
}

impl ShapeAttr {
    pub(super) fn new(dims: &[i64]) -> Self {
        Self {
            bytes: BytesAttr::new(dims.iter().flat_map(|dim| dim.to_le_bytes()).collect()),
        }
    }

    pub(super) fn values(&self) -> Vec<i64> {
        let (dims, remainder) = self.bytes.as_ref().as_chunks::<8>();
        assert!(remainder.is_empty(), "malformed shape attribute");
        dims.iter().map(|&dim| i64::from_le_bytes(dim)).collect()
    }
}

impl ElementTypeAttr {
    pub(super) fn new(dtype: DType) -> Self {
        Self {
            bytes: BytesAttr::new(dtype.as_raw().to_le_bytes().to_vec()),
        }
    }

    pub(super) fn value(&self) -> DType {
        DType::from_raw(u32::from_le_bytes(
            self.bytes
                .as_ref()
                .as_slice()
                .try_into()
                .expect("malformed element type attribute"),
        ))
    }
}

impl BatchRankAttr {
    pub(super) fn new(value: usize) -> Self {
        Self {
            bytes: BytesAttr::new((value as u64).to_le_bytes().to_vec()),
        }
    }

    pub(super) fn value(&self) -> usize {
        usize::try_from(u64::from_le_bytes(
            self.bytes
                .as_ref()
                .as_slice()
                .try_into()
                .expect("malformed batch rank attribute"),
        ))
        .expect("batch rank attribute overflows usize")
    }
}

impl GatherGradientAttr {
    pub(super) fn new(axis: usize, batched: bool) -> Self {
        let mut bytes = (axis as u64).to_le_bytes().to_vec();
        bytes.push(u8::from(batched));
        Self {
            bytes: BytesAttr::new(bytes),
        }
    }

    pub(super) fn values(&self) -> (usize, bool) {
        let bytes = self.bytes.as_ref();
        assert_eq!(bytes.len(), 9, "malformed gather gradient attribute");
        let axis = usize::try_from(u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .expect("malformed gather gradient attribute"),
        ))
        .expect("gather gradient axis overflows usize");
        let batched = match bytes.get(8) {
            Some(0) => false,
            Some(1) => true,
            _ => panic!("malformed gather gradient attribute"),
        };
        (axis, batched)
    }
}

impl ShardingAttr {
    pub(super) fn new(sharding: &Sharding) -> Result<Self> {
        Ok(Self {
            bytes: BytesAttr::new(sharding.encode_ir()?),
        })
    }

    pub(super) fn sharding(&self) -> Sharding {
        Sharding::decode_ir(self.bytes.as_ref()).expect("verified sharding attribute must decode")
    }
}

impl Conv2dOptionsAttr {
    pub(super) fn new(options: Conv2dOptions) -> Self {
        Self {
            bytes: BytesAttr::new(
                [
                    options.strides[0],
                    options.strides[1],
                    options.padding[0][0],
                    options.padding[0][1],
                    options.padding[1][0],
                    options.padding[1][1],
                    options.dilation[0],
                    options.dilation[1],
                    options.groups,
                ]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
            ),
        }
    }

    pub(super) fn options(&self) -> Conv2dOptions {
        let (values, remainder) = self.bytes.as_ref().as_chunks::<8>();
        assert!(
            remainder.is_empty() && values.len() == 9,
            "malformed conv2d options"
        );
        let values = values
            .iter()
            .map(|&value| i64::from_le_bytes(value))
            .collect::<Vec<_>>();
        Conv2dOptions {
            strides: [values[0], values[1]],
            padding: [[values[2], values[3]], [values[4], values[5]]],
            dilation: [values[6], values[7]],
            groups: values[8],
        }
    }
}

impl ConvTranspose2dOptionsAttr {
    pub(super) fn new(options: ConvTranspose2dOptions) -> Self {
        let values = [
            options.strides[0],
            options.strides[1],
            options.padding[0][0],
            options.padding[0][1],
            options.padding[1][0],
            options.padding[1][1],
            options.dilation[0],
            options.dilation[1],
            options.output_padding[0],
            options.output_padding[1],
        ];
        Self {
            bytes: BytesAttr::new(values.into_iter().flat_map(i64::to_le_bytes).collect()),
        }
    }

    pub(super) fn options(&self) -> ConvTranspose2dOptions {
        let (values, remainder) = self.bytes.as_ref().as_chunks::<8>();
        assert!(
            remainder.is_empty() && values.len() == 10,
            "malformed conv_transpose2d options"
        );
        let value = |index| i64::from_le_bytes(values[index]);
        ConvTranspose2dOptions {
            strides: [value(0), value(1)],
            padding: [[value(2), value(3)], [value(4), value(5)]],
            dilation: [value(6), value(7)],
            output_padding: [value(8), value(9)],
        }
    }
}

impl Pool2dOptionsAttr {
    pub(super) fn new(options: Pool2dOptions) -> Self {
        Self {
            bytes: BytesAttr::new(
                [
                    options.window[0],
                    options.window[1],
                    options.strides[0],
                    options.strides[1],
                    options.padding[0][0],
                    options.padding[0][1],
                    options.padding[1][0],
                    options.padding[1][1],
                ]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
            ),
        }
    }

    pub(super) fn options(&self) -> Pool2dOptions {
        let (values, remainder) = self.bytes.as_ref().as_chunks::<8>();
        assert!(
            remainder.is_empty() && values.len() == 8,
            "malformed pool2d options"
        );
        let values = values
            .iter()
            .copied()
            .map(i64::from_le_bytes)
            .collect::<Vec<_>>();
        Pool2dOptions {
            window: [values[0], values[1]],
            strides: [values[2], values[3]],
            padding: [[values[4], values[5]], [values[6], values[7]]],
        }
    }
}
