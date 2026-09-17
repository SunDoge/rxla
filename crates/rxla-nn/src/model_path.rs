use crate::{Result, validate_name};
use std::fmt;

/// A pure coordinate in a model's parameter/effect tree.
///
/// Paths do not retain a tracing context. They can therefore also be used by
/// schemas, parameter selections, weight stores, and checkpoint mappings.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelPath {
    segments: Vec<String>,
}

impl ModelPath {
    pub fn root() -> Self {
        Self::default()
    }

    pub fn is_root(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn segments(&self) -> impl ExactSizeIterator<Item = &str> {
        self.segments.iter().map(String::as_str)
    }

    pub fn at<P: ModelPathSpec>(&self, path: P) -> Result<Self> {
        let mut child = self.clone();
        path.append_to(&mut child)?;
        Ok(child)
    }

    pub(crate) fn parameter(&self, name: &str) -> Result<String> {
        validate_name(name)?;
        if self.is_root() {
            Ok(name.to_owned())
        } else {
            Ok(format!("{self}.{name}"))
        }
    }

    fn push_name(&mut self, name: String) -> Result<()> {
        validate_name(&name)?;
        self.segments.push(name);
        Ok(())
    }

    fn push_index(&mut self, index: usize) {
        self.segments.push(index.to_string());
    }
}

impl fmt::Display for ModelPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, segment) in self.segments.iter().enumerate() {
            if index != 0 {
                formatter.write_str(".")?;
            }
            formatter.write_str(segment)?;
        }
        Ok(())
    }
}

/// A path fragment accepted by [`crate::Cx::at`] and [`ModelPath::at`].
///
/// Tuple implementations allow names and indices to be mixed without
/// allocating temporary strings: `cx.at(("blocks", index, "conv"))`.
pub trait ModelPathSpec {
    fn append_to(self, path: &mut ModelPath) -> Result<()>;
}

impl ModelPathSpec for &str {
    fn append_to(self, path: &mut ModelPath) -> Result<()> {
        path.push_name(self.to_owned())
    }
}

impl ModelPathSpec for String {
    fn append_to(self, path: &mut ModelPath) -> Result<()> {
        path.push_name(self)
    }
}

impl ModelPathSpec for usize {
    fn append_to(self, path: &mut ModelPath) -> Result<()> {
        path.push_index(self);
        Ok(())
    }
}

impl ModelPathSpec for ModelPath {
    fn append_to(self, path: &mut ModelPath) -> Result<()> {
        path.segments.extend(self.segments);
        Ok(())
    }
}

impl ModelPathSpec for &ModelPath {
    fn append_to(self, path: &mut ModelPath) -> Result<()> {
        path.segments.extend(self.segments.iter().cloned());
        Ok(())
    }
}

macro_rules! impl_tuple_path {
    ($(($($part:ident),+)),+ $(,)?) => {
        $(
            impl<$($part: ModelPathSpec),+> ModelPathSpec for ($($part,)+) {
                #[allow(non_snake_case)]
                fn append_to(self, path: &mut ModelPath) -> Result<()> {
                    let ($($part,)+) = self;
                    $($part.append_to(path)?;)+
                    Ok(())
                }
            }
        )+
    };
}

impl_tuple_path!(
    (A, B),
    (A, B, C),
    (A, B, C, D),
    (A, B, C, D, E),
    (A, B, C, D, E, F),
    (A, B, C, D, E, F, G),
    (A, B, C, D, E, F, G, H),
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuples_mix_names_and_indices() {
        let path = ModelPath::root()
            .at(("encoder", "blocks", 17usize, "conv"))
            .unwrap();

        assert_eq!(path.to_string(), "encoder.blocks.17.conv");
        assert_eq!(
            path.segments().collect::<Vec<_>>(),
            ["encoder", "blocks", "17", "conv"]
        );
    }

    #[test]
    fn invalid_names_fail_at_the_path_boundary() {
        assert!(ModelPath::root().at(("encoder", "bad.name")).is_err());
    }
}
