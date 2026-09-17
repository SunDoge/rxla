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

    pub fn at<S: ModelPathSegment>(&self, segment: S) -> Result<Self> {
        let mut child = self.clone();
        child.push(segment)?;
        Ok(child)
    }

    pub fn push<S: ModelPathSegment>(&mut self, segment: S) -> Result<&mut Self> {
        segment.push_to(self)?;
        Ok(self)
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

/// One name or numeric index accepted by [`crate::Cx::at`] and
/// [`ModelPath::push`].
pub trait ModelPathSegment {
    fn push_to(self, path: &mut ModelPath) -> Result<()>;
}

impl ModelPathSegment for &str {
    fn push_to(self, path: &mut ModelPath) -> Result<()> {
        path.push_name(self.to_owned())
    }
}

impl ModelPathSegment for String {
    fn push_to(self, path: &mut ModelPath) -> Result<()> {
        path.push_name(self)
    }
}

impl ModelPathSegment for usize {
    fn push_to(self, path: &mut ModelPath) -> Result<()> {
        path.push_index(self);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pushes_mix_names_and_indices() {
        let mut path = ModelPath::root();
        path.push("encoder")
            .unwrap()
            .push("blocks")
            .unwrap()
            .push(17usize)
            .unwrap()
            .push("conv")
            .unwrap();

        assert_eq!(path.to_string(), "encoder.blocks.17.conv");
        assert_eq!(
            path.segments().collect::<Vec<_>>(),
            ["encoder", "blocks", "17", "conv"]
        );
    }

    #[test]
    fn invalid_names_fail_at_the_path_boundary() {
        assert!(ModelPath::root().at("bad.name").is_err());
    }
}
