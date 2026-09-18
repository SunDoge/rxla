use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// An einops-style static layout transformation.
///
/// Parsing and validation happen while tracing. Applying it emits only reshape
/// and transpose operations, so the compiled program has no parser overhead.
#[derive(Clone, Debug)]
pub struct Rearrange<'a> {
    pattern: &'a str,
    axes: BTreeMap<&'a str, i64>,
}

impl<'a> Rearrange<'a> {
    pub fn new(pattern: &'a str) -> Self {
        Self {
            pattern,
            axes: BTreeMap::new(),
        }
    }

    /// Supply an axis size that cannot be inferred independently.
    #[must_use]
    pub fn axis(mut self, name: &'a str, size: i64) -> Self {
        self.axes.insert(name, size);
        self
    }

    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        let (source, destination) = self
            .pattern
            .split_once("->")
            .ok_or_else(|| err("rearrange pattern requires `->`"))?;
        let source = parse_side(source)?;
        let destination = parse_side(destination)?;
        if source.len() != input.shape().len() {
            return Err(err("rearrange source rank does not match tensor rank"));
        }

        let mut sizes = self
            .axes
            .iter()
            .map(|(&name, &size)| (name.to_owned(), size))
            .collect::<BTreeMap<_, _>>();
        if sizes.values().any(|&size| size <= 0) {
            return Err(err("rearrange axis sizes must be positive"));
        }
        for (group, &dimension) in source.iter().zip(input.shape()) {
            bind_group(group, dimension, &mut sizes)?;
        }

        let source_axes = named_axes(&source)?;
        let destination_axes = named_axes(&destination)?;
        if source_axes.len() != destination_axes.len()
            || source_axes.iter().collect::<BTreeSet<_>>()
                != destination_axes.iter().collect::<BTreeSet<_>>()
        {
            return Err(err("rearrange named axes must occur once on each side"));
        }
        let atomic_shape = source_axes
            .iter()
            .map(|axis| {
                sizes
                    .get(axis)
                    .copied()
                    .ok_or_else(|| err("unresolved axis"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut output = input.reshape(atomic_shape)?;
        let permutation = destination_axes
            .iter()
            .map(|axis| {
                source_axes
                    .iter()
                    .position(|candidate| candidate == axis)
                    .ok_or_else(|| err("unknown destination axis"))
            })
            .collect::<Result<Vec<_>>>()?;
        if permutation
            .iter()
            .enumerate()
            .any(|(axis, &source)| axis != source)
        {
            output = output.transpose(permutation)?;
        }
        let destination_shape = destination
            .iter()
            .map(|group| {
                group.iter().try_fold(1_i64, |size, axis| {
                    let factor = match axis {
                        Axis::Unit => 1,
                        Axis::Named(name) => *sizes
                            .get(name)
                            .ok_or_else(|| err("unresolved destination axis"))?,
                    };
                    size.checked_mul(factor)
                        .ok_or_else(|| err("rearrange shape overflow"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        output.reshape(destination_shape)
    }
}

#[derive(Clone, Debug)]
enum Axis {
    Named(String),
    Unit,
}

fn parse_side(value: &str) -> Result<Vec<Vec<Axis>>> {
    let normalized = value.replace('(', " ( ").replace(')', " ) ");
    let mut result = Vec::new();
    let mut group = None;
    for token in normalized.split_whitespace() {
        match token {
            "(" if group.is_none() => group = Some(Vec::new()),
            "(" => return Err(err("rearrange groups cannot be nested")),
            ")" => {
                let axes = group
                    .take()
                    .ok_or_else(|| err("unmatched rearrange group"))?;
                if axes.is_empty() {
                    return Err(err("rearrange groups cannot be empty"));
                }
                result.push(axes);
            }
            _ => {
                let axis = if token == "1" {
                    Axis::Unit
                } else if token
                    .chars()
                    .next()
                    .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
                    && token
                        .chars()
                        .all(|character| character.is_ascii_alphanumeric() || character == '_')
                {
                    Axis::Named(token.to_owned())
                } else {
                    return Err(err("invalid rearrange axis name"));
                };
                if let Some(axes) = &mut group {
                    axes.push(axis);
                } else {
                    result.push(vec![axis]);
                }
            }
        }
    }
    if group.is_some() || result.is_empty() {
        return Err(err("unclosed or empty rearrange pattern"));
    }
    Ok(result)
}

fn bind_group(group: &[Axis], dimension: i64, sizes: &mut BTreeMap<String, i64>) -> Result<()> {
    let mut product = 1_i64;
    let mut unknown = None;
    for axis in group {
        match axis {
            Axis::Unit => {}
            Axis::Named(name) => match sizes.get(name) {
                Some(&size) => {
                    product = product
                        .checked_mul(size)
                        .ok_or_else(|| err("rearrange shape overflow"))?;
                }
                None if unknown.is_none() => unknown = Some(name.clone()),
                None => return Err(err("only one unknown is allowed per source group")),
            },
        }
    }
    if let Some(name) = unknown {
        if dimension <= 0 || dimension % product != 0 {
            return Err(err("rearrange cannot infer an integral axis size"));
        }
        sizes.insert(name, dimension / product);
    } else if product != dimension {
        return Err(err("rearrange source group has the wrong size"));
    }
    Ok(())
}

fn named_axes(groups: &[Vec<Axis>]) -> Result<Vec<String>> {
    let axes = groups
        .iter()
        .flatten()
        .filter_map(|axis| match axis {
            Axis::Named(name) => Some(name.clone()),
            Axis::Unit => None,
        })
        .collect::<Vec<_>>();
    if axes.iter().collect::<BTreeSet<_>>().len() != axes.len() {
        return Err(err("rearrange axes cannot be repeated"));
    }
    Ok(axes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_merged_axes_and_reorders() {
        let graph = Graph::default();
        let input = graph.input(&[6, 4, 5, 1]).unwrap();
        let output = Rearrange::new("(n c) h w 1 -> n h w c")
            .axis("n", 2)
            .apply(&input)
            .unwrap();
        assert_eq!(output.shape(), [2, 4, 5, 3]);
        let stablehlo = graph.stablehlo(&output).unwrap();
        assert!(stablehlo.contains("stablehlo.reshape"));
        assert!(stablehlo.contains("stablehlo.transpose"));
    }

    #[test]
    fn rejects_ambiguous_and_mismatched_patterns() {
        let graph = Graph::default();
        let input = graph.input(&[6, 4]).unwrap();
        assert!(Rearrange::new("(n c) h -> n h c").apply(&input).is_err());
        assert!(
            Rearrange::new("(n c) h -> n h c")
                .axis("n", 4)
                .apply(&input)
                .is_err()
        );
    }
}
