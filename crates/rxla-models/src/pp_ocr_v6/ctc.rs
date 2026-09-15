use rayon::prelude::*;

use super::{Result, error::InvalidCtcShapeSnafu};

/// One greedy CTC result. Token zero is conventionally the blank token.
#[derive(Debug, PartialEq)]
pub struct DecodedSequence {
    pub tokens: Vec<usize>,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CtcDecoder {
    blank: usize,
}

impl CtcDecoder {
    pub fn new(blank: usize) -> Self {
        Self { blank }
    }

    /// Decode contiguous `[batch, time, classes]` probabilities in one pass.
    /// The returned confidence is the mean probability of emitted tokens.
    pub fn decode(&self, probabilities: &[f32], shape: [usize; 3]) -> Result<Vec<DecodedSequence>> {
        let [batch, time, classes] = shape;
        snafu::ensure!(
            classes > self.blank
                && batch
                    .checked_mul(time)
                    .and_then(|size| size.checked_mul(classes))
                    == Some(probabilities.len()),
            InvalidCtcShapeSnafu
        );
        Ok(probabilities
            .par_chunks_exact(time * classes)
            .map(|sequence| {
                let mut previous = self.blank;
                let mut tokens = Vec::with_capacity(time);
                let mut confidence = 0.0;
                for step in sequence.chunks_exact(classes) {
                    let (token, &score) = step
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap();
                    if token != self.blank && token != previous {
                        tokens.push(token);
                        confidence += score;
                    }
                    previous = token;
                }
                let confidence = if tokens.is_empty() {
                    0.0
                } else {
                    confidence / tokens.len() as f32
                };
                DecodedSequence { tokens, confidence }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_repeats_separated_by_blanks() {
        let classes = 3;
        let ids = [1, 1, 0, 1, 2];
        let mut probabilities = vec![0.0; ids.len() * classes];
        for (step, id) in ids.into_iter().enumerate() {
            probabilities[step * classes + id] = 0.75;
        }
        assert_eq!(
            CtcDecoder::default()
                .decode(&probabilities, [1, 5, classes])
                .unwrap(),
            vec![DecodedSequence {
                tokens: vec![1, 1, 2],
                confidence: 0.75
            }]
        );
    }
}
