use super::*;

/// Pairing of last-axis channels for rotary position embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotaryLayout {
    /// Pair channel i with i + width/2 (Llama-style).
    SplitHalf,
    /// Pair channels 2*i and 2*i+1.
    Interleaved,
}
impl Tensor {
    /// Apply RoPE from graph-computed angles in radians, computing cosine/sine
    /// on the backend. Angles use the same half-width broadcasting convention as
    /// `rotary_embedding`. Gradients can flow into angles, positions and frequency
    /// tensors that construct them; no host table or implicit detach is used.
    /// Frequency schedule, position layout and scaling remain caller policy.
    pub fn rotary_embedding_angles(&self, angles: &Tensor, layout: RotaryLayout) -> Result<Tensor> {
        self.rotary_embedding(&angles.cos()?, &angles.sin()?, layout)
    }

    /// Apply `(a*cos-b*sin, b*cos+a*sin)` to last-axis channel pairs.
    /// Width must be positive/even. Cos/sin must each broadcast to the input
    /// shape with its final dimension halved (one angle per pair, not duplicated
    /// full-width tables). Both tensors must belong to the input graph.
    ///
    /// Frequency generation, position lookup, scaling and causal policy belong
    /// to the caller. Cos/sin can be runtime inputs. This rotates the full final
    /// axis; partial rotation can be composed with narrow/concatenate explicitly.
    pub fn rotary_embedding(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        layout: RotaryLayout,
    ) -> Result<Tensor> {
        let Some(&width) = self.shape().last() else {
            return Err(err("RoPE requires rank >= 1"));
        };
        if width <= 0 || width % 2 != 0 {
            return Err(err("RoPE requires positive even last-axis width"));
        }
        for angle in [cos, sin] {
            if !Arc::ptr_eq(&self.graph().0, &angle.graph().0) {
                return Err(err("cross-graph RoPE angle tensor"));
            }
        }
        let axis = self.shape().len() - 1;
        let mut shape = self.shape().to_vec();
        shape[axis] = width / 2;
        let cos = cos.broadcast_to(&shape)?;
        let sin = sin.broadcast_to(&shape)?;
        let (a, b) = match layout {
            RotaryLayout::SplitHalf => (
                self.narrow(axis, 0, width / 2)?,
                self.narrow(axis, width / 2, width / 2)?,
            ),
            RotaryLayout::Interleaved => {
                let mut paired = shape;
                paired.push(2);
                let paired = self.reshape(&paired)?;
                (
                    paired.narrow(axis + 1, 0, 1)?.squeeze(axis + 1)?,
                    paired.narrow(axis + 1, 1, 1)?.squeeze(axis + 1)?,
                )
            }
        };
        let first = a.mul(&cos)?.sub(&b.mul(&sin)?)?;
        let second = b.mul(&cos)?.add(&a.mul(&sin)?)?;
        match layout {
            RotaryLayout::SplitHalf => Tensor::concatenate(&[first, second], axis),
            RotaryLayout::Interleaved => {
                Tensor::stack(&[first, second], axis + 1)?.reshape(self.shape())
            }
        }
    }
}
