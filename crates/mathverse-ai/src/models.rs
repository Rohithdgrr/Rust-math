//! Sequential model container: chains layers + activations for forward pass.

use crate::tensor::Tensor;
use crate::layers::{Linear, LayerNorm, Dropout};
use mathverse_core::error::MathResult;

/// Activation type enum.
#[derive(Clone, Copy)]
pub enum Activation {
    /// Rectified Linear Unit.
    ReLU,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Gaussian Error Linear Unit (tanh approximation).
    GELU,
    /// Swish / SiLU.
    Swish,
    /// Softmax along a given `axis`.
    Softmax {
        /// Axis along which softmax is applied.
        axis: usize,
    },
    /// No activation (identity).
    None,
}

/// Layer enum for sequential models.
pub enum Layer {
    /// Fully connected linear layer.
    Linear(Linear),
    /// Layer normalization.
    LayerNorm(LayerNorm),
    /// Dropout layer.
    Dropout(Dropout),
    /// Activation function.
    Activation(Activation),
}

/// Sequential model: chains layers in order.
pub struct Sequential {
    /// Layers executed in order.
    pub layers: Vec<Layer>,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    /// Create an empty sequential model.
    pub fn new() -> Self { Self { layers: Vec::new() } }

    /// Append a [`Linear`] layer.
    pub fn add_linear(mut self, in_f: usize, out_f: usize) -> Self {
        self.layers.push(Layer::Linear(Linear::new(in_f, out_f)));
        self
    }

    /// Append a [`LayerNorm`] layer.
    pub fn add_layer_norm(mut self, normalized_shape: usize) -> Self {
        self.layers.push(Layer::LayerNorm(LayerNorm::new(normalized_shape, 1e-5)));
        self
    }

    /// Append a [`Dropout`] layer.
    pub fn add_dropout(mut self, p: f64) -> Self {
        self.layers.push(Layer::Dropout(Dropout::new(p)));
        self
    }

    /// Append an activation.
    pub fn add_activation(mut self, act: Activation) -> Self {
        self.layers.push(Layer::Activation(act));
        self
    }

    /// Forward pass through all layers.
    pub fn forward(&mut self, x: &Tensor, training: bool) -> MathResult<Tensor> {
        let mut out = x.clone();
        for layer in &mut self.layers {
            match layer {
                Layer::Linear(l) => out = l.forward(&out)?,
                Layer::LayerNorm(l) => out = l.forward(&out)?,
                Layer::Dropout(d) => out = d.forward(&out, training),
                Layer::Activation(act) => {
                    out = match act {
                        Activation::ReLU => crate::activations::relu(&out),
                        Activation::Sigmoid => crate::activations::sigmoid(&out),
                        Activation::Tanh => crate::activations::tanh(&out),
                        Activation::GELU => crate::activations::gelu(&out),
                        Activation::Swish => crate::activations::swish(&out),
                        Activation::Softmax { axis } => crate::activations::softmax(&out, *axis)?,
                        Activation::None => out,
                    };
                }
            }
        }
        Ok(out)
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| match l {
            Layer::Linear(l) => l.num_params(),
            Layer::LayerNorm(l) => l.normalized_shape * 2,
            _ => 0,
        }).sum()
    }
}

/// Multi-layer perceptron (MLP) for classification.
pub struct MLP {
    /// Hidden linear layers.
    pub hidden_layers: Vec<Linear>,
    /// Output linear layer.
    pub output_layer: Linear,
    /// Activation used between hidden layers.
    pub activation: Activation,
}

impl MLP {
    /// Create a new MLP.
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Self {
        let mut hidden_layers = Vec::new();
        let mut prev = input_size;
        for &h in hidden_sizes {
            hidden_layers.push(Linear::new(prev, h));
            prev = h;
        }
        Self { hidden_layers, output_layer: Linear::new(prev, output_size), activation: Activation::ReLU }
    }

    /// Forward pass through all layers.
    pub fn forward(&self, x: &Tensor) -> MathResult<Tensor> {
        let mut out = x.clone();
        for layer in &self.hidden_layers {
            out = layer.forward(&out)?;
            out = match self.activation {
                Activation::ReLU => crate::activations::relu(&out),
                Activation::Sigmoid => crate::activations::sigmoid(&out),
                Activation::Tanh => crate::activations::tanh(&out),
                _ => out,
            };
        }
        self.output_layer.forward(&out)
    }
}

/// Transformer encoder block.
pub struct TransformerBlock {
    /// Query projection weight.
    pub attn_wq: Tensor,
    /// Key projection weight.
    pub attn_wk: Tensor,
    /// Value projection weight.
    pub attn_wv: Tensor,
    /// Output projection weight.
    pub attn_wo: Tensor,
    /// First feed-forward layer.
    pub ff_w1: Linear,
    /// Second feed-forward layer.
    pub ff_w2: Linear,
    /// First layer norm (pre-attention).
    pub ln1: LayerNorm,
    /// Second layer norm (pre-FFN).
    pub ln2: LayerNorm,
    /// Model width.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

impl TransformerBlock {
    /// Create a transformer block with random-initialized parameters.
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        assert!(d_model % num_heads == 0, "TransformerBlock: d_model ({d_model}) must be divisible by num_heads ({num_heads})");
        let _d_k = d_model / num_heads;
        Self {
            attn_wq: Tensor::randn(&[d_model, d_model]).mul_scalar((2.0 / d_model as f64).sqrt()),
            attn_wk: Tensor::randn(&[d_model, d_model]).mul_scalar((2.0 / d_model as f64).sqrt()),
            attn_wv: Tensor::randn(&[d_model, d_model]).mul_scalar((2.0 / d_model as f64).sqrt()),
            attn_wo: Tensor::randn(&[d_model, d_model]).mul_scalar((2.0 / d_model as f64).sqrt()),
            ff_w1: Linear::new(d_model, d_ff),
            ff_w2: Linear::new(d_ff, d_model),
            ln1: LayerNorm::new(d_model, 1e-5),
            ln2: LayerNorm::new(d_model, 1e-5),
            d_model,
            num_heads,
        }
    }

    /// Forward pass with optional attention mask.
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> MathResult<Tensor> {
        // Self-attention with residual
        let normed = self.ln1.forward(x)?;
        let attn_out = crate::attention::multi_head_attention(
            &normed, &self.attn_wq, &self.attn_wk, &self.attn_wv, &self.attn_wo,
            self.num_heads, mask,
        )?;
        let h = x.add(&attn_out)?;

        // Feed-forward with residual
        let normed2 = self.ln2.forward(&h)?;
        let ff_out = self.ff_w1.forward(&normed2)?;
        let ff_out = crate::activations::gelu(&ff_out);
        let ff_out = self.ff_w2.forward(&ff_out)?;
        h.add(&ff_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_test() {
        let mut model = Sequential::new()
            .add_linear(4, 8)
            .add_activation(Activation::ReLU)
            .add_linear(8, 2);
        let x = Tensor::zeros(&[3, 4]);
        let out = model.forward(&x, false).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(model.num_params(), 4 * 8 + 8 + 8 * 2 + 2);
    }

    #[test]
    fn mlp_test() {
        let mlp = MLP::new(10, &[32, 16], 3);
        let x = Tensor::zeros(&[5, 10]);
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.shape, vec![5, 3]);
    }

    #[test]
    fn transformer_block_test() {
        let block = TransformerBlock::new(16, 4, 32);
        let x = Tensor::randn(&[2, 8, 16]);
        let out = block.forward(&x, None).unwrap();
        assert_eq!(out.shape, vec![2, 8, 16]);
    }
}

