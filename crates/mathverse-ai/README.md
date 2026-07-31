# mathverse-ai

AI/ML mathematical primitives: tensors, activations, losses, optimizers, and attention mechanisms — all in pure Rust with zero external dependencies.

## Features

- N-dimensional tensor with broadcasting, reshaping, and element-wise math
- Activation functions: ReLU, GELU, Swish, Mish, Softmax, Sigmoid, Tanh, and more
- Loss functions: MSE, MAE, Huber, Cross-Entropy, Binary Cross-Entropy, KL Divergence, Hinge
- Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC AUC, R², Explained Variance
- Stateful optimizers: SGD (with momentum), Adam, AdamW
- Learning rate schedulers: Constant, Step Decay, Cosine Annealing, Linear Warmup
- Attention math: QKV projection, scaled dot-product attention, multi-head attention
- Positional encodings: sinusoidal (additive), rotary (RoPE)
- Layer/Batch/RMS normalization

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `tensor` | N-D tensor with row-major layout, broadcasting, math ops | `Tensor::zeros`, `matmul`, `add`, `layer_norm`, `broadcast_to` |
| `activations` | Element-wise activation functions and derivatives | `relu`, `gelu`, `sigmoid`, `softmax`, `swish`, `mish` |
| `losses` | Regression and classification loss functions | `mse`, `cross_entropy`, `binary_cross_entropy_with_logits`, `huber` |
| `metrics` | Model evaluation metrics | `accuracy`, `f1`, `confusion_matrix`, `roc_auc`, `r_squared` |
| `optimizers` | Stateful gradient-based optimizers | `Sgd`, `Adam`, `AdamW`, `LrScheduler` |
| `attention` | Transformer attention math | `scaled_dot_product_attention`, `multi_head_attention`, `apply_rope` |

## Installation

```bash
cargo add mathverse-ai
```

Or add to `Cargo.toml`:

```toml
[dependencies]
mathverse-ai = { path = "../mathverse-ai" }
```

## Quick Start

```rust
use mathverse_ai::{Tensor, sigmoid, softmax, mse, Adam};

fn main() {
    // Create tensors
    let logits = Tensor::new(&[1, 3], &[2.0, 1.0, 0.5]).unwrap();
    let probs = softmax(&logits, 1).unwrap();
    println!("Softmax: {:?}", probs.data);
    // Softmax: [0.5906, 0.2447, 0.1647]

    // MSE loss
    let pred = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
    let target = Tensor::new(&[3], &[1.0, 2.0, 5.0]).unwrap();
    let loss = mse(&pred, &target).unwrap();
    println!("MSE: {:.4}", loss);
    // MSE: 1.3333

    // Adam optimizer minimizing f(x) = x²
    let mut opt = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0);
    let mut x = [5.0];
    for _ in 0..100 {
        let g = [2.0 * x[0]];
        opt.step(&mut x, &g);
    }
    println!("x after 100 steps: {:.6}", x[0]);
    // x after 100 steps: 0.000001
}
```

## Module Documentation

### Tensor

The core N-dimensional array type. Supports broadcasting (NumPy-style trailing dimensions), element-wise arithmetic, matrix multiply, reductions, normalization, and shape manipulation.

```
Memory Layout (row-major):
┌──────────────────────────────────────┐
│  Shape: [2, 3, 4]                   │
│  Data:  [a₀₀₀ a₀₀₁ a₀₀₂ a₀₀₃       │
│          a₀₁₀ a₀₁₁ a₀₁₂ a₀₁₃       │
│          a₀₂₀ a₀₂₁ a₀₂₂ a₀₂₃       │
│          a₁₀₀ a₁₀₁ a₁₀₂ a₁₀₃       │
│          a₁₁₀ a₁₁₁ a₁₁₂ a₁₁₃       │
│          a₁₂₀ a₁₂₁ a₁₂₂ a₁₂₃]      │
│  Numel: 24                           │
└──────────────────────────────────────┘
```

```rust
use mathverse_ai::Tensor;

// Constructors
let t = Tensor::zeros(&[2, 3]);
let t = Tensor::arange(0.0, 6.0, 1.0).reshape(&[2, 3]).unwrap();

// Broadcasting
let a = Tensor::new(&[2, 1], &[1.0, 2.0]).unwrap();
let b = Tensor::new(&[1, 3], &[10.0, 20.0, 30.0]).unwrap();
let c = a.add(&b).unwrap(); // shape [2, 3]
// c = [[11, 21, 31],
//      [12, 22, 32]]

// Matrix multiply
let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
let b = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
let c = a.matmul(&b).unwrap(); // shape [2, 2]
// c = [[58, 64],
//      [139, 154]]

// Layer normalization
let t = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
let norm = t.layer_norm(1e-5); // each row has mean ≈ 0
```

**Real-world use**: Data preprocessing, neural network forward passes, batch operations.

### Activations

All activation functions operate element-wise on tensors. Derivative functions (suffixed `_grad`) are provided for backpropagation.

```
Activation Comparison:
         ReLU          GELU           Swish          Sigmoid
    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
    │       /│    │      / │    │      / │    │        │
    │      / │    │     /  │    │    _/  │    │───────/│
────┼──────/─┼────┼────/───┼────┼──/─────┼────┼──────/─┼───
    │     /  │    │   /    │    │ /      │    │     /  │
    │    /   │    │  /     │    │/       │    │    /   │
    └────────┘    └────────┘    └────────┘    └────────┘
    f(x)=max(0,x) f(x)=0.5x(1+tanh) f(x)=x·σ(x) f(x)=1/(1+e⁻ˣ)
```

```rust
use mathverse_ai::{Tensor, relu, gelu, sigmoid, softmax, swish};

let t = Tensor::new(&[5], &[-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();

let r = relu(&t);
// [0.0, 0.0, 0.0, 1.0, 2.0]

let s = sigmoid(&t);
// [0.1192, 0.2689, 0.5, 0.7311, 0.8808]

// Softmax (probabilities sum to 1)
let logits = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
let probs = softmax(&logits, 0).unwrap();
// [0.0900, 0.2447, 0.6652]

let g = gelu(&t);
// GELU is smoother than ReLU, used in GPT/BERT
```

**Real-world use**: Hidden layer activations in neural networks. GELU in transformers, ReLU in CNNs, Sigmoid for binary output.

### Loss Functions

Compute scalar losses and their gradients for training neural networks.

```rust
use mathverse_ai::{Tensor, mse, cross_entropy, binary_cross_entropy_with_logits};

// Regression: MSE
let pred = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
let target = Tensor::new(&[3], &[1.0, 2.0, 5.0]).unwrap();
println!("MAE: {:.4}", mse(&pred, &target).unwrap());
// MAE: 1.3333

// Classification: Cross-Entropy
let logits = Tensor::new(&[2, 3], &[
    10.0, 1.0, 1.0,  // confident class 0
    1.0, 10.0, 1.0,  // confident class 1
]).unwrap();
let targets = Tensor::new(&[2], &[0.0, 1.0]).unwrap();
println!("CE: {:.4}", cross_entropy(&logits, &targets).unwrap());
// CE: 0.0001

// Binary: BCE with logits (numerically stable)
let logits = Tensor::new(&[3], &[10.0, -10.0, 0.0]).unwrap();
let target = Tensor::new(&[3], &[1.0, 0.0, 0.5]).unwrap();
println!("BCE: {:.4}", binary_cross_entropy_with_logits(&logits, &target).unwrap());
// BCE: 0.0023
```

**Formulas**:
- MSE: `L = mean((pred - target)²)`
- Cross-Entropy: `L = -mean(log_softmax(logits)[target])`
- Huber: `L = 0.5(p-t)² if |p-t|≤δ, else δ(|p-t| - 0.5δ)`

### Metrics

```rust
use mathverse_ai::{Tensor, accuracy, f1, confusion_matrix};

let pred = Tensor::new(&[5], &[0.0, 1.0, 2.0, 1.0, 0.0]).unwrap();
let target = Tensor::new(&[5], &[0.0, 2.0, 2.0, 1.0, 1.0]).unwrap();

println!("Accuracy: {:.2}", accuracy(&pred, &target).unwrap());
// Accuracy: 0.60

let f1_scores = f1(&pred, &target, 3).unwrap();
// Per-class F1 scores

let cm = confusion_matrix(&pred, &target, 3).unwrap();
// 3×3 confusion matrix
```

### Optimizers

Stateful optimizers that maintain internal momentum/adaptation state.

```
Optimizer Convergence (minimizing f(x) = x²):

  x
  10│*
    │  *
    │    *
    │      *  Adam (fast)
    │        ·····
  5 │  ·  SGD
    │   ·  (slower)
    │    ········
    │          ·············
  0 └───────────────────────
    0     50    100   150  steps
```

```rust
use mathverse_ai::{Adam, AdamW, Sgd, LrScheduler, Schedule};

// Adam optimizer
let mut opt = Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0);
let mut params = vec![5.0, -3.0, 2.0];
for _ in 0..1000 {
    let grads: Vec<f64> = params.iter().map(|p| 2.0 * p).collect();
    opt.step(&mut params, &grads);
}
// params ≈ [0, 0, 0]

// Cosine annealing LR schedule
let mut scheduler = LrScheduler::new(
    Schedule::CosineAnnealing { t_max: 100, eta_min: 0.0001 }
);
for _ in 0..100 {
    let lr = scheduler.get_lr();
    scheduler.step();
}
// LR decays from 1.0 → 0.0001 following cosine curve
```

### Attention

Transformer attention mechanisms: QKV projection, scaled dot-product attention, multi-head attention, and positional encodings.

```
Multi-Head Attention:
                    ┌─────┐
    X ──────────────┤Wq,Wk│──── Q,K,V
    [batch,seq,d]   │Wv   │
                    └──┬──┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  Head 1  │ │  Head 2  │ │  Head h  │
    │ Attn(Q₁, │ │ Attn(Q₂, │ │ Attn(Qₕ, │
    │  K₁,V₁) │ │  K₂,V₂) │ │  Kₕ,Vₕ) │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └────────┬───┘────────────┘
                  │ Concat
              ┌───▼───┐
              │  Wo   │
              └───┬───┘
                  ▼
              Output
```

```rust
use mathverse_ai::{Tensor, multi_head_attention, sinusoidal_encoding, apply_rope};

// Scaled dot-product attention
let q = Tensor::new(&[1, 2, 4], &[1.0; 8]).unwrap();
let k = Tensor::new(&[1, 2, 4], &[1.0; 8]).unwrap();
let v = Tensor::new(&[1, 2, 4], &[1.0; 8]).unwrap();
let (out, weights) = mathverse_ai::scaled_dot_product_attention(&q, &k, &v, None, None).unwrap();
// out: [1, 2, 4], weights: [1, 2, 2] (softmax attention map)

// Sinusoidal positional encoding
let enc = sinusoidal_encoding(10, 64, 10000);
// enc shape: [1, 10, 64]

// RoPE (Rotary Position Embeddings)
let x = Tensor::randn(&[1, 8, 64]);
let y = apply_rope(&x, 8, 10000.0);
// Norm-preserving rotation by position
```

**Formulas**:
- Attention: `Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V`
- RoPE: `θ_i = pos / base^(2i/d)`, apply 2D rotation to each pair

## Future Scope

- [ ] Autograd / automatic differentiation
- [ ] GPU acceleration via `wgpu` compute shaders
- [ ] Convolution operations (Conv1d, Conv2d)
- [ ] Dropout, LayerNorm with learnable parameters
- [ ] KV-cache for autoregressive inference
- [ ] Flash attention for memory-efficient attention
- [ ] Mixed precision (f16/bf16) support

## License

MIT OR Apache-2.0
