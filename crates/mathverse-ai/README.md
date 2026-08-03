# MathVerse AI

[![Crates.io](https://img.shields.io/crates/v/mathverse-ai.svg)](https://crates.io/crates/mathverse-ai)
[![docs.rs](https://docs.rs/mathverse-ai/badge.svg)](https://docs.rs/mathverse-ai)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

AI/ML mathematical primitives: tensors, activations, losses, optimizers, and attention mechanisms — all in pure Rust with zero external dependencies.

---

## Features

- **N-dimensional tensor** — Broadcasting, reshaping, element-wise math, matrix multiply
- **Activation functions** — ReLU, GELU, Swish, Mish, Softmax, Sigmoid, Tanh, and more
- **Loss functions** — MSE, MAE, Huber, Cross-Entropy, Binary Cross-Entropy, KL Divergence, Hinge
- **Evaluation metrics** — Accuracy, Precision, Recall, F1, Confusion Matrix, ROC AUC, R²
- **Stateful optimizers** — SGD (with momentum), Adam, AdamW
- **Learning rate schedulers** — Constant, Step Decay, Cosine Annealing, Linear Warmup
- **Attention math** — QKV projection, scaled dot-product attention, multi-head attention
- **Positional encodings** — Sinusoidal (additive), rotary (RoPE)
- **Normalization** — Layer, Batch, RMS normalization
- **Neural network layers** — Linear, Dropout, Sequential, MLP, TransformerBlock

## Module Overview

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `tensor` | N-D tensor with row-major layout, broadcasting, math ops | `Tensor::zeros`, `matmul`, `add`, `layer_norm`, `broadcast_to` |
| `activations` | Element-wise activation functions and derivatives | `relu`, `gelu`, `sigmoid`, `softmax`, `swish`, `mish` |
| `losses` | Regression and classification loss functions | `mse`, `cross_entropy`, `binary_cross_entropy_with_logits`, `huber` |
| `metrics` | Model evaluation metrics | `accuracy`, `f1`, `confusion_matrix`, `roc_auc`, `r_squared` |
| `optimizers` | Stateful gradient-based optimizers | `Sgd`, `Adam`, `AdamW`, `LrScheduler` |
| `attention` | Transformer attention math | `scaled_dot_product_attention`, `multi_head_attention`, `apply_rope` |
| `layers` | Neural network layers | `Linear`, `LayerNorm`, `BatchNorm`, `Dropout` |
| `models` | Model architectures | `Sequential`, `MLP`, `TransformerBlock` |
| `autograd` | Automatic differentiation | Backward pass computation |
| `data` | Data loading utilities | `DataLoader`, `Batch`, `train_test_split` |

## Installation

```toml
[dependencies]
mathverse-ai = { path = "crates/mathverse-ai" }
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

---

## Module Documentation

### Tensor

The core N-dimensional array type. Supports broadcasting (NumPy-style trailing dimensions), element-wise arithmetic, matrix multiply, reductions, normalization, and shape manipulation.

```rust
use mathverse_ai::Tensor;

// Constructors
let t = Tensor::zeros(&[2, 3]);
let t = Tensor::arange(0.0, 6.0, 1.0).reshape(&[2, 3]).unwrap();

// Broadcasting
let a = Tensor::new(&[2, 1], &[1.0, 2.0]).unwrap();
let b = Tensor::new(&[1, 3], &[10.0, 20.0, 30.0]).unwrap();
let c = a.add(&b).unwrap(); // shape [2, 3]

// Matrix multiply
let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
let b = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
let c = a.matmul(&b).unwrap(); // shape [2, 2]

// Layer normalization
let t = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
let norm = t.layer_norm(1e-5); // each row has mean ≈ 0
```

### Activations

All activation functions operate element-wise on tensors. Derivative functions (suffixed `_grad`) are provided for backpropagation.

```rust
use mathverse_ai::{Tensor, relu, gelu, sigmoid, softmax, swish};

let t = Tensor::new(&[5], &[-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
let r = relu(&t);   // [0.0, 0.0, 0.0, 1.0, 2.0]
let s = sigmoid(&t); // [0.1192, 0.2689, 0.5, 0.7311, 0.8808]

// Softmax (probabilities sum to 1)
let logits = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
let probs = softmax(&logits, 0).unwrap();
// [0.0900, 0.2447, 0.6652]
```

### Loss Functions

```rust
use mathverse_ai::{Tensor, mse, cross_entropy, binary_cross_entropy_with_logits};

// Regression: MSE
let pred = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
let target = Tensor::new(&[3], &[1.0, 2.0, 5.0]).unwrap();
println!("MSE: {:.4}", mse(&pred, &target).unwrap());

// Classification: Cross-Entropy
let logits = Tensor::new(&[2, 3], &[10.0, 1.0, 1.0, 1.0, 10.0, 1.0]).unwrap();
let targets = Tensor::new(&[2], &[0.0, 1.0]).unwrap();
println!("CE: {:.4}", cross_entropy(&logits, &targets).unwrap());
```

### Optimizers

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
```

### Attention

Transformer attention mechanisms: QKV projection, scaled dot-product attention, multi-head attention, and positional encodings.

```rust
use mathverse_ai::{Tensor, multi_head_attention, sinusoidal_encoding, apply_rope};

// Scaled dot-product attention
let q = Tensor::new(&[1, 2, 4], &[1.0; 8]).unwrap();
let k = Tensor::new(&[1, 2, 4], &[1.0; 8]).unwrap();
let v = Tensor::new(&[1, 2, 4], &[1.0; 8]).unwrap();
let (out, weights) = mathverse_ai::scaled_dot_product_attention(&q, &k, &v, None, None).unwrap();

// Sinusoidal positional encoding
let enc = sinusoidal_encoding(10, 64, 10000);
// enc shape: [1, 10, 64]

// RoPE (Rotary Position Embeddings)
let x = Tensor::randn(&[1, 8, 64]);
let y = apply_rope(&x, 8, 10000.0);
```

**Formulas:**

- Attention: `Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V`
- RoPE: `θ_i = pos / base^(2i/d)`, apply 2D rotation to each pair

---

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE).
