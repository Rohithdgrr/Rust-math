# Architecture

## Purpose

AI/ML mathematical primitives in pure Rust.

## Components

- `Tensor`: N-D array
- `activations`: Element-wise functions
- `losses`: Regression/classification losses
- `optimizers`: Gradient descent
- `attention`: Transformer math
- `layers`: Neural network layers

## Data Flow

```
User -> Tensor -> Activation/Loss -> Optimizer -> Updated Params
```
