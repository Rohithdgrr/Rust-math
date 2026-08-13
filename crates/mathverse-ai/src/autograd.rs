//! Automatic differentiation via reverse-mode autodiff (backpropagation).
//!
//! Wraps Tensor operations in a computation graph. Call `.backward()` on the
//! loss to compute gradients, then read them from `.grad` on each tensor.

use crate::tensor::Tensor;
use mathverse_core::error::MathError;
use std::cell::RefCell;

/// Assert two shapes match, panicking with a clear message if not.
///
/// Uses a non-debug assertion so shape mismatches fail loudly in release
/// builds too, before any `.unwrap()` on the underlying tensor operation.
macro_rules! assert_shape {
    ($a:expr, $b:expr) => {
        assert_eq!(
            $a.shape,
            $b.shape,
            "shape mismatch: lhs {:?} vs rhs {:?}",
            $a.shape,
            $b.shape
        );
    };
}

/// Assert a matmul dimension match, panicking with a clear message if not.
///
/// Uses a non-debug assertion so mismatches fail loudly in release builds too.
macro_rules! assert_matmul {
    ($a:expr, $b:expr) => {
        assert_eq!(
            $a.shape[1],
            $b.shape[0],
            "matmul dimension mismatch: lhs cols {} vs rhs rows {}",
            $a.shape[1],
            $b.shape[0]
        );
    };
}

thread_local! {
    static GRAPH: RefCell<Vec<GraphEntry>> = const { RefCell::new(Vec::new()) };
    static GRAD_REGISTRY: RefCell<Vec<Option<Tensor>>> = RefCell::new(Vec::new());
}

enum BackwardOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Matmul,
    ReLU,
    Sigmoid,
    Tanh,
    Sum,
    MseLoss,
}

enum GraphEntry {
    Tensor(Tensor),
    Op { tensor: Tensor, inputs: Vec<usize>, op: BackwardOp },
}

fn tensor_from_entry(entry: &GraphEntry) -> Tensor {
    match entry {
        GraphEntry::Tensor(t) => t.clone(),
        GraphEntry::Op { tensor, .. } => tensor.clone(),
    }
}

fn sum_keepdim(t: &Tensor, axis: usize) -> Result<Tensor, MathError> {
    if axis >= t.shape.len() {
        return Err(MathError::InvalidArgument("axis out of range"));
    }
    let mut out_shape = t.shape.clone();
    let axis_size = out_shape[axis];
    out_shape[axis] = 1;

    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis + 1..].iter().product();
    let mut out_data = Vec::with_capacity(outer * inner);

    for i in 0..outer {
        for j in 0..inner {
            let mut sum = 0.0;
            for k in 0..axis_size {
                sum += t.data[i * axis_size * inner + k * inner + j];
            }
            out_data.push(sum);
        }
    }

    Ok(Tensor { shape: out_shape, data: out_data })
}

fn reduce_broadcast_grad(mut grad: Tensor, target_shape: &[usize]) -> Result<Tensor, MathError> {
    if grad.shape == target_shape {
        return Ok(grad);
    }

    let nd = grad.shape.len().max(target_shape.len());
    let mut grad_shape = vec![1; nd];
    grad_shape[nd - grad.shape.len()..].copy_from_slice(&grad.shape);
    let mut target_padded = vec![1; nd];
    target_padded[nd - target_shape.len()..].copy_from_slice(target_shape);

    for axis in 0..nd {
        let g_dim = grad_shape[axis];
        let t_dim = target_padded[axis];
        if g_dim == t_dim {
            continue;
        }
        if t_dim == 1 && g_dim > 1 {
            grad = sum_keepdim(&grad, axis)?;
            grad_shape[axis] = 1;
        } else {
            return Err(MathError::DimensionMismatch);
        }
    }

    grad.reshape(target_shape)
}

/// Clear the computation graph and gradient registry.
/// Call this before starting a new forward pass to avoid memory leaks.
pub fn clear_graph() {
    GRAPH.with(|g| g.borrow_mut().clear());
    GRAD_REGISTRY.with(|g| g.borrow_mut().clear());
}

/// A tensor with gradient tracking.
#[derive(Clone)]
pub struct GradTensor {
    /// The forward-pass value of this node.
    pub tensor: Tensor,
    /// Accumulated gradient after [`backward`]; `None` until a backward pass.
    pub grad: Option<Tensor>,
    node_id: usize,
}

impl GradTensor {
    /// Create a leaf tensor (no gradient tracking).
    pub fn new(tensor: Tensor) -> Self {
        let id = GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let id = g.len();
            g.push(GraphEntry::Tensor(tensor.clone()));
            id
        });
        GRAD_REGISTRY.with(|g| {
            let mut g = g.borrow_mut();
            if g.len() <= id {
                g.resize_with(id + 1, || None);
            }
        });
        Self { tensor, grad: None, node_id: id }
    }

    /// Create a leaf tensor from data + shape.
    pub fn from_data(shape: &[usize], data: Vec<f64>) -> Self {
        let t = Tensor::from_vec(shape, data);
        assert!(t.is_ok(), "from_vec failed for shape {:?}", shape);
        Self::new(t.expect("shape check above"))
    }

    /// Zero the gradient.
    pub fn zero_grad(&mut self) { self.grad = None; }

    /// Add two tracked tensors.
    pub fn add(&self, other: &GradTensor) -> GradTensor {
        add(self, other)
    }

    /// Subtract two tracked tensors.
    pub fn sub(&self, other: &GradTensor) -> GradTensor {
        sub(self, other)
    }

    /// Multiply two tracked tensors.
    pub fn mul(&self, other: &GradTensor) -> GradTensor {
        mul(self, other)
    }

    /// Divide two tracked tensors.
    pub fn div(&self, other: &GradTensor) -> GradTensor {
        div(self, other)
    }

    /// Negate the tracked tensor.
    pub fn neg(&self) -> GradTensor {
        neg(self)
    }

    /// Sigmoid activation on the tracked tensor.
    pub fn sigmoid(&self) -> GradTensor {
        sigmoid(self)
    }

    /// Tanh activation on the tracked tensor.
    pub fn tanh(&self) -> GradTensor {
        tanh(self)
    }

    /// Matrix multiply two tracked tensors.
    pub fn matmul(&self, other: &GradTensor) -> GradTensor {
        matmul(self, other)
    }

    /// ReLU activation on the tracked tensor.
    pub fn relu(&self) -> GradTensor {
        relu_op(self)
    }

    /// Sum all elements in the tracked tensor.
    pub fn sum(&self) -> GradTensor {
        sum(self)
    }

    /// Backward pass from this tensor.
    pub fn backward(&mut self, scale: f64) {
        backward(self, scale)
    }

    /// Return accumulated gradient after backward.
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }
}

/// Add two GradTensors (tracked).
pub fn add(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_shape!(a.tensor, b.tensor);
    let out = a.tensor.add(&b.tensor).unwrap();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id, b.node_id], op: BackwardOp::Add });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Multiply two GradTensors (tracked).
pub fn mul(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_shape!(a.tensor, b.tensor);
    let out = a.tensor.mul(&b.tensor).unwrap();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id, b.node_id], op: BackwardOp::Mul });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Element-wise subtraction (tracked).
pub fn sub(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_shape!(a.tensor, b.tensor);
    let out = a.tensor.sub(&b.tensor).unwrap();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id, b.node_id], op: BackwardOp::Sub });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Element-wise division (tracked).
pub fn div(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_shape!(a.tensor, b.tensor);
    let out = a.tensor.div(&b.tensor).unwrap();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id, b.node_id], op: BackwardOp::Div });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Negate a tensor (tracked).
pub fn neg(a: &GradTensor) -> GradTensor {
    let out = a.tensor.neg();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id], op: BackwardOp::Neg });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Sigmoid activation (tracked).
pub fn sigmoid(a: &GradTensor) -> GradTensor {
    let out = crate::activations::sigmoid(&a.tensor);
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id], op: BackwardOp::Sigmoid });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Tanh activation (tracked).
pub fn tanh(a: &GradTensor) -> GradTensor {
    let out = crate::activations::tanh(&a.tensor);
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id], op: BackwardOp::Tanh });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Matrix multiply (tracked).
pub fn matmul(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_matmul!(a.tensor, b.tensor);
    let out = a.tensor.matmul(&b.tensor).unwrap();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id, b.node_id], op: BackwardOp::Matmul });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// ReLU (tracked).
pub fn relu_op(a: &GradTensor) -> GradTensor {
    let out = crate::activations::relu(&a.tensor);
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id], op: BackwardOp::ReLU });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Sum all elements (tracked).
pub fn sum(a: &GradTensor) -> GradTensor {
    let val = a.tensor.sum();
    let out = Tensor::scalar(val);
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![a.node_id], op: BackwardOp::Sum });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// MSE loss (tracked).
pub fn mse_loss(pred: &GradTensor, target: &GradTensor) -> GradTensor {
    assert_shape!(pred.tensor, target.tensor);
    let val = crate::losses::mse(&pred.tensor, &target.tensor).unwrap();
    let out = Tensor::scalar(val);
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor: out.clone(), inputs: vec![pred.node_id, target.node_id], op: BackwardOp::MseLoss });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= out_id {
            g.resize_with(out_id + 1, || None);
        }
    });
    GradTensor { tensor: out, grad: None, node_id: out_id }
}

/// Backward pass from a scalar loss tensor. Computes gradients for all nodes
/// in the graph and stores them in the gradient registry.
pub fn backward(loss: &mut GradTensor, scale: f64) {
    GRAPH.with(|g| {
        let g = g.borrow();
        let n = g.len();
        let mut grads: Vec<Option<Tensor>> = vec![None; n];

        // Seed: d(loss)/d(loss) = 1
        grads[loss.node_id] = Some(Tensor::scalar(scale));

        // Reverse pass
        for i in (0..n).rev() {
            let grad_out = match &grads[i] {
                Some(g) => g.clone(),
                None => continue,
            };

            match &g[i] {
                GraphEntry::Tensor(_) => {}
                GraphEntry::Op { inputs, op, .. } => {
                    let input_ids = inputs.clone();
                    let input_grads = match op {
                        BackwardOp::Add => vec![
                            reduce_broadcast_grad(grad_out.clone(), &tensor_from_entry(&g[input_ids[0]]).shape).unwrap(),
                            reduce_broadcast_grad(grad_out.clone(), &tensor_from_entry(&g[input_ids[1]]).shape).unwrap(),
                        ],
                        BackwardOp::Sub => vec![
                            reduce_broadcast_grad(grad_out.clone(), &tensor_from_entry(&g[input_ids[0]]).shape).unwrap(),
                            reduce_broadcast_grad(grad_out.clone().neg(), &tensor_from_entry(&g[input_ids[1]]).shape).unwrap(),
                        ],
                        BackwardOp::Mul => {
                            let a = tensor_from_entry(&g[input_ids[0]]);
                            let b = tensor_from_entry(&g[input_ids[1]]);
                            let b_expanded = b.broadcast_to(&grad_out.shape).unwrap();
                            let a_expanded = a.broadcast_to(&grad_out.shape).unwrap();
                            let grad_a = grad_out.mul(&b_expanded).unwrap();
                            let grad_b = grad_out.mul(&a_expanded).unwrap();
                            vec![
                                reduce_broadcast_grad(grad_a, &a.shape).unwrap(),
                                reduce_broadcast_grad(grad_b, &b.shape).unwrap(),
                            ]
                        }
                        BackwardOp::Div => {
                            let a = tensor_from_entry(&g[input_ids[0]]);
                            let b = tensor_from_entry(&g[input_ids[1]]);
                            let b_expanded = b.broadcast_to(&grad_out.shape).unwrap();
                            let a_expanded = a.broadcast_to(&grad_out.shape).unwrap();
                            let inv_b = Tensor::ones(&b_expanded.shape).div(&b_expanded).unwrap();
                            let grad_a = grad_out.mul(&inv_b).unwrap();
                            let grad_b = grad_out
                                .mul(&a_expanded.mul_scalar(-1.0))
                                .unwrap()
                                .mul(&inv_b.mul(&inv_b).unwrap())
                                .unwrap();
                            vec![
                                reduce_broadcast_grad(grad_a, &a.shape).unwrap(),
                                reduce_broadcast_grad(grad_b, &b.shape).unwrap(),
                            ]
                        }
                        BackwardOp::Neg => vec![grad_out.neg()],
                        BackwardOp::Matmul => {
                            let a = tensor_from_entry(&g[input_ids[0]]);
                            let b = tensor_from_entry(&g[input_ids[1]]);
                            let bt = b.transpose().unwrap();
                            let at = a.transpose().unwrap();
                            assert_matmul!(grad_out, bt);
                            assert_matmul!(at, grad_out);
                            vec![grad_out.matmul(&bt).unwrap(), at.matmul(&grad_out).unwrap()]
                        }
                        BackwardOp::ReLU => {
                            let input = tensor_from_entry(&g[input_ids[0]]);
                            let mask = crate::activations::relu_grad(&input);
                            assert_shape!(grad_out, mask);
                            vec![grad_out.mul(&mask).unwrap()]
                        }
                        BackwardOp::Sigmoid => {
                            let input = tensor_from_entry(&g[input_ids[0]]);
                            let sig = crate::activations::sigmoid(&input);
                            let grad_activation = sig.mul(&Tensor::ones(&sig.shape).sub(&sig).unwrap()).unwrap();
                            vec![grad_out.mul(&grad_activation).unwrap()]
                        }
                        BackwardOp::Tanh => {
                            let input = tensor_from_entry(&g[input_ids[0]]);
                            let tanh = crate::activations::tanh(&input);
                            let grad_activation = Tensor::ones(&tanh.shape)
                                .sub(&tanh.mul(&tanh).unwrap())
                                .unwrap();
                            vec![grad_out.mul(&grad_activation).unwrap()]
                        }
                        BackwardOp::Sum => {
                            let input = tensor_from_entry(&g[input_ids[0]]);
                            vec![Tensor::full(&input.shape, grad_out.data[0])]
                        }
                        BackwardOp::MseLoss => {
                            let pred = tensor_from_entry(&g[input_ids[0]]);
                            let target = tensor_from_entry(&g[input_ids[1]]);
                            assert_shape!(pred, target);
                            let g = crate::losses::mse_grad(&pred, &target).unwrap();
                            vec![g, Tensor::zeros(&target.shape)]
                        }
                    };
                    for (j, &input_id) in input_ids.iter().enumerate() {
                        if j < input_grads.len() {
                            grads[input_id] = Some(match &grads[input_id] {
                                Some(existing) => {
                                    assert_shape!(existing, &input_grads[j]);
                                    existing.add(&input_grads[j]).unwrap()
                                }
                                None => input_grads[j].clone(),
                            });
                        }
                    }
                }
            }
        }

        // Write computed gradients back to the registry
        GRAD_REGISTRY.with(|reg| {
            let mut reg = reg.borrow_mut();
            for (id, grad) in grads.into_iter().enumerate() {
                if let Some(g) = grad {
                    if id < reg.len() {
                        reg[id] = Some(g);
                    }
                }
            }
        });
    });

    // Update the loss tensor's grad
    loss.grad = Some(Tensor::scalar(scale));
}

/// Look up the gradient for a node by its node_id.
/// Returns `None` if no gradient was computed for this node.
pub fn get_grad(node_id: usize) -> Option<Tensor> {
    GRAD_REGISTRY.with(|reg| {
        let reg = reg.borrow();
        reg.get(node_id).cloned().flatten()
    })
}

/// Get the node_id for a GradTensor (for use with [`get_grad`]).
pub fn node_id(gt: &GradTensor) -> usize {
    gt.node_id
}

#[cfg(test)]
mod tests {
    use super::*;

    const E: f64 = 1e-9;

    #[test]
    fn add_backward() {
        clear_graph();
        let a = GradTensor::from_data(&[2], vec![1.0, 2.0]);
        let b = GradTensor::from_data(&[2], vec![3.0, 4.0]);
        let c = add(&a, &b);
        let mut loss = sum(&c);
        backward(&mut loss, 1.0);
        assert_eq!(get_grad(a.node_id).unwrap().data, vec![1.0, 1.0]);
        assert_eq!(get_grad(b.node_id).unwrap().data, vec![1.0, 1.0]);
        assert_eq!(loss.grad.unwrap().data, vec![1.0]);
    }

#[test]
    fn mul_add_backward() {
        clear_graph();
        let a = GradTensor::from_data(&[2], vec![1.0, 2.0]);
        let b = GradTensor::from_data(&[2], vec![3.0, 4.0]);
        let c = sum(&mul(&a, &b));
        let mut loss = sum(&c);
        backward(&mut loss, 1.0);
        // d/dx (x·y) = y, d/dy (x·y) = x
        assert_eq!(get_grad(a.node_id).unwrap().data, vec![3.0, 4.0]);
        assert_eq!(get_grad(b.node_id).unwrap().data, vec![1.0, 2.0]);
    }

    #[test]
    fn neg_backward() {
        clear_graph();
        let a = GradTensor::from_data(&[2], vec![1.0, -2.0]);
        let c = neg(&a);
        let mut loss = sum(&c);
        backward(&mut loss, 1.0);
        assert_eq!(get_grad(a.node_id).unwrap().data, vec![-1.0, -1.0]);
    }

    #[test]
    fn sigmoid_backward() {
        clear_graph();
        let a = GradTensor::from_data(&[2], vec![0.0, 2.0]);
        let c = sigmoid(&a);
        let mut loss = sum(&c);
        backward(&mut loss, 1.0);
        let grad = get_grad(a.node_id).unwrap().data;
        assert!((grad[0] - 0.25).abs() < E);
        assert!((grad[1] - (0.8807970779778823 * (1.0 - 0.8807970779778823))).abs() < 1e-8);
    }

    #[test]
    fn tanh_backward() {
        clear_graph();
        let a = GradTensor::from_data(&[2], vec![0.0, 1.0]);
        let c = tanh(&a);
        let mut loss = sum(&c);
        backward(&mut loss, 1.0);
        let grad = get_grad(a.node_id).unwrap().data;
        let tanh_val: f64 = 0.7615941559557649;
        assert!((grad[0] - 1.0).abs() < E);
        assert!((grad[1] - (1.0 - tanh_val.powi(2))).abs() < 1e-8);
    }

    #[test]
    fn sub_backward() {
        clear_graph();
        let a = GradTensor::from_data(&[2], vec![5.0, 7.0]);
        let b = GradTensor::from_data(&[2], vec![2.0, 3.0]);
        let c = sub(&a, &b);
        let mut loss = sum(&c);
        backward(&mut loss, 1.0);
        assert_eq!(get_grad(a.node_id).unwrap().data, vec![1.0, 1.0]);
        assert_eq!(get_grad(b.node_id).unwrap().data, vec![-1.0, -1.0]);
    }

#[test]
    fn mse_loss_backward() {
        clear_graph();
        let pred = GradTensor::from_data(&[2], vec![1.0, 3.0]);
        let target = GradTensor::from_data(&[2], vec![2.0, 1.0]);
        let mut loss = mse_loss(&pred, &target);
        backward(&mut loss, 1.0);
        // d/dx MSE = 2*(x-t)/n → 2*(1-2)/2=-1, 2*(3-1)/2=2
        assert_eq!(get_grad(pred.node_id).unwrap().data, vec![-1.0, 2.0]);
        assert_eq!(get_grad(target.node_id).unwrap().data, vec![0.0, 0.0]);
    }
}
