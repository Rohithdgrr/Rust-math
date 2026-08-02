//! Automatic differentiation via reverse-mode autodiff (backpropagation).
//!
//! Wraps Tensor operations in a computation graph. Call `.backward()` on the
//! loss to compute gradients, then read them from `.grad` on each tensor.

use crate::tensor::Tensor;
use std::cell::RefCell;

thread_local! {
    static GRAPH: RefCell<Vec<GraphEntry>> = const { RefCell::new(Vec::new()) };
    static GRAD_REGISTRY: RefCell<Vec<Option<Tensor>>> = RefCell::new(Vec::new());
}

enum GraphEntry {
    Tensor(Tensor),
    Op { inputs: Vec<usize>, backward_fn: usize },
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
    pub tensor: Tensor,
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
        Self::new(Tensor::from_vec(shape, data).unwrap())
    }

    /// Zero the gradient.
    pub fn zero_grad(&mut self) { self.grad = None; }
}

/// Add two GradTensors (tracked).
pub fn add(a: &GradTensor, b: &GradTensor) -> GradTensor {
    let out = a.tensor.add(&b.tensor).unwrap();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { inputs: vec![a.node_id, b.node_id], backward_fn: 0 });
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
    let out = a.tensor.mul(&b.tensor).unwrap();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { inputs: vec![a.node_id, b.node_id], backward_fn: 1 });
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
    let out = a.tensor.matmul(&b.tensor).unwrap();
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { inputs: vec![a.node_id, b.node_id], backward_fn: 2 });
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
        g.push(GraphEntry::Op { inputs: vec![a.node_id], backward_fn: 3 });
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
        g.push(GraphEntry::Op { inputs: vec![a.node_id], backward_fn: 4 });
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
    let val = crate::losses::mse(&pred.tensor, &target.tensor).unwrap();
    let out = Tensor::scalar(val);
    let out_id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { inputs: vec![pred.node_id, target.node_id], backward_fn: 5 });
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
                GraphEntry::Op { inputs, backward_fn } => {
                    let input_grads = match backward_fn {
                        // add backward: grad flows to both
                        0 => vec![grad_out.clone(), grad_out.clone()],
                        // mul backward: grad * other, grad * self
                        1 => {
                            let a = match &g[inputs[1]] {
                                GraphEntry::Tensor(t) => t.clone(),
                                _ => Tensor::zeros(&[1]),
                            };
                            let b = match &g[inputs[0]] {
                                GraphEntry::Tensor(t) => t.clone(),
                                _ => Tensor::zeros(&[1]),
                            };
                            vec![grad_out.mul(&a).unwrap(), grad_out.mul(&b).unwrap()]
                        }
                        // matmul backward
                        2 => {
                            let a = match &g[inputs[0]] {
                                GraphEntry::Tensor(t) => t.clone(),
                                _ => Tensor::zeros(&[1]),
                            };
                            let b = match &g[inputs[1]] {
                                GraphEntry::Tensor(t) => t.clone(),
                                _ => Tensor::zeros(&[1]),
                            };
                            let bt = b.transpose().unwrap();
                            let at = a.transpose().unwrap();
                            vec![grad_out.matmul(&bt).unwrap(), at.matmul(&grad_out).unwrap()]
                        }
                        // relu backward
                        3 => {
                            let input = match &g[inputs[0]] {
                                GraphEntry::Tensor(t) => t.clone(),
                                _ => Tensor::zeros(&[1]),
                            };
                            let mask = crate::activations::relu_grad(&input);
                            vec![grad_out.mul(&mask).unwrap()]
                        }
                        // sum backward: broadcast gradient to input shape
                        4 => {
                            let input = match &g[inputs[0]] {
                                GraphEntry::Tensor(t) => t.clone(),
                                _ => Tensor::zeros(&[1]),
                            };
                            vec![Tensor::full(&input.shape, grad_out.data[0])]
                        }
                        // mse backward: 2*(pred-target)/n
                        5 => {
                            let pred = match &g[inputs[0]] {
                                GraphEntry::Tensor(t) => t.clone(),
                                _ => Tensor::zeros(&[1]),
                            };
                            let target = match &g[inputs[1]] {
                                GraphEntry::Tensor(t) => t.clone(),
                                _ => Tensor::zeros(&[1]),
                            };
                            let g = crate::losses::mse_grad(&pred, &target).unwrap();
                            vec![g, Tensor::zeros(&target.shape)]
                        }
                        _ => vec![],
                    };
                    for (j, &input_id) in inputs.iter().enumerate() {
                        if j < input_grads.len() {
                            grads[input_id] = Some(match &grads[input_id] {
                                Some(existing) => existing.add(&input_grads[j]).unwrap(),
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
