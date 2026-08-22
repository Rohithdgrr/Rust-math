//! Automatic differentiation via reverse-mode autodiff (backpropagation).
//!
//! Operations are recorded in an explicit, user-owned [`ComputationGraph`].
//! Create a graph, register leaf variables with [`ComputationGraph::variable`],
//! build expressions out of [`GradTensor`] ops, then run
//! [`ComputationGraph::backward`] and read gradients with
//! [`ComputationGraph::grad_of`]. There is no global state: any number of
//! independent graphs may coexist and be differentiated independently.
//!
//! # Examples
//!
//! ```
//! use mathverse_ai::autograd::ComputationGraph;
//! use mathverse_ai::tensor::Tensor;
//!
//! let mut g = ComputationGraph::new();
//! let x = g.variable(Tensor::from_vec(&[2], vec![1.0, 2.0]).unwrap());
//! let y = g.variable(Tensor::from_vec(&[2], vec![3.0, 4.0]).unwrap());
//! let loss = x.mul(&y).sum();
//! g.backward(&loss);
//! assert_eq!(g.grad_of(&x).unwrap().as_slice(), &[3.0, 4.0]);
//! ```

use crate::activations::{relu, relu_grad, sigmoid, tanh};
use crate::losses::{mse, mse_grad};
use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};
use std::cell::RefCell;
use std::rc::Rc;

/// Assert two shapes match, panicking with a clear message if not.
///
/// Uses a non-debug assertion so shape mismatches fail loudly in release
/// builds too, before any `.unwrap()` on the underlying tensor operation.
macro_rules! assert_shape {
    ($a:expr, $b:expr) => {
        assert_eq!(
            $a.shape(),
            $b.shape(),
            "shape mismatch: lhs {:?} vs rhs {:?}",
            $a.shape(),
            $b.shape()
        );
    };
}

/// Assert a matmul dimension match, panicking with a clear message if not.
///
/// Uses a non-debug assertion so mismatches fail loudly in release builds too.
macro_rules! assert_matmul {
    ($a:expr, $b:expr) => {
        assert_eq!(
            $a.shape()[1],
            $b.shape()[0],
            "matmul dimension mismatch: lhs cols {} vs rhs rows {}",
            $a.shape()[1],
            $b.shape()[0]
        );
    };
}

#[derive(Clone, Copy)]
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

impl GraphEntry {
    fn tensor(&self) -> &Tensor {
        match self {
            GraphEntry::Tensor(t) => t,
            GraphEntry::Op { tensor, .. } => tensor,
        }
    }
}

struct GraphData {
    entries: Vec<GraphEntry>,
    grads: Vec<Option<Tensor>>,
}

/// A user-owned computation graph recording forward operations for
/// reverse-mode automatic differentiation.
///
/// The graph is a cheaply clonable handle (`Rc<RefCell<...>>` internally):
/// clones share the same node storage, which is how [`GradTensor`] values keep
/// their defining graph alive. Nodes are stored in insertion order, which is a
/// valid topological order for the backward pass.
#[derive(Clone)]
pub struct ComputationGraph {
    inner: Rc<RefCell<GraphData>>,
}

impl ComputationGraph {
    /// Create an empty computation graph.
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphData { entries: Vec::new(), grads: Vec::new() })),
        }
    }

    /// Register a leaf variable (a node with no inputs).
    ///
    /// Returns a [`GradTensor`] handle bound to this graph.
    pub fn variable(&mut self, t: Tensor) -> GradTensor {
        let node = {
            let mut d = self.inner.borrow_mut();
            let node = d.entries.len();
            d.entries.push(GraphEntry::Tensor(t));
            d.grads.push(None);
            node
        };
        GradTensor { graph: self.clone(), node }
    }

    /// Register a leaf variable from data + shape.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not match the product of `shape`. For a
    /// fallible constructor use [`ComputationGraph::try_variable_from_data`].
    pub fn variable_from_data(&mut self, shape: &[usize], data: Vec<f64>) -> GradTensor {
        self.try_variable_from_data(shape, data)
            .expect("variable_from_data: data length does not match shape")
    }

    /// Fallible variant of [`ComputationGraph::variable_from_data`].
    pub fn try_variable_from_data(
        &mut self,
        shape: &[usize],
        data: Vec<f64>,
    ) -> MathResult<GradTensor> {
        let t = Tensor::from_vec(shape, data)?;
        Ok(self.variable(t))
    }

    /// Backward pass from a scalar loss node, seeding its gradient with 1.0.
    ///
    /// Computes gradients for all nodes reachable in the reverse pass and
    /// stores them in the graph; read them back with [`Self::grad_of`].
    ///
    /// # Panics
    ///
    /// Panics with the underlying [`MathError`] if gradient computation fails;
    /// prefer [`Self::try_backward`] in code that must not panic.
    pub fn backward(&mut self, loss: &GradTensor) {
        if let Err(e) = self.try_backward_scaled(loss, 1.0) {
            panic!("backward failed: {e}");
        }
    }

    /// Fallible backward pass seeding the loss gradient with 1.0.
    pub fn try_backward(&self, loss: &GradTensor) -> MathResult<()> {
        self.try_backward_scaled(loss, 1.0)
    }

    /// Fallible backward pass seeding the loss gradient with `scale`.
    pub fn try_backward_scaled(&self, loss: &GradTensor, scale: f64) -> MathResult<()> {
        let mut data = self.inner.borrow_mut();

        let n = data.entries.len();
        assert!(loss.node < n, "backward: loss node does not belong to this graph");
        let mut grads: Vec<Option<Tensor>> = vec![None; n];

        // Seed: d(loss)/d(loss) = scale, shaped like the loss itself so that
        // downstream shape checks (e.g. matmul backward) see consistent ranks.
        let loss_shape = data.entries[loss.node].tensor().shape().to_vec();
        grads[loss.node] = Some(Tensor::full(&loss_shape, scale));

        // Reverse pass over the graph (insertion order is topological).
        for i in (0..n).rev() {
            let grad_out = match &grads[i] {
                Some(gv) => gv.clone(),
                None => continue,
            };
            let (input_ids, op) = match &data.entries[i] {
                GraphEntry::Tensor(_) => continue,
                GraphEntry::Op { inputs, op, .. } => (inputs.clone(), *op),
            };
            let input =
                |id: usize| -> Tensor { data.entries[id].tensor().clone() };

            let input_grads = match op {
                BackwardOp::Add => vec![
                    reduce_broadcast_grad(&grad_out, input(input_ids[0]).shape())?,
                    reduce_broadcast_grad(&grad_out, input(input_ids[1]).shape())?,
                ],
                BackwardOp::Sub => vec![
                    reduce_broadcast_grad(&grad_out, input(input_ids[0]).shape())?,
                    reduce_broadcast_grad(&grad_out.neg(), input(input_ids[1]).shape())?,
                ],
                BackwardOp::Mul => {
                    let a = input(input_ids[0]);
                    let b = input(input_ids[1]);
                    let b_expanded = b.broadcast_to(&grad_out.shape()).map_err(|_| {
                        autograd_error("autograd mul: cannot broadcast operand b to grad shape")
                    })?;
                    let a_expanded = a.broadcast_to(&grad_out.shape()).map_err(|_| {
                        autograd_error("autograd mul: cannot broadcast operand a to grad shape")
                    })?;
                    let grad_a = grad_out
                        .mul(&b_expanded)
                        .map_err(|_| autograd_error("autograd mul: grad_a computation failed"))?;
                    let grad_b = grad_out
                        .mul(&a_expanded)
                        .map_err(|_| autograd_error("autograd mul: grad_b computation failed"))?;
                    vec![
                        reduce_broadcast_grad(&grad_a, a.shape())?,
                        reduce_broadcast_grad(&grad_b, b.shape())?,
                    ]
                }
                BackwardOp::Div => {
                    let a = input(input_ids[0]);
                    let b = input(input_ids[1]);
                    let b_expanded = b.broadcast_to(&grad_out.shape()).map_err(|_| {
                        autograd_error("autograd div: cannot broadcast operand b to grad shape")
                    })?;
                    let a_expanded = a.broadcast_to(&grad_out.shape()).map_err(|_| {
                        autograd_error("autograd div: cannot broadcast operand a to grad shape")
                    })?;
                    let inv_b = Tensor::ones(b_expanded.shape())
                        .div(&b_expanded)
                        .map_err(|_| autograd_error("autograd div: 1/b computation failed"))?;
                    let grad_a = grad_out
                        .mul(&inv_b)
                        .map_err(|_| autograd_error("autograd div: grad_a computation failed"))?;
                    let inv_b_sq = inv_b
                        .mul(&inv_b)
                        .map_err(|_| autograd_error("autograd div: (1/b)² computation failed"))?;
                    let grad_b = grad_out
                        .mul(&a_expanded.mul_scalar(-1.0))
                        .map_err(|_| autograd_error("autograd div: grad_b numerator failed"))?
                        .mul(&inv_b_sq)
                        .map_err(|_| autograd_error("autograd div: grad_b computation failed"))?;
                    vec![
                        reduce_broadcast_grad(&grad_a, a.shape())?,
                        reduce_broadcast_grad(&grad_b, b.shape())?,
                    ]
                }
                BackwardOp::Neg => vec![grad_out.neg()],
                BackwardOp::Matmul => {
                    let a = input(input_ids[0]);
                    let b = input(input_ids[1]);
                    let bt = b
                        .transpose()
                        .map_err(|_| autograd_error("autograd matmul: transpose failed"))?;
                    let at = a
                        .transpose()
                        .map_err(|_| autograd_error("autograd matmul: transpose failed"))?;
                    assert_matmul!(grad_out, bt);
                    assert_matmul!(at, grad_out);
                    let ga = grad_out
                        .matmul(&bt)
                        .map_err(|_| autograd_error("autograd matmul: grad wrt a failed"))?;
                    let gb = at
                        .matmul(&grad_out)
                        .map_err(|_| autograd_error("autograd matmul: grad wrt b failed"))?;
                    vec![ga, gb]
                }
                BackwardOp::ReLU => {
                    let input_t = input(input_ids[0]);
                    let mask = relu_grad(&input_t);
                    assert_shape!(grad_out, mask);
                    let gated = grad_out
                        .mul(&mask)
                        .map_err(|_| autograd_error("autograd relu: mask multiply failed"))?;
                    vec![gated]
                }
                BackwardOp::Sigmoid => {
                    let input_t = input(input_ids[0]);
                    let sig = sigmoid(&input_t);
                    let ones = Tensor::ones(sig.shape());
                    let grad_activation = sig
                        .mul(&ones.sub(&sig).map_err(|_| {
                            autograd_error("autograd sigmoid: 1−σ failed")
                        })?)
                        .map_err(|_| autograd_error("autograd sigmoid: σ(1−σ) failed"))?;
                    let gated = grad_out
                        .mul(&grad_activation)
                        .map_err(|_| autograd_error("autograd sigmoid: chain multiply failed"))?;
                    vec![gated]
                }
                BackwardOp::Tanh => {
                    let input_t = input(input_ids[0]);
                    let tanh_v = tanh(&input_t);
                    let ones = Tensor::ones(tanh_v.shape());
                    let sq = tanh_v
                        .mul(&tanh_v)
                        .map_err(|_| autograd_error("autograd tanh: tanh² failed"))?;
                    let grad_activation = ones
                        .sub(&sq)
                        .map_err(|_| autograd_error("autograd tanh: 1−tanh² failed"))?;
                    let gated = grad_out
                        .mul(&grad_activation)
                        .map_err(|_| autograd_error("autograd tanh: chain multiply failed"))?;
                    vec![gated]
                }
                BackwardOp::Sum => {
                    let input_t = input(input_ids[0]);
                    vec![Tensor::full(input_t.shape(), grad_out.get_flat(0))]
                }
                BackwardOp::MseLoss => {
                    let pred = input(input_ids[0]);
                    let target = input(input_ids[1]);
                    assert_shape!(pred, target);
                    let gd = mse_grad(&pred, &target)
                        .map_err(|_| autograd_error("autograd mse_loss: mse_grad failed"))?;
                    vec![gd, Tensor::zeros(target.shape())]
                }
            };

            for (j, &input_id) in input_ids.iter().enumerate() {
                if j < input_grads.len() {
                    grads[input_id] = Some(match &grads[input_id] {
                        Some(existing) => {
                            assert_shape!(existing, &input_grads[j]);
                            existing.add(&input_grads[j]).map_err(|_| {
                                autograd_error("autograd accumulate: gradient accumulation failed")
                            })?
                        }
                        None => input_grads[j].clone(),
                    });
                }
            }
        }

        data.grads = grads;
        Ok(())
    }

    /// Return the accumulated gradient for a node after [`Self::backward`].
    ///
    /// Returns `None` before a backward pass or if `g` belongs to a different
    /// graph. The gradient tensor is cloned out of the graph.
    pub fn grad_of(&self, g: &GradTensor) -> Option<Tensor> {
        if !Rc::ptr_eq(&g.graph.inner, &self.inner) {
            return None;
        }
        let d = self.inner.borrow();
        d.grads.get(g.node).and_then(|slot| slot.as_ref()).cloned()
    }

    /// Zero every stored gradient (node structure is preserved).
    pub fn zero_grad(&mut self) {
        let mut d = self.inner.borrow_mut();
        for slot in &mut d.grads {
            *slot = None;
        }
    }

    /// Clear all nodes and gradients, resetting the graph to empty.
    pub fn clear(&mut self) {
        let mut d = self.inner.borrow_mut();
        d.entries.clear();
        d.grads.clear();
    }

    /// Number of nodes recorded in the graph.
    pub fn len(&self) -> usize {
        self.inner.borrow().entries.len()
    }

    /// True if no nodes are recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push_op(&self, tensor: Tensor, inputs: Vec<usize>, op: BackwardOp) -> usize {
        let mut d = self.inner.borrow_mut();
        let id = d.entries.len();
        d.entries.push(GraphEntry::Op { tensor, inputs, op });
        d.grads.push(None);
        id
    }

    fn node_value(&self, node: usize) -> Tensor {
        self.inner.borrow().entries[node].tensor().clone()
    }
}

fn sum_keepdim(t: &Tensor, axis: usize) -> Result<Tensor, MathError> {
    if axis >= t.shape().len() {
        return Err(MathError::InvalidArgument("axis out of range"));
    }
    let mut out_shape = t.shape().to_vec();
    let axis_size = out_shape[axis];
    out_shape[axis] = 1;

    let outer: usize = t.shape()[..axis].iter().product();
    let inner: usize = t.shape()[axis + 1..].iter().product();
    let mut out_data = Vec::with_capacity(outer * inner);

    for i in 0..outer {
        for j in 0..inner {
            let mut sum = 0.0;
            for k in 0..axis_size {
                sum += t.get_flat(i * axis_size * inner + k * inner + j);
            }
            out_data.push(sum);
        }
    }

    Ok(Tensor { shape: out_shape, data: out_data })
}

fn reduce_broadcast_grad(grad: &Tensor, target_shape: &[usize]) -> Result<Tensor, MathError> {
    let mut grad = grad.clone();
    if grad.shape() == target_shape {
        return Ok(grad);
    }

    let nd = grad.shape().len().max(target_shape.len());
    let mut grad_shape = vec![1; nd];
    grad_shape[nd - grad.shape().len()..].copy_from_slice(grad.shape());
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

/// Build an [`MathError::InvalidArgument`] for an internal autodiff failure —
/// used to surface broken graphs without panicking.
fn autograd_error(msg: &'static str) -> MathError {
    MathError::InvalidArgument(msg)
}

/// A handle to one node in a [`ComputationGraph`], with gradient tracking.
///
/// `GradTensor` values are cheaply clonable; each holds its graph plus a node
/// index. Ops on a `GradTensor` record result nodes in the shared graph.
#[derive(Clone)]
pub struct GradTensor {
    graph: ComputationGraph,
    node: usize,
}

impl GradTensor {
    /// The graph-internal node id of this handle.
    pub fn node_id(&self) -> usize {
        self.node
    }

    /// Clone of the forward-pass value recorded for this node, or `None` if
    /// the owning graph has been cleared.
    pub fn value(&self) -> Option<Tensor> {
        let d = self.graph.inner.borrow();
        d.entries.get(self.node).map(|e| e.tensor().clone())
    }

    /// Add two tracked tensors.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different graphs or shapes differ.
    pub fn add(&self, other: &GradTensor) -> GradTensor {
        assert_same_graph(self, other);
        let a = self.graph.node_value(self.node);
        let b = other.graph.node_value(other.node);
        assert_shape!(a, b);
        let out = a.add(&b).expect("shape checked by assert_shape!");
        let node = self.graph.push_op(out, vec![self.node, other.node], BackwardOp::Add);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// Subtract two tracked tensors.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different graphs or shapes differ.
    pub fn sub(&self, other: &GradTensor) -> GradTensor {
        assert_same_graph(self, other);
        let a = self.graph.node_value(self.node);
        let b = other.graph.node_value(other.node);
        assert_shape!(a, b);
        let out = a.sub(&b).expect("shape checked by assert_shape!");
        let node = self.graph.push_op(out, vec![self.node, other.node], BackwardOp::Sub);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// Multiply two tracked tensors.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different graphs or shapes differ.
    pub fn mul(&self, other: &GradTensor) -> GradTensor {
        assert_same_graph(self, other);
        let a = self.graph.node_value(self.node);
        let b = other.graph.node_value(other.node);
        assert_shape!(a, b);
        let out = a.mul(&b).expect("shape checked by assert_shape!");
        let node = self.graph.push_op(out, vec![self.node, other.node], BackwardOp::Mul);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// Divide two tracked tensors.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different graphs or shapes differ.
    pub fn div(&self, other: &GradTensor) -> GradTensor {
        assert_same_graph(self, other);
        let a = self.graph.node_value(self.node);
        let b = other.graph.node_value(other.node);
        assert_shape!(a, b);
        let out = a.div(&b).expect("shape checked by assert_shape!");
        let node = self.graph.push_op(out, vec![self.node, other.node], BackwardOp::Div);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// Negate the tracked tensor.
    pub fn neg(&self) -> GradTensor {
        let a = self.graph.node_value(self.node);
        let out = a.neg();
        let node = self.graph.push_op(out, vec![self.node], BackwardOp::Neg);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// Sigmoid activation on the tracked tensor.
    pub fn sigmoid(&self) -> GradTensor {
        let a = self.graph.node_value(self.node);
        let out = sigmoid(&a);
        let node = self.graph.push_op(out, vec![self.node], BackwardOp::Sigmoid);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// Tanh activation on the tracked tensor.
    pub fn tanh(&self) -> GradTensor {
        let a = self.graph.node_value(self.node);
        let out = tanh(&a);
        let node = self.graph.push_op(out, vec![self.node], BackwardOp::Tanh);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// Matrix multiply two tracked tensors.
    ///
    /// # Panics
    ///
    /// Panics if inner dimensions do not match (`a.shape[1] != b.shape[0]`)
    /// or the operands belong to different graphs.
    pub fn matmul(&self, other: &GradTensor) -> GradTensor {
        assert_same_graph(self, other);
        let a = self.graph.node_value(self.node);
        let b = other.graph.node_value(other.node);
        assert_matmul!(a, b);
        let out = a.matmul(&b).expect("dims checked by assert_matmul!");
        let node = self.graph.push_op(out, vec![self.node, other.node], BackwardOp::Matmul);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// ReLU activation on the tracked tensor.
    pub fn relu(&self) -> GradTensor {
        let a = self.graph.node_value(self.node);
        let out = relu(&a);
        let node = self.graph.push_op(out, vec![self.node], BackwardOp::ReLU);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// Sum all elements in the tracked tensor into a scalar node.
    pub fn sum(&self) -> GradTensor {
        let a = self.graph.node_value(self.node);
        let val = a.sum();
        let out = Tensor::scalar(val);
        let node = self.graph.push_op(out, vec![self.node], BackwardOp::Sum);
        GradTensor { graph: self.graph.clone(), node }
    }

    /// MSE loss against a tracked target.
    ///
    /// # Panics
    ///
    /// Panics if the operand shapes differ or belong to different graphs.
    pub fn mse_loss(&self, target: &GradTensor) -> GradTensor {
        assert_same_graph(self, target);
        let pred = self.graph.node_value(self.node);
        let tgt = target.graph.node_value(target.node);
        assert_shape!(pred, tgt);
        let val = mse(&pred, &tgt).expect("shape checked by assert_shape!");
        let out = Tensor::scalar(val);
        let node = self
            .graph
            .push_op(out, vec![self.node, target.node], BackwardOp::MseLoss);
        GradTensor { graph: self.graph.clone(), node }
    }
}

fn assert_same_graph(a: &GradTensor, b: &GradTensor) {
    assert!(
        Rc::ptr_eq(&a.graph.inner, &b.graph.inner),
        "operands belong to different ComputationGraph instances"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const E: f64 = 1e-9;

    #[test]
    fn add_backward() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2], vec![1.0, 2.0]);
        let b = g.variable_from_data(&[2], vec![3.0, 4.0]);
        let c = a.add(&b);
        let loss = c.sum();
        g.backward(&loss);
        assert_eq!(g.grad_of(&a).unwrap().as_slice(), &[1.0, 1.0]);
        assert_eq!(g.grad_of(&b).unwrap().as_slice(), &[1.0, 1.0]);
        assert_eq!(g.grad_of(&loss).unwrap().as_slice(), &[1.0]);
    }

    #[test]
    fn mul_add_backward() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2], vec![1.0, 2.0]);
        let b = g.variable_from_data(&[2], vec![3.0, 4.0]);
        let c = a.mul(&b);
        let loss = c.sum();
        g.backward(&loss);
        // d/dx (x·y) = y, d/dy (x·y) = x
        assert_eq!(g.grad_of(&a).unwrap().as_slice(), &[3.0, 4.0]);
        assert_eq!(g.grad_of(&b).unwrap().as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn neg_backward() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2], vec![1.0, -2.0]);
        let c = a.neg();
        let loss = c.sum();
        g.backward(&loss);
        assert_eq!(g.grad_of(&a).unwrap().as_slice(), &[-1.0, -1.0]);
    }

    #[test]
    fn sigmoid_backward() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2], vec![0.0, 2.0]);
        let c = a.sigmoid();
        let loss = c.sum();
        g.backward(&loss);
        let grad = g.grad_of(&a).unwrap();
        assert!((grad.get_flat(0) - 0.25).abs() < E);
        assert!((grad.get_flat(1) - (0.8807970779778823 * (1.0 - 0.8807970779778823))).abs() < 1e-8);
    }

    #[test]
    fn tanh_backward() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2], vec![0.0, 1.0]);
        let c = a.tanh();
        let loss = c.sum();
        g.backward(&loss);
        let grad = g.grad_of(&a).unwrap();
        let tanh_val: f64 = 0.7615941559557649;
        assert!((grad.get_flat(0) - 1.0).abs() < E);
        assert!((grad.get_flat(1) - (1.0 - tanh_val.powi(2))).abs() < 1e-8);
    }

    #[test]
    fn sub_backward() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2], vec![5.0, 7.0]);
        let b = g.variable_from_data(&[2], vec![2.0, 3.0]);
        let c = a.sub(&b);
        let loss = c.sum();
        g.backward(&loss);
        assert_eq!(g.grad_of(&a).unwrap().as_slice(), &[1.0, 1.0]);
        assert_eq!(g.grad_of(&b).unwrap().as_slice(), &[-1.0, -1.0]);
    }

    #[test]
    fn div_backward() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2], vec![6.0, 8.0]);
        let b = g.variable_from_data(&[2], vec![2.0, 4.0]);
        let loss = a.div(&b).sum();
        g.backward(&loss);
        // d/da (a/b) = 1/b → [0.5, 0.25]; d/db = −a/b² → [−1.5, −0.5]
        assert_eq!(g.grad_of(&a).unwrap().as_slice(), &[0.5, 0.25]);
        assert_eq!(g.grad_of(&b).unwrap().as_slice(), &[-1.5, -0.5]);
    }

    #[test]
    fn relu_backward() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[3], vec![-1.0, 0.0, 2.0]);
        let loss = a.relu().sum();
        g.backward(&loss);
        assert_eq!(g.grad_of(&a).unwrap().as_slice(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn mse_loss_backward() {
        let mut g = ComputationGraph::new();
        let pred = g.variable_from_data(&[2], vec![1.0, 3.0]);
        let target = g.variable_from_data(&[2], vec![2.0, 1.0]);
        let loss = pred.mse_loss(&target);
        g.backward(&loss);
        // d/dx MSE = 2*(x-t)/n → 2*(1-2)/2=-1, 2*(3-1)/2=2
        assert_eq!(g.grad_of(&pred).unwrap().as_slice(), &[-1.0, 2.0]);
        assert_eq!(g.grad_of(&target).unwrap().as_slice(), &[0.0, 0.0]);
    }

    #[test]
    fn scaled_backward_matches_expected() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = g.variable_from_data(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let loss = a.mul(&b).sum();
        assert!(g.try_backward_scaled(&loss, 2.0).is_ok());
        // d/dx (x·y) summed = y, scaled by 2
        assert_eq!(g.grad_of(&a).unwrap().as_slice(), &[10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn zero_grad_clears_gradients() {
        let mut g = ComputationGraph::new();
        let a = g.variable_from_data(&[2], vec![1.0, 2.0]);
        let loss = a.sum();
        g.backward(&loss);
        assert!(g.grad_of(&a).is_some());
        g.zero_grad();
        assert!(g.grad_of(&a).is_none());
        assert_eq!(g.len(), 2);
        g.clear();
        assert!(g.is_empty());
        assert!(a.value().is_none());
    }

    #[test]
    fn try_variable_rejects_bad_shape() {
        let mut g = ComputationGraph::new();
        assert!(g.try_variable_from_data(&[2, 2], vec![1.0, 2.0, 3.0]).is_err());
        assert!(g.try_variable_from_data(&[2], vec![1.0, 2.0]).is_ok());
    }

    #[test]
    fn matmul_backward_via_try() {
        let mut g = ComputationGraph::new();
        // (1×2) @ (2×1) → scalar; d/da = bᵀ, d/db = aᵀ
        let a = g.variable_from_data(&[1, 2], vec![3.0, 4.0]);
        let b = g.variable_from_data(&[2, 1], vec![5.0, 6.0]);
        let loss = a.matmul(&b);
        assert!(g.try_backward(&loss).is_ok());
        assert_eq!(g.grad_of(&a).unwrap().as_slice(), &[5.0, 6.0]);
        assert_eq!(g.grad_of(&b).unwrap().as_slice(), &[3.0, 4.0]);
    }

    #[test]
    fn independent_graphs_coexist() {
        let mut g1 = ComputationGraph::new();
        let x1 = g1.variable_from_data(&[2], vec![1.0, 2.0]);
        let loss1 = x1.mul(&x1).sum();

        let mut g2 = ComputationGraph::new();
        let x2 = g2.variable_from_data(&[2], vec![3.0, 4.0]);
        let y2 = g2.variable_from_data(&[2], vec![1.0, 1.0]);
        let loss2 = x2.mul(&y2).sum();

        // Differentiate both graphs; neither interferes with the other.
        g2.backward(&loss2);
        g1.backward(&loss1);
        // d/dx (x²) summed = 2x
        assert_eq!(g1.grad_of(&x1).unwrap().as_slice(), &[2.0, 4.0]);
        // d/dx (x·y) summed = y
        assert_eq!(g2.grad_of(&x2).unwrap().as_slice(), &[1.0, 1.0]);

        // Cross-graph lookups find nothing.
        assert!(g1.grad_of(&x2).is_none());
        assert!(g2.grad_of(&x1).is_none());

        // Mixing handles across graphs is rejected loudly.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = x1.add(&x2);
        }));
        assert!(result.is_err());
    }
}
