//! Automatic differentiation via reverse-mode autodiff (backpropagation).
//!
//! Wraps Tensor operations in a computation graph. Call `.backward()` on the
//! loss to compute gradients, then read them from `.grad` on each tensor.

use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};
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

fn reduce_broadcast_grad(grad: &Tensor, target_shape: &[usize]) -> Result<Tensor, MathError> {
    let mut grad = grad.clone();
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

/// Append an entry to the computation graph, grow the gradient registry to
/// match, and return the new node id.
fn push_node(tensor: Tensor, inputs: Vec<usize>, op: BackwardOp) -> usize {
    let id = GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        let id = g.len();
        g.push(GraphEntry::Op { tensor, inputs, op });
        id
    });
    GRAD_REGISTRY.with(|g| {
        let mut g = g.borrow_mut();
        if g.len() <= id {
            g.resize_with(id + 1, || None);
        }
    });
    id
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
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not match the product of `shape`. For a
    /// fallible constructor use [`GradTensor::try_from_data`].
    pub fn from_data(shape: &[usize], data: Vec<f64>) -> Self {
        Self::try_from_data(shape, data)
            .expect("GradTensor::from_data: data length does not match shape")
    }

    /// Fallible variant of [`GradTensor::from_data`].
    pub fn try_from_data(shape: &[usize], data: Vec<f64>) -> MathResult<Self> {
        Ok(Self::new(Tensor::from_vec(shape, data)?))
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
    ///
    /// # Panics
    ///
    /// Panics if gradient computation fails; see [`GradTensor::try_backward`].
    pub fn backward(&mut self, scale: f64) {
        backward(self, scale)
    }

    /// Fallible backward pass from this tensor.
    pub fn try_backward(&mut self, scale: f64) -> MathResult<()> {
        try_backward(self, scale)
    }

    /// Return accumulated gradient after backward.
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }
}

/// Add two GradTensors (tracked).
///
/// # Panics
///
/// Panics if the operand shapes differ (a programmer error, not runtime data
/// dependence). The same applies to [`mul`], [`sub`], [`div`] and [`matmul`].
pub fn add(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_shape!(a.tensor, b.tensor);
    // Shape pre-checked above, so elementwise add cannot fail here.
    let out = a.tensor.add(&b.tensor).expect("shape checked by assert_shape!");
    let id = push_node(out.clone(), vec![a.node_id, b.node_id], BackwardOp::Add);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Multiply two GradTensors (tracked).
///
/// # Panics
///
/// Panics if the operand shapes differ.
pub fn mul(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_shape!(a.tensor, b.tensor);
    let out = a.tensor.mul(&b.tensor).expect("shape checked by assert_shape!");
    let id = push_node(out.clone(), vec![a.node_id, b.node_id], BackwardOp::Mul);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Element-wise subtraction (tracked).
///
/// # Panics
///
/// Panics if the operand shapes differ.
pub fn sub(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_shape!(a.tensor, b.tensor);
    let out = a.tensor.sub(&b.tensor).expect("shape checked by assert_shape!");
    let id = push_node(out.clone(), vec![a.node_id, b.node_id], BackwardOp::Sub);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Element-wise division (tracked).
///
/// # Panics
///
/// Panics if the operand shapes differ.
pub fn div(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_shape!(a.tensor, b.tensor);
    let out = a.tensor.div(&b.tensor).expect("shape checked by assert_shape!");
    let id = push_node(out.clone(), vec![a.node_id, b.node_id], BackwardOp::Div);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Negate a tensor (tracked).
pub fn neg(a: &GradTensor) -> GradTensor {
    let out = a.tensor.neg();
    let id = push_node(out.clone(), vec![a.node_id], BackwardOp::Neg);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Sigmoid activation (tracked).
pub fn sigmoid(a: &GradTensor) -> GradTensor {
    let out = crate::activations::sigmoid(&a.tensor);
    let id = push_node(out.clone(), vec![a.node_id], BackwardOp::Sigmoid);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Tanh activation (tracked).
pub fn tanh(a: &GradTensor) -> GradTensor {
    let out = crate::activations::tanh(&a.tensor);
    let id = push_node(out.clone(), vec![a.node_id], BackwardOp::Tanh);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Matrix multiply (tracked).
///
/// # Panics
///
/// Panics if inner dimensions do not match (`a.shape[1] != b.shape[0]`).
pub fn matmul(a: &GradTensor, b: &GradTensor) -> GradTensor {
    assert_matmul!(a.tensor, b.tensor);
    let out = a.tensor.matmul(&b.tensor).expect("dims checked by assert_matmul!");
    let id = push_node(out.clone(), vec![a.node_id, b.node_id], BackwardOp::Matmul);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// ReLU (tracked).
pub fn relu_op(a: &GradTensor) -> GradTensor {
    let out = crate::activations::relu(&a.tensor);
    let id = push_node(out.clone(), vec![a.node_id], BackwardOp::ReLU);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Sum all elements (tracked).
pub fn sum(a: &GradTensor) -> GradTensor {
    let val = a.tensor.sum();
    let out = Tensor::scalar(val);
    let id = push_node(out.clone(), vec![a.node_id], BackwardOp::Sum);
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// MSE loss (tracked).
///
/// # Panics
///
/// Panics if the operand shapes differ.
pub fn mse_loss(pred: &GradTensor, target: &GradTensor) -> GradTensor {
    assert_shape!(pred.tensor, target.tensor);
    let val =
        crate::losses::mse(&pred.tensor, &target.tensor).expect("shape checked by assert_shape!");
    let out = Tensor::scalar(val);
    let id = push_node(
        out.clone(),
        vec![pred.node_id, target.node_id],
        BackwardOp::MseLoss,
    );
    GradTensor { tensor: out, grad: None, node_id: id }
}

/// Backward pass from a scalar loss tensor.
///
/// # Panics
///
/// Panics with the underlying [`MathError`] if gradient reduction fails
/// (e.g. inconsistent shapes recorded in the graph). Prefer
/// [`try_backward`] in code that must not panic.
pub fn backward(loss: &mut GradTensor, scale: f64) {
    if let Err(e) = try_backward(loss, scale) {
        panic!("backward failed: {e}");
    }
}

/// Fallible variant of [`backward`]: propagates shape/broadcast errors as
/// [`MathResult`] instead of panicking.
///
/// Computes gradients for all nodes in the graph and stores them in the
/// gradient registry; returns `Ok(())` on success.
pub fn try_backward(loss: &mut GradTensor, scale: f64) -> MathResult<()> {
    let mut grads: Vec<Option<Tensor>> = GRAPH.with(|g| vec![None; g.borrow().len()]);

    // Seed: d(loss)/d(loss) = scale, shaped like the loss itself so that
    // downstream shape checks (e.g. matmul backward) see consistent ranks.
    grads[loss.node_id] = Some(Tensor::full(&loss.tensor.shape, scale));

    GRAPH.with(|g| -> MathResult<()> {
        let g = g.borrow();

        // Reverse pass over the graph
        for i in (0..g.len()).rev() {
            let grad_out = match &grads[i] {
                Some(gv) => gv.clone(),
                None => continue,
            };

            match &g[i] {
                GraphEntry::Tensor(_) => {}
                GraphEntry::Op { inputs, op, .. } => {
                    let input_ids = inputs.clone();
                    let input_grads = match op {
                        BackwardOp::Add => vec![
                            reduce_broadcast_grad(&grad_out, &tensor_from_entry(&g[input_ids[0]]).shape)?,
                            reduce_broadcast_grad(&grad_out, &tensor_from_entry(&g[input_ids[1]]).shape)?,
                        ],
                        BackwardOp::Sub => vec![
                            reduce_broadcast_grad(&grad_out, &tensor_from_entry(&g[input_ids[0]]).shape)?,
                            reduce_broadcast_grad(&grad_out.neg(), &tensor_from_entry(&g[input_ids[1]]).shape)?,
                        ],
                        BackwardOp::Mul => {
                            let a = tensor_from_entry(&g[input_ids[0]]);
                            let b = tensor_from_entry(&g[input_ids[1]]);
                            let b_expanded =
                                b.broadcast_to(&grad_out.shape)
                                    .map_err(|_| autograd_error("autograd mul: cannot broadcast operand b to grad shape"))?;
                            let a_expanded =
                                a.broadcast_to(&grad_out.shape)
                                    .map_err(|_| autograd_error("autograd mul: cannot broadcast operand a to grad shape"))?;
                            let grad_a = grad_out
                                .mul(&b_expanded)
                                .map_err(|_| autograd_error("autograd mul: grad_a computation failed"))?;
                            let grad_b = grad_out
                                .mul(&a_expanded)
                                .map_err(|_| autograd_error("autograd mul: grad_b computation failed"))?;
                            vec![
                                reduce_broadcast_grad(&grad_a, &a.shape)?,
                                reduce_broadcast_grad(&grad_b, &b.shape)?,
                            ]
                        }
                        BackwardOp::Div => {
                            let a = tensor_from_entry(&g[input_ids[0]]);
                            let b = tensor_from_entry(&g[input_ids[1]]);
                            let b_expanded =
                                b.broadcast_to(&grad_out.shape)
                                    .map_err(|_| autograd_error("autograd div: cannot broadcast operand b to grad shape"))?;
                            let a_expanded =
                                a.broadcast_to(&grad_out.shape)
                                    .map_err(|_| autograd_error("autograd div: cannot broadcast operand a to grad shape"))?;
                            let inv_b = Tensor::ones(&b_expanded.shape)
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
                                reduce_broadcast_grad(&grad_a, &a.shape)?,
                                reduce_broadcast_grad(&grad_b, &b.shape)?,
                            ]
                        }
                        BackwardOp::Neg => vec![grad_out.neg()],
                        BackwardOp::Matmul => {
                            let a = tensor_from_entry(&g[input_ids[0]]);
                            let b = tensor_from_entry(&g[input_ids[1]]);
                            let bt = b.transpose().map_err(|_| autograd_error("autograd matmul: transpose failed"))?;
                            let at = a.transpose().map_err(|_| autograd_error("autograd matmul: transpose failed"))?;
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
                            let input = tensor_from_entry(&g[input_ids[0]]);
                            let mask = crate::activations::relu_grad(&input);
                            assert_shape!(grad_out, mask);
                            let gated = grad_out
                                .mul(&mask)
                                .map_err(|_| autograd_error("autograd relu: mask multiply failed"))?;
                            vec![gated]
                        }
                        BackwardOp::Sigmoid => {
                            let input = tensor_from_entry(&g[input_ids[0]]);
                            let sig = crate::activations::sigmoid(&input);
                            let ones = Tensor::ones(&sig.shape);
                            let grad_activation = sig
                                .mul(&ones.sub(&sig).map_err(|_| autograd_error("autograd sigmoid: 1−σ failed"))?)
                                .map_err(|_| autograd_error("autograd sigmoid: σ(1−σ) failed"))?;
                            let gated = grad_out
                                .mul(&grad_activation)
                                .map_err(|_| autograd_error("autograd sigmoid: chain multiply failed"))?;
                            vec![gated]
                        }
                        BackwardOp::Tanh => {
                            let input = tensor_from_entry(&g[input_ids[0]]);
                            let tanh = crate::activations::tanh(&input);
                            let ones = Tensor::ones(&tanh.shape);
                            let sq = tanh.mul(&tanh).map_err(|_| autograd_error("autograd tanh: tanh² failed"))?;
                            let grad_activation = ones
                                .sub(&sq)
                                .map_err(|_| autograd_error("autograd tanh: 1−tanh² failed"))?;
                            let gated = grad_out
                                .mul(&grad_activation)
                                .map_err(|_| autograd_error("autograd tanh: chain multiply failed"))?;
                            vec![gated]
                        }
                        BackwardOp::Sum => {
                            let input = tensor_from_entry(&g[input_ids[0]]);
                            vec![Tensor::full(&input.shape, grad_out.data[0])]
                        }
                        BackwardOp::MseLoss => {
                            let pred = tensor_from_entry(&g[input_ids[0]]);
                            let target = tensor_from_entry(&g[input_ids[1]]);
                            assert_shape!(pred, target);
                            let gd = crate::losses::mse_grad(&pred, &target)
                                .map_err(|_| autograd_error("autograd mse_loss: mse_grad failed"))?;
                            vec![gd, Tensor::zeros(&target.shape)]
                        }
                    };
                    for (j, &input_id) in input_ids.iter().enumerate() {
                        if j < input_grads.len() {
                            grads[input_id] = Some(match &grads[input_id] {
                                Some(existing) => {
                                    assert_shape!(existing, &input_grads[j]);
                                    existing.add(&input_grads[j])
                                        .map_err(|_| autograd_error("autograd accumulate: gradient accumulation failed"))?
                                }
                                None => input_grads[j].clone(),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    })?;

    // Write computed gradients back to the registry
    GRAD_REGISTRY.with(|reg| {
        let mut reg = reg.borrow_mut();
        for (id, grad) in grads.into_iter().enumerate() {
            if let Some(gv) = grad {
                if id < reg.len() {
                    reg[id] = Some(gv);
                }
            }
        }
    });

    // Update the loss tensor's grad
    loss.grad = Some(Tensor::full(&loss.tensor.shape, scale));
    Ok(())
}

/// Build an [`MathError::InvalidArgument`] for an internal autodiff failure —
/// used to surface broken graphs without panicking.
fn autograd_error(msg: &'static str) -> MathError {
    MathError::InvalidArgument(msg)
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

    #[test]
    fn try_backward_matches_backward() {
        clear_graph();
        let a = GradTensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = GradTensor::from_data(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let mut loss = sum(&mul(&a, &b));
        assert!(loss.try_backward(2.0).is_ok());
        // d/dx (x·y) summed = y, scaled by 2
        assert_eq!(get_grad(a.node_id).unwrap().data, vec![10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn try_from_data_rejects_bad_shape() {
        assert!(GradTensor::try_from_data(&[2, 2], vec![1.0, 2.0, 3.0]).is_err());
        assert!(GradTensor::try_from_data(&[2], vec![1.0, 2.0]).is_ok());
    }

    #[test]
    fn matmul_backward_via_try() {
        clear_graph();
        // (1×2) @ (2×1) → scalar; d/da = bᵀ, d/db = aᵀ
        let a = GradTensor::from_data(&[1, 2], vec![3.0, 4.0]);
        let b = GradTensor::from_data(&[2, 1], vec![5.0, 6.0]);
        let mut loss = matmul(&a, &b);
        assert!(loss.try_backward(1.0).is_ok());
        assert_eq!(get_grad(a.node_id).unwrap().data, vec![5.0, 6.0]);
        assert_eq!(get_grad(b.node_id).unwrap().data, vec![3.0, 4.0]);
    }
}
