# MathVerse Features

Complete feature inventory by crate. Each feature maps to a public API item in
the corresponding crate.

## mathverse-core

- Numeric abstraction traits (ring, field, real, complex).
- Error types and error taxonomy.
- Constants (high precision).
- Generic operations shared across domains.
- Precision utilities (epsilon, tolerance, rounding).
- Common algorithms.
- `no_std` support via `libm` feature
  (`default-features = false, features = ["libm"]`).
- Const-generic fixed-size numeric arrays (`Array<T, N>`) for embedded/no-std
  use.

## mathverse-arithmetic

- Addition, subtraction, multiplication, division.
- Modulus, powers, roots.
- Absolute value, percentage.
- Logarithms, exponentials.
- Rounding, precision control.

## mathverse-algebra

- Polynomial representation and operations.
- Equation solving (linear, quadratic, cubic, quartic).
- Factorization.
- Simplification.
- Rational expressions.
- Solvability-by-radicals classification (Galois-flavored).
- LaTeX rendering of polynomials and solutions.

## mathverse-trigonometry

- sin, cos, tan, cot, sec, cosec.
- Hyperbolic functions.
- Inverse functions.
- Angle conversions (degrees/radians/gradians).
- Trig identities, triangle laws, special functions.
- Batched slice evaluation (DSP/audio) and exact special-angle values.
- `no_std` / `libm` support.

## mathverse-geometry

### 2D

Shapes: circle, triangle, rectangle, polygon, ellipse.

Operations: area, perimeter, centroid, rotation, scaling, translation, distance,
intersection.

### 3D

Shapes: sphere, cube, cylinder, cone, plane, line.

Operations: volume, surface area, projection, collision, distance.

## mathverse-linear-algebra

- Matrix, vector, tensor.
- Sparse matrix.
- Eigenvalues, eigenvectors.
- LU, QR, SVD, Cholesky decompositions.
- Matrix inverse, determinant.

## mathverse-calculus

- Derivatives (including partial, chain rule).
- Integrals (definite, indefinite, numerical).
- Gradient, curl, divergence, Laplacian.

## mathverse-complex

- Complex number arithmetic and functions.

## mathverse-probability

- Random variables.
- Bayes theorem, conditional probability.
- Markov chains.
- Monte Carlo methods.
- Distributions: normal, Poisson, uniform, Bernoulli, binomial, gamma, beta,
  chi-square.

## mathverse-statistics

- Descriptive: mean, median, mode, variance, standard deviation, quartiles.
- Inferential: t-test, z-test, ANOVA, confidence intervals, regression.

## mathverse-number-theory

- Prime generation and testing.
- GCD, LCM.
- Modular arithmetic.
- Euler theorem.
- RSA helpers.

## mathverse-combinatorics

- Combinatorial functions (permutations, combinations, etc.).

## mathverse-graph

- Graph algorithms (BFS/DFS, shortest paths, connectivity, etc.).

## mathverse-optimization

- Gradient descent.
- SGD.
- Adam, RMSProp.
- Simulated annealing.
- Genetic algorithms.

## mathverse-numerical

- Root finding: Newton-Raphson, bisection.
- Runge-Kutta ODE solvers.
- Interpolation.
- Approximation.

## mathverse-equations

- Equation solving utilities (bridges algebra and numerical).

## mathverse-transforms

- FFT, DCT.
- Wavelets.

## mathverse-signal

- FIR, IIR filters.
- Convolution, correlation.
- Filtering utilities.

## mathverse-image

- Kernels.
- Gaussian blur.
- Sobel, Canny edge detection.
- Histogram, equalization.
- Morphology.
- Resizing, rotation.
- Affine transform, perspective transform.

## mathverse-vision

- Camera matrix.
- Homography.
- Epipolar geometry.
- Feature extraction.
- Image matching.
- Optical flow.

## mathverse-graphics

- 2D transformations.
- Bezier curves.
- Quaternions.
- Projection, view, model matrices.
- Lighting math.
- Frustum.

## mathverse-ai

- Activations: ReLU, GELU, SiLU, softmax.
- Losses: cross entropy, MSE, MAE, Huber.
- Metrics: accuracy, precision, recall, F1, ROC AUC.
- Tensor ops: broadcasting, matrix multiplication, normalization.
- Optimizers: SGD, Adam, AdamW.
- Attention math: QKV, scaled dot product, rotary embeddings, positional
  encoding.

## mathverse-physics

- Mechanics.
- Electricity, magnetism.
- Optics.
- Thermodynamics.
- Fluid mechanics.
- Quantum basics.
- Relativity helpers.

## mathverse-finance

- Compound interest.
- Loan calculations, EMI.
- NPV, IRR.
- Black-Scholes helpers.
- Risk metrics.

## mathverse-symbolic

- Simplification.
- Expression trees.
- Symbolic derivatives.
- Equation solving.
- LaTeX generation.

## mathverse-units

- SI units, imperial.
- Currency abstraction.
- Temperature, length, mass, time, energy.
- Compile-time dimensional analysis.

## mathverse-plot

- Plot, scatter, histogram, heatmap.
- SVG, HTML.
- Terminal charts.

## mathverse-prelude

- Re-exports of every crate for `use mathverse::prelude::*;`.
