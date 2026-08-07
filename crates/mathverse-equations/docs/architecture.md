# Architecture

## Purpose
Equation solvers for the MathVerse ecosystem.

## Components
- `polynomial` — root finding
- `linear_system` — direct solvers
- `nonlinear` — iterative methods
- `differential` — ODE integration
- `optimization` — 1D minimization
- `matrix_eq` — linear algebra
- `dynamical` — discrete systems

## Data Flow
User -> lib.rs -> module -> mathverse-core/algebra/matrix/vector/numerical
