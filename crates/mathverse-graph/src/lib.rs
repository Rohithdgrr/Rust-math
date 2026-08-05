//! # mathverse-graph
//!
//! Graph data structures and algorithms for the `MathVerse` ecosystem.
//!
//! Provides three graph representations:
//! - [`Graph`]: undirected, unweighted adjacency-list graph
//! - [`WeightedGraph`]: weighted graph supporting both directed and undirected edges
//! - [`DirectedGraph`]: directed, unweighted adjacency-list graph
//!
//! Supports construction, traversal, and basic query operations.
//! Edge weights default to `0.0` for unweighted queries.

mod graph;
pub use graph::{Graph, WeightedGraph, DirectedGraph};
