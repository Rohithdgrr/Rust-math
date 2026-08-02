//! # mathverse-graph
//!
//! Graph data structures and algorithms for the MathVerse ecosystem.
//!
//! Provides three graph representations:
//! - [`Graph`]: undirected, unweighted adjacency-list graph
//! - [`WeightedGraph`]: undirected graph with edge weights
//! - [`DirectedGraph`]: directed graph with edge weights
//!
//! Supports construction, traversal, and basic query operations.
//! Edge weights default to `0.0` for unweighted queries.

mod graph;
pub use graph::{Graph, WeightedGraph, DirectedGraph};
