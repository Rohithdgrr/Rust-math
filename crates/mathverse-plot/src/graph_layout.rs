//! Graph layout visualization via `mathverse-graph`.
//!
//! Renders graphs as SVG with nodes as circles and edges as lines.
//! Supports BFS and DFS tree visualization.

use mathverse_graph::{Graph, WeightedGraph};
use std::collections::VecDeque;

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::svg::SvgPlot;

/// Configuration for a graph layout plot.
#[derive(Debug, Clone)]
pub struct GraphLayoutConfig {
    /// Plot configuration (title, labels, dimensions).
    pub plot_config: PlotConfig,
    /// Node radius in SVG units.
    pub node_radius: f64,
    /// Edge color.
    pub edge_color: crate::style::Color,
    /// Node color.
    pub node_color: crate::style::Color,
    /// Highlight color for BFS/DFS tree edges.
    pub tree_color: crate::style::Color,
}

impl GraphLayoutConfig {
    /// Create a new graph layout config with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plot_config: PlotConfig::new()
                .with_title("Graph Layout".to_string()),
            node_radius: 8.0,
            edge_color: crate::style::Color::rgb(0x88, 0x88, 0x88),
            node_color: crate::style::Color::BLUE,
            tree_color: crate::style::Color::RED,
        }
    }

    /// Set the node radius.
    #[must_use]
    pub fn with_node_radius(mut self, r: f64) -> Self {
        self.node_radius = r.max(1.0);
        self
    }

    /// Set the edge color.
    #[must_use]
    pub fn with_edge_color(mut self, color: crate::style::Color) -> Self {
        self.edge_color = color;
        self
    }

    /// Set the node color.
    #[must_use]
    pub fn with_node_color(mut self, color: crate::style::Color) -> Self {
        self.node_color = color;
        self
    }

    /// Set the tree highlight color.
    #[must_use]
    pub fn with_tree_color(mut self, color: crate::style::Color) -> Self {
        self.tree_color = color;
        self
    }
}

/// Compute a circular layout for `n` nodes.
fn circular_layout(n: usize) -> Vec<(f64, f64)> {
    if n == 0 {
        return Vec::new();
    }
    let radius = 1.0;
    (0..n)
        .map(|i| {
            let angle = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect()
}

/// Render an unweighted graph as SVG.
pub fn render_graph(
    graph: &Graph,
    config: GraphLayoutConfig,
) -> PlotResult<String> {
    let positions = circular_layout(graph.len());
    render_graph_with_positions(graph, &positions, config)
}

/// Render a weighted graph as SVG.
pub fn render_weighted_graph(
    graph: &WeightedGraph,
    config: GraphLayoutConfig,
) -> PlotResult<String> {
    let positions = circular_layout(graph.len());
    render_weighted_graph_with_positions(graph, &positions, config)
}

/// Render a graph with pre-computed node positions.
pub fn render_graph_with_positions(
    graph: &Graph,
    positions: &[(f64, f64)],
    config: GraphLayoutConfig,
) -> PlotResult<String> {
    if graph.len() != positions.len() {
        return Err(PlotError::InvalidData(
            "positions length must match graph size".into(),
        ));
    }

    let mut plot = SvgPlot::new(config.plot_config);

    // Draw edges
    for u in 0..graph.len() {
        for &v in graph.neighbors(u) {
            if u < v {
                let (x1, y1) = positions[u];
                let (x2, y2) = positions[v];
                plot.add_series(crate::DataSeries::new(
                    format!("edge_{}_{}", u, v),
                    vec![
                        crate::DataPoint::new(x1, y1),
                        crate::DataPoint::new(x2, y2),
                    ],
                ));
            }
        }
    }

    // Draw nodes
    for (i, &(x, y)) in positions.iter().enumerate() {
        let label = format!("node_{}", i);
        let series = crate::DataSeries::new(label, vec![crate::DataPoint::new(x, y)]);
        plot.add_series(series);
    }

    Ok(plot.generate())
}

/// Render a weighted graph with pre-computed node positions.
pub fn render_weighted_graph_with_positions(
    graph: &WeightedGraph,
    positions: &[(f64, f64)],
    config: GraphLayoutConfig,
) -> PlotResult<String> {
    if graph.len() != positions.len() {
        return Err(PlotError::InvalidData(
            "positions length must match graph size".into(),
        ));
    }

    let mut plot = SvgPlot::new(config.plot_config);

    // Draw edges with weights
    for u in 0..graph.len() {
        for &(v, w) in graph.neighbors(u) {
            if u < v {
                let (x1, y1) = positions[u];
                let (x2, y2) = positions[v];
                let label = format!("edge_{}_{}_w={:.2}", u, v, w);
                let series = crate::DataSeries::new(
                    label,
                    vec![
                        crate::DataPoint::new(x1, y1),
                        crate::DataPoint::new(x2, y2),
                    ],
                );
                plot.add_series(series);
            }
        }
    }

    // Draw nodes
    for (i, &(x, y)) in positions.iter().enumerate() {
        let label = format!("node_{}", i);
        let series = crate::DataSeries::new(label, vec![crate::DataPoint::new(x, y)]);
        plot.add_series(series);
    }

    Ok(plot.generate())
}

/// Compute BFS tree edges from `start`. Returns `(parent, child)` pairs.
fn bfs_tree_edges(graph: &Graph, start: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    let mut visited = vec![false; graph.len()];
    let mut parent = vec![None; graph.len()];
    let mut q = VecDeque::from([start]);
    visited[start] = true;
    while let Some(u) = q.pop_front() {
        for &v in graph.neighbors(u) {
            if !visited[v] {
                visited[v] = true;
                parent[v] = Some(u);
                edges.push((u, v));
                q.push_back(v);
            }
        }
    }
    edges
}

/// Compute DFS tree edges from `start`. Returns `(parent, child)` pairs.
fn dfs_tree_edges(graph: &Graph, start: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    let mut visited = vec![false; graph.len()];
    let mut stack = vec![(start, None::<usize>)];
    while let Some((u, par)) = stack.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        if let Some(p) = par {
            edges.push((p, u));
        }
        for &v in graph.neighbors(u).iter().rev() {
            if !visited[v] {
                stack.push((v, Some(u)));
            }
        }
    }
    edges
}

/// Render a BFS tree from a start node.
pub fn render_bfs_tree(
    graph: &Graph,
    start: usize,
    config: GraphLayoutConfig,
) -> PlotResult<String> {
    let positions = circular_layout(graph.len());
    let tree_edges = bfs_tree_edges(graph, start);
    render_tree_with_positions(graph, &positions, &tree_edges, config)
}

/// Render a DFS tree from a start node.
pub fn render_dfs_tree(
    graph: &Graph,
    start: usize,
    config: GraphLayoutConfig,
) -> PlotResult<String> {
    let positions = circular_layout(graph.len());
    let tree_edges = dfs_tree_edges(graph, start);
    render_tree_with_positions(graph, &positions, &tree_edges, config)
}

fn render_tree_with_positions(
    graph: &Graph,
    positions: &[(f64, f64)],
    tree_edges: &[(usize, usize)],
    config: GraphLayoutConfig,
) -> PlotResult<String> {
    let mut plot = SvgPlot::new(config.plot_config);

    // Draw all edges in gray
    for u in 0..graph.len() {
        for &v in graph.neighbors(u) {
            if u < v {
                let (x1, y1) = positions[u];
                let (x2, y2) = positions[v];
                let series = crate::DataSeries::new(
                    format!("edge_{}_{}", u, v),
                    vec![
                        crate::DataPoint::new(x1, y1),
                        crate::DataPoint::new(x2, y2),
                    ],
                );
                plot.add_series(series);
            }
        }
    }

    // Draw BFS/DFS tree edges in highlight color
    for &(u, v) in tree_edges {
        let (x1, y1) = positions[u];
        let (x2, y2) = positions[v];
        let series = crate::DataSeries::new(
            format!("tree_{}_{}", u, v),
            vec![
                crate::DataPoint::new(x1, y1),
                crate::DataPoint::new(x2, y2),
            ],
        );
        plot.add_series(series);
    }

    // Draw nodes
    for (i, &(x, y)) in positions.iter().enumerate() {
        let label = format!("node_{}", i);
        let series = crate::DataSeries::new(label, vec![crate::DataPoint::new(x, y)]);
        plot.add_series(series);
    }

    Ok(plot.generate())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_renders_svg() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);

        let config = GraphLayoutConfig::new();
        let svg = render_graph(&graph, config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn weighted_graph_renders_svg() {
        let mut graph = WeightedGraph::new(4);
        graph.add_undirected_edge(0, 1, 1.5);
        graph.add_undirected_edge(1, 2, 2.0);
        graph.add_undirected_edge(2, 3, 0.5);

        let config = GraphLayoutConfig::new();
        let svg = render_weighted_graph(&graph, config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn bfs_tree_renders_svg() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);

        let config = GraphLayoutConfig::new();
        let svg = render_bfs_tree(&graph, 0, config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn dfs_tree_renders_svg() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);

        let config = GraphLayoutConfig::new();
        let svg = render_dfs_tree(&graph, 0, config).unwrap();
        assert!(svg.contains("<svg"));
    }
}