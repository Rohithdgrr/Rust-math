//! Simple graph layout example with BFS tree visualization.

use mathverse_graph::Graph;
use mathverse_plot::{GraphLayoutConfig, render_graph, render_bfs_tree};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple graph: 0-1-2-3-4 (linear chain)
    let mut graph = Graph::new(5);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 3);
    graph.add_edge(3, 4);
    // Add a cross edge
    graph.add_edge(0, 4);

    // Render full graph
    let config = GraphLayoutConfig::new()
        .with_node_radius(10.0);

    let svg = render_graph(&graph, config)?;
    std::fs::write("graph.svg", &svg)?;
    println!("Wrote graph.svg ({} bytes)", svg.len());

    // Render BFS tree from node 0
    let tree_config = GraphLayoutConfig::new()
        .with_node_radius(10.0);

    let svg = render_bfs_tree(&graph, 0, tree_config)?;
    std::fs::write("bfs_tree.svg", &svg)?;
    println!("Wrote bfs_tree.svg ({} bytes)", svg.len());

    Ok(())
}
