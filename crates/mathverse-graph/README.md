# MathVerse Graph

[![Crates.io](https://img.shields.io/crates/v/mathverse-graph.svg)](https://crates.io/crates/mathverse-graph)
[![docs.rs](https://docs.rs/mathverse-graph/badge.svg)](https://docs.rs/mathverse-graph)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Graph data structures and algorithms: BFS/DFS traversal, shortest paths, minimum spanning trees, connectivity analysis, and topological sorting.

---

## Features

- **Graph types** — undirected, weighted, and directed graphs with adjacency list representation
- **Traversal** — BFS and DFS with order tracking
- **Shortest paths** — Dijkstra, Bellman-Ford (negative cycle detection), Floyd-Warshall (all-pairs)
- **Minimum spanning trees** — Prim's and Kruskal's algorithms
- **Connectivity** — connected components, cycle detection, bipartiteness testing
- **Directed graphs** — topological sort (Kahn's algorithm), strongly connected components (Tarjan's)

## Module Overview

| Struct | Purpose |
|--------|---------|
| `Graph` | Undirected unweighted — BFS, DFS, shortest path, connectivity, cycle detection |
| `WeightedGraph` | Directed weighted — Dijkstra, Bellman-Ford, Floyd-Warshall, Prim, Kruskal |
| `DirectedGraph` | Directed unweighted — topological sort, strongly connected components |

## Installation

```toml
[dependencies]
mathverse-graph = "0.1"
```

## Quick Start

```rust
use mathverse_graph::*;

fn main() {
    // Create an undirected graph with 5 nodes
    let mut g = Graph::new(5);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 4);

    // BFS traversal
    let order = g.bfs(0);
    println!("BFS order: {:?}", order); // [0, 1, 2, 3, 4]

    // Check connectivity
    println!("Connected: {}", g.is_connected()); // true

    // Find shortest path
    let path = g.shortest_path(0, 4).unwrap();
    println!("Path 0→4: {:?}", path); // [0, 1, 2, 3, 4]
}
```

---

## Per-Module Documentation

### Graph (Undirected Unweighted)

```rust
use mathverse_graph::Graph;

let mut g = Graph::new(6);
g.add_edge(0, 1);
g.add_edge(0, 2);
g.add_edge(1, 3);
g.add_edge(2, 4);
g.add_edge(4, 5);

let bfs_order = g.bfs(0);
// [0, 1, 2, 3, 4, 5]

let components = g.connected_components();
// [[0, 1, 2, 3, 4, 5]]

assert!(g.is_bipartite()); // path graph is bipartite
```

### WeightedGraph (Directed Weighted)

```rust
use mathverse_graph::WeightedGraph;

let mut g = WeightedGraph::new(5);
g.add_undirected_edge(0, 1, 1.0);
g.add_undirected_edge(0, 2, 4.0);
g.add_undirected_edge(1, 2, 2.0);
g.add_undirected_edge(1, 3, 6.0);
g.add_undirected_edge(2, 3, 3.0);

// Dijkstra's algorithm
let (dist, pred) = g.dijkstra(0);
// d(0)=0, d(1)=1, d(2)=3, d(3)=6

// Bellman-Ford with negative cycle detection
let (dist, pred, has_neg_cycle) = g.bellman_ford(0);

// All-pairs shortest paths
let matrix = g.floyd_warshall();

// Minimum spanning tree
if let Some(mst) = g.prim() {
    let total: f64 = mst.iter().map(|&(_, _, w)| w).sum();
    println!("MST total weight: {}", total);
}
```

### DirectedGraph

```rust
use mathverse_graph::DirectedGraph;

let mut g = DirectedGraph::new(5);
g.add_edge(0, 1);
g.add_edge(0, 2);
g.add_edge(1, 3);
g.add_edge(2, 3);
g.add_edge(2, 4);

// Topological sort (Kahn's algorithm)
match g.topological_sort() {
    Some(order) => println!("Topological order: {:?}", order),
    None => println!("Graph has a cycle!"),
}

// Strongly connected components (Tarjan's)
let mut g2 = DirectedGraph::new(5);
g2.add_edge(0, 1); g2.add_edge(1, 2); g2.add_edge(2, 0);
g2.add_edge(2, 3); g2.add_edge(3, 4); g2.add_edge(4, 3);
let sccs = g2.scc();
// [[0, 1, 2], [3, 4]]
```

---

## Future Scope

- Max flow (Ford-Fulkerson, Edmonds-Karp)
- Bipartite matching (Hopcroft-Karp)
- Graph coloring and chromatic polynomial
- Centrality measures (betweenness, closeness, eigenvector)
- Community detection (Louvain method)

## License

MIT — see [LICENSE](LICENSE).
