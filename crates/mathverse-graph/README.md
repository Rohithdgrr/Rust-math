# mathverse-graph

A Rust library for graph data structures and algorithms including BFS/DFS traversal, shortest paths (Dijkstra, Bellman-Ford, Floyd-Warshall), minimum spanning trees (Prim, Kruskal), connectivity analysis, topological sorting, and strongly connected components.

## Features

- **Graph types**: Undirected, weighted, and directed graphs with adjacency list representation
- **Traversal**: BFS and DFS with order tracking
- **Shortest paths**: Dijkstra's algorithm, Bellman-Ford (with negative cycle detection), Floyd-Warshall (all-pairs)
- **Minimum spanning trees**: Prim's and Kruskal's algorithms
- **Connectivity**: Connected components, cycle detection, bipartiteness testing
- **Directed graphs**: Topological sort (Kahn's algorithm), strongly connected components (Tarjan's)

## Module Overview

| Struct | Description | Key Methods |
|--------|-------------|-------------|
| `Graph` | Undirected unweighted graph | `bfs`, `dfs`, `shortest_path`, `is_connected`, `has_cycle`, `is_bipartite` |
| `WeightedGraph` | Directed weighted graph | `dijkstra`, `bellman_ford`, `floyd_warshall`, `prim`, `kruskal` |
| `DirectedGraph` | Directed unweighted graph | `topological_sort`, `scc` |

## ASCII Diagram: Graph Structures

### BFS vs DFS Traversal
```
Graph:       0---1---2
             |       |
             3---4---5

BFS from 0:  0 → 1 → 3 → 2 → 4 → 5
             (Level-order: visit neighbors first)

DFS from 0:  0 → 1 → 2 → 5 → 4 → 3
             (Depth-first: go deep before wide)
```

### BFS Tree
```
BFS Tree from node 0:
        0           Level 0
       / \
      1   3         Level 1
     /     \
    2       4       Level 2
              \
               5    Level 3

Distance from 0:
  d(0)=0, d(1)=1, d(3)=1, d(2)=2, d(4)=2, d(5)=3
```

### DFS Tree
```
DFS Tree from node 0:
        0           Root
        |
        1           Parent
        |
        2           Child
        |
        5           Child
       / \
      4   3         Leaves

Discovery/Finish times:
  0: [0, 11]
  1: [1, 10]
  2: [2, 7]
  3: [3, 4]
  4: [5, 6]
  5: [8, 9]
```

### Minimum Spanning Tree
```
Original Graph:           MST (Prim or Kruskal):
  2       4                 2       4
  /|     /|                 |      /
 1 |    / |                 1     /
 /|  3  / |                 |   3
0 |  | /| |                 0   |   5
 \|  |/ \|                  \  |  /
  6  5   7                   6  5

MST Edges: (0,1)=1, (1,2)=2, (0,3)=3, (3,5)=5, (3,4)=4
Total Weight: 15
```

### Shortest Path (Dijkstra)
```
Dijkstra from node 0:
  0 --1--> 1 --2--> 2
  |                 |
  4                 1
  v                 v
  3 --3--> 4       5

Shortest distances:
  d(0) = 0
  d(1) = 1  (0→1)
  d(2) = 3  (0→1→2)
  d(3) = 4  (0→3)
  d(4) = 7  (0→3→4)
  d(5) = 4  (0→1→2→5)
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mathverse-graph = { path = "../mathverse-graph" }
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

## Per-Module Documentation

### Graph (Undirected Unweighted)

Basic graph operations and traversal algorithms.

**Struct:**
```rust
pub struct Graph {
    n: usize,           // Number of vertices
    adj: Vec<Vec<usize>>, // Adjacency list
}
```

**Methods:**

- `new(n: usize) -> Self` — Create graph with n vertices
- `add_edge(u: usize, v: usize)` — Add undirected edge
- `add_directed_edge(u: usize, v: usize)` — Add directed edge (useful for mixed use)
- `neighbors(u: usize) -> &[usize]` — Get neighbors of u
- `degree(u: usize) -> usize` — Degree of vertex u
- `bfs(start: usize) -> Vec<usize>` — Breadth-first search order
- `dfs(start: usize) -> Vec<usize>` — Depth-first search order
- `shortest_path(start: usize, end: usize) -> Option<Vec<usize>>` — BFS shortest path
- `is_connected() -> bool` — Check connectivity
- `connected_components() -> Vec<Vec<usize>>` — Find all components
- `has_cycle() -> bool` — Cycle detection
- `is_bipartite() -> bool` — Bipartiteness test

**Example: BFS Traversal**
```rust
let mut g = Graph::new(6);
g.add_edge(0, 1);
g.add_edge(0, 2);
g.add_edge(1, 3);
g.add_edge(2, 4);
g.add_edge(4, 5);

let bfs_order = g.bfs(0);
println!("BFS: {:?}", bfs_order); // [0, 1, 2, 3, 4, 5]
```

**Example: Connected Components**
```rust
let mut g = Graph::new(6);
g.add_edge(0, 1);
g.add_edge(2, 3);
g.add_edge(4, 5);

let components = g.connected_components();
println!("Components: {:?}", components);
// [[0, 1], [2, 3], [4, 5]]
```

**Example: Bipartiteness**
```rust
// Path graph is bipartite
let mut g = Graph::new(4);
g.add_edge(0, 1);
g.add_edge(1, 2);
g.add_edge(2, 3);
assert!(g.is_bipartite());

// Triangle is not bipartite
let mut g2 = Graph::new(3);
g2.add_edge(0, 1);
g2.add_edge(1, 2);
g2.add_edge(2, 0);
assert!(!g2.is_bipartite());
```

**Use Cases:** Network analysis, social graphs, puzzle solving, maze solving.

### WeightedGraph (Directed Weighted)

Weighted graph algorithms for shortest paths and MST.

**Struct:**
```rust
pub struct WeightedGraph {
    n: usize,
    adj: Vec<Vec<(usize, f64)>>, // (neighbor, weight)
}
```

**Methods:**

- `new(n: usize) -> Self` — Create weighted graph
- `add_edge(u: usize, v: usize, w: f64)` — Add directed weighted edge
- `add_undirected_edge(u: usize, v: usize, w: f64)` — Add undirected weighted edge
- `dijkstra(source: usize) -> (Vec<f64>, Vec<Option<usize>>)` — Dijkstra's algorithm
- `bellman_ford(source: usize) -> (Vec<f64>, Vec<Option<usize>>, bool)` — Bellman-Ford (returns negative cycle flag)
- `floyd_warshall() -> Vec<Vec<f64>>` — All-pairs shortest paths
- `prim() -> Option<Vec<(usize, usize, f64)>>` — Prim's MST
- `kruskal() -> Vec<(usize, usize, f64)>` — Kruskal's MST

**Dijkstra's Algorithm Visualization:**
```
Dijkstra from source 0:

  Step 1:  dist = [0, ∞, ∞, ∞, ∞]
           visit 0, relax edges to 1 (w=1), 3 (w=4)

  Step 2:  dist = [0, 1, ∞, 4, ∞]
           visit 1, relax edge to 2 (w=2)

  Step 3:  dist = [0, 1, 3, 4, ∞]
           visit 2, relax edge to 5 (w=1)

  Step 4:  dist = [0, 1, 3, 4, 4]
           visit 5, done

Final: d[0]=0, d[1]=1, d[2]=3, d[3]=4, d[5]=4
```

**Bellman-Ford Negative Cycle Detection:**
```
Graph with negative cycle:
  0 --(1)--> 1 --(-2)--> 2 --(1)--> 3 --(-1)--> 1

After |V|-1 = 3 iterations:
  dist = [0, 1, -1, 0]

Extra iteration detects: dist[1] can still be improved
→ Negative cycle exists: 1 → 2 → 3 → 1 (total weight = -2)
```

**Example: Dijkstra's Algorithm**
```rust
let mut g = WeightedGraph::new(5);
g.add_undirected_edge(0, 1, 1.0);
g.add_undirected_edge(0, 2, 4.0);
g.add_undirected_edge(1, 2, 2.0);
g.add_undirected_edge(1, 3, 6.0);
g.add_undirected_edge(2, 3, 3.0);

let (dist, pred) = g.dijkstra(0);
println!("Shortest distances from 0:");
for (i, d) in dist.iter().enumerate() {
    println!("  d({}) = {:.1}", i, d);
}
// d(0) = 0.0, d(1) = 1.0, d(2) = 3.0, d(3) = 6.0
```

**Example: Bellman-Ford**
```rust
let mut g = WeightedGraph::new(3);
g.add_edge(0, 1, 1.0);
g.add_edge(0, 2, 5.0);
g.add_edge(1, 2, -2.0); // Negative weight

let (dist, pred, has_neg_cycle) = g.bellman_ford(0);
assert!(!has_neg_cycle); // No negative cycle reachable
println!("d(2) = {}", dist[2]); // 3.0 (0→1→2)
```

**Example: Prim's MST**
```rust
let mut g = WeightedGraph::new(4);
g.add_undirected_edge(0, 1, 1.0);
g.add_undirected_edge(0, 2, 3.0);
g.add_undirected_edge(1, 2, 2.0);
g.add_undirected_edge(2, 3, 4.0);

if let Some(mst) = g.prim() {
    println!("MST edges:");
    for (u, v, w) in &mst {
        println!("  {} -- {} (weight {})", u, v, w);
    }
    let total: f64 = mst.iter().map(|&(_, _, w)| w).sum();
    println!("Total weight: {}", total);
}
```

**Floyd-Warshall All-Pairs:**
```
Input Graph:              Distance Matrix (after Floyd-Warshall):
0 --1--> 1               [0, 1, 3, 4]
|       /|               [∞, 0, 2, 3]
4     1  |               [∞, ∞, 0, 1]
|   /    |               [∞, ∞, ∞, 0]
3 -2--> 2 --1--> 3

d[0][3] = 4 (0→1→2→3)
d[1][3] = 3 (1→2→3)
```

**Use Cases:** Route planning, network optimization, resource allocation, game pathfinding.

### DirectedGraph (Directed Unweighted)

Directed graph algorithms for ordering and strongly connected components.

**Struct:**
```rust
pub struct DirectedGraph {
    n: usize,
    adj: Vec<Vec<usize>>,
}
```

**Methods:**

- `new(n: usize) -> Self` — Create directed graph
- `add_edge(u: usize, v: usize)` — Add directed edge u→v
- `topological_sort() -> Option<Vec<usize>>` — Topological ordering (None if cycle)
- `scc() -> Vec<Vec<usize>>` — Strongly connected components (Tarjan's)

**Topological Sort (Kahn's Algorithm):**
```
DAG:        0 → 1 → 3
            ↓   ↓
            2 → 4

In-degrees: [0, 1, 1, 1, 2]
Queue: [0]

Step 1: Process 0 → [1, 2]    Result: [0]
Step 2: Process 1 → [3, 4]    Result: [0, 1]
Step 3: Process 2 → [4]       Result: [0, 1, 2]
Step 4: Process 3 → []        Result: [0, 1, 2, 3]
Step 5: Process 4 → []        Result: [0, 1, 2, 3, 4]

Valid topological order: [0, 1, 2, 3, 4]
```

**Strongly Connected Components:**
```
Graph with 2 SCCs:
  0 → 1 → 2 → 0   (SCC 1: {0, 1, 2})
  3 → 4 → 3        (SCC 2: {3, 4})
  2 → 3            (Edge between SCCs)

SCC Result: [[0, 1, 2], [3, 4]]
Condensation DAG: {0,1,2} → {3,4}
```

**Example: Topological Sort**
```rust
let mut g = DirectedGraph::new(5);
g.add_edge(0, 1);
g.add_edge(0, 2);
g.add_edge(1, 3);
g.add_edge(2, 3);
g.add_edge(2, 4);

match g.topological_sort() {
    Some(order) => println!("Topological order: {:?}", order),
    None => println!("Graph has a cycle!"),
}
// Topological order: [0, 1, 2, 3, 4] (or [0, 2, 1, 4, 3])
```

**Example: Strongly Connected Components**
```rust
let mut g = DirectedGraph::new(5);
g.add_edge(0, 1);
g.add_edge(1, 2);
g.add_edge(2, 0); // Cycle: 0→1→2→0
g.add_edge(2, 3);
g.add_edge(3, 4);
g.add_edge(4, 3); // Cycle: 3→4→3

let sccs = g.scc();
println!("SCCs: {:?}", sccs);
// [[0, 1, 2], [3, 4]] or similar
```

**Use Cases:** Task scheduling, dependency resolution, compilation order, dead code detection.

## Future Scope

- **Max flow**: Ford-Fulkerson, Edmonds-Karp, push-relabel
- **Matching**: Hopcroft-Karp for bipartite matching
- **Planarity**: Planarity testing, embedding
- **Coloring**: Graph coloring, chromatic polynomial
- **Centrality**: Betweenness, closeness, eigenvector centrality
- **Community detection**: Louvain method, label propagation
- **Dynamic graphs**: Incremental updates, sliding window connectivity

## License

This project is dual-licensed under **MIT** and **Apache-2.0** licenses. You may choose either license for your use.

- MIT License: See [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0: See [LICENSE-APACHE](LICENSE-APACHE)
