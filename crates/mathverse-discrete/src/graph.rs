//! Graph theory: basic graph algorithms, traversals, shortest paths.

use std::collections::{HashMap, HashSet, VecDeque};

/// Directed graph using adjacency list representation.
#[derive(Debug, Clone)]
pub struct DirectedGraph {
    pub adjacency: HashMap<usize, Vec<usize>>,
    pub vertices: HashSet<usize>,
}

impl DirectedGraph {
    /// Create a new empty directed graph.
    pub fn new() -> Self {
        DirectedGraph {
            adjacency: HashMap::new(),
            vertices: HashSet::new(),
        }
    }

    /// Add a vertex to the graph.
    pub fn add_vertex(&mut self, v: usize) {
        self.vertices.insert(v);
        self.adjacency.entry(v).or_insert_with(Vec::new);
    }

    /// Add a directed edge from u to v.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.add_vertex(u);
        self.add_vertex(v);
        self.adjacency.entry(u).or_insert_with(Vec::new).push(v);
    }

    /// Get neighbors of vertex v.
    pub fn neighbors(&self, v: usize) -> &[usize] {
        self.adjacency.get(&v).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Check if edge (u, v) exists.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.neighbors(u).contains(&v)
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|v| v.len()).sum()
    }

    /// In-degree of vertex v.
    pub fn in_degree(&self, v: usize) -> usize {
        self.adjacency.values().filter(|neighbors| neighbors.contains(&v)).count()
    }

    /// Out-degree of vertex v.
    pub fn out_degree(&self, v: usize) -> usize {
        self.neighbors(v).len()
    }

    /// BFS traversal starting from vertex v.
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(v) = queue.pop_front() {
            result.push(v);
            
            for &neighbor in self.neighbors(v) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        result
    }

    /// DFS traversal starting from vertex v.
    pub fn dfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        self.dfs_recursive(start, &mut visited, &mut result);
        result
    }

    fn dfs_recursive(&self, v: usize, visited: &mut HashSet<usize>, result: &mut Vec<usize>) {
        if visited.contains(&v) {
            return;
        }
        
        visited.insert(v);
        result.push(v);
        
        for &neighbor in self.neighbors(v) {
            self.dfs_recursive(neighbor, visited, result);
        }
    }

    /// Check if graph is connected (undirected interpretation).
    pub fn is_connected(&self) -> bool {
        if self.vertices.is_empty() {
            return true;
        }
        
        let start = *self.vertices.iter().next().unwrap();
        let visited = self.bfs(start);
        
        visited.len() == self.vertex_count()
    }

    /// Detect cycle in directed graph.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();
        
        for &v in &self.vertices {
            if self.has_cycle_recursive(v, &mut visited, &mut recursion_stack) {
                return true;
            }
        }
        
        false
    }

    fn has_cycle_recursive(
        &self,
        v: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
    ) -> bool {
        if recursion_stack.contains(&v) {
            return true;
        }
        
        if visited.contains(&v) {
            return false;
        }
        
        visited.insert(v);
        recursion_stack.insert(v);
        
        for &neighbor in self.neighbors(v) {
            if self.has_cycle_recursive(neighbor, visited, recursion_stack) {
                return true;
            }
        }
        
        recursion_stack.remove(&v);
        false
    }

    /// Topological sort of DAG.
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        if self.has_cycle() {
            return None;
        }
        
        let mut in_degrees: HashMap<usize, usize> = HashMap::new();
        for &v in &self.vertices {
            in_degrees.insert(v, self.in_degree(v));
        }
        
        let mut queue = VecDeque::new();
        for (&v, &deg) in &in_degrees {
            if deg == 0 {
                queue.push_back(v);
            }
        }
        
        let mut result = Vec::new();
        
        while let Some(v) = queue.pop_front() {
            result.push(v);
            
            for &neighbor in self.neighbors(v) {
                let deg = in_degrees.get_mut(&neighbor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
        
        Some(result)
    }

    /// Shortest path using BFS (unweighted graph).
    pub fn shortest_path_bfs(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<usize, usize> = HashMap::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(v) = queue.pop_front() {
            if v == end {
                // Reconstruct path
                let mut path = vec![end];
                let mut current = end;
                
                while current != start {
                    if let Some(&p) = parent.get(&current) {
                        path.push(p);
                        current = p;
                    } else {
                        return None;
                    }
                }
                
                path.reverse();
                return Some(path);
            }
            
            for &neighbor in self.neighbors(v) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, v);
                    queue.push_back(neighbor);
                }
            }
        }
        
        None
    }

    /// Compute strongly connected components using Kosaraju's algorithm.
    pub fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        
        // First pass: fill order
        for &v in &self.vertices {
            if !visited.contains(&v) {
                self.dfs_order(v, &mut visited, &mut order);
            }
        }
        
        // Create transpose graph
        let transpose = self.transpose();
        
        // Second pass: find SCCs
        let mut visited = HashSet::new();
        let mut sccs = Vec::new();
        
        for v in order.into_iter().rev() {
            if !visited.contains(&v) {
                let mut component = Vec::new();
                transpose.dfs_collect(v, &mut visited, &mut component);
                sccs.push(component);
            }
        }
        
        sccs
    }

    fn dfs_order(&self, v: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
        visited.insert(v);
        
        for &neighbor in self.neighbors(v) {
            if !visited.contains(&neighbor) {
                self.dfs_order(neighbor, visited, order);
            }
        }
        
        order.push(v);
    }

    fn dfs_collect(&self, v: usize, visited: &mut HashSet<usize>, component: &mut Vec<usize>) {
        visited.insert(v);
        component.push(v);
        
        for &neighbor in self.neighbors(v) {
            if !visited.contains(&neighbor) {
                self.dfs_collect(neighbor, visited, component);
            }
        }
    }

    fn transpose(&self) -> Self {
        let mut graph = DirectedGraph::new();
        
        for &v in &self.vertices {
            graph.add_vertex(v);
        }
        
        for &u in &self.vertices {
            for &v in self.neighbors(u) {
                graph.add_edge(v, u);
            }
        }
        
        graph
    }
}

/// Undirected graph.
#[derive(Debug, Clone)]
pub struct UndirectedGraph {
    pub adjacency: HashMap<usize, Vec<usize>>,
    pub vertices: HashSet<usize>,
}

impl UndirectedGraph {
    /// Create a new empty undirected graph.
    pub fn new() -> Self {
        UndirectedGraph {
            adjacency: HashMap::new(),
            vertices: HashSet::new(),
        }
    }

    /// Add a vertex to the graph.
    pub fn add_vertex(&mut self, v: usize) {
        self.vertices.insert(v);
        self.adjacency.entry(v).or_insert_with(Vec::new);
    }

    /// Add an undirected edge between u and v.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.add_vertex(u);
        self.add_vertex(v);
        self.adjacency.entry(u).or_insert_with(Vec::new).push(v);
        self.adjacency.entry(v).or_insert_with(Vec::new).push(u);
    }

    /// Get neighbors of vertex v.
    pub fn neighbors(&self, v: usize) -> &[usize] {
        self.adjacency.get(&v).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Degree of vertex v.
    pub fn degree(&self, v: usize) -> usize {
        self.neighbors(v).len()
    }

    /// Check if graph is connected.
    pub fn is_connected(&self) -> bool {
        if self.vertices.is_empty() {
            return true;
        }
        
        let start = *self.vertices.iter().next().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(v) = queue.pop_front() {
            for &neighbor in self.neighbors(v) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        visited.len() == self.vertex_count()
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|v| v.len()).sum() / 2
    }

    /// Minimum spanning tree using Prim's algorithm.
    pub fn mst_prim(&self) -> Option<Vec<(usize, usize)>> {
        if self.vertices.is_empty() {
            return Some(Vec::new());
        }
        
        if !self.is_connected() {
            return None;
        }
        
        let start = *self.vertices.iter().next().unwrap();
        let mut in_mst = HashSet::new();
        let mut mst_edges = Vec::new();
        
        in_mst.insert(start);
        
        while in_mst.len() < self.vertex_count() {
            let mut min_edge = None;
            let mut min_weight = usize::MAX;
            
            for &u in &in_mst {
                for &v in self.neighbors(u) {
                    if !in_mst.contains(&v) {
                        // Use 1 as weight for unweighted graph
                        if 1 < min_weight {
                            min_weight = 1;
                            min_edge = Some((u, v));
                        }
                    }
                }
            }
            
            if let Some((u, v)) = min_edge {
                in_mst.insert(v);
                mst_edges.push((u, v));
            } else {
                return None;
            }
        }
        
        Some(mst_edges)
    }

    /// Detect cycle in undirected graph.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        
        for &v in &self.vertices {
            if !visited.contains(&v) {
                if self.has_cycle_undirected(v, None, &mut visited) {
                    return true;
                }
            }
        }
        
        false
    }

    fn has_cycle_undirected(
        &self,
        v: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> bool {
        visited.insert(v);
        
        for &neighbor in self.neighbors(v) {
            if !visited.contains(&neighbor) {
                if self.has_cycle_undirected(neighbor, Some(v), visited) {
                    return true;
                }
            } else if parent != Some(neighbor) {
                return true;
            }
        }
        
        false
    }

    /// Bipartite graph check using BFS coloring.
    pub fn is_bipartite(&self) -> bool {
        let mut color = HashMap::new();
        
        for &start in &self.vertices {
            if color.contains_key(&start) {
                continue;
            }
            
            let mut queue = VecDeque::new();
            queue.push_back(start);
            color.insert(start, 0);
            
            while let Some(v) = queue.pop_front() {
                for &neighbor in self.neighbors(v) {
                    if !color.contains_key(&neighbor) {
                        color.insert(neighbor, 1 - color[&v]);
                        queue.push_back(neighbor);
                    } else if color[&neighbor] == color[&v] {
                        return false;
                    }
                }
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directed_graph() {
        let mut g = DirectedGraph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(0, 2);
        
        assert_eq!(g.vertex_count(), 3);
        assert_eq!(g.edge_count(), 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(2, 0));
    }

    #[test]
    fn test_bfs() {
        let mut g = DirectedGraph::new();
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        g.add_edge(2, 3);
        
        let result = g.bfs(0);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(result.contains(&2));
        assert!(result.contains(&3));
    }

    #[test]
    fn test_dfs() {
        let mut g = DirectedGraph::new();
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        
        let result = g.dfs(0);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn test_topological_sort() {
        let mut g = DirectedGraph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(0, 2);
        
        let result = g.topological_sort();
        assert!(result.is_some());
        
        let order = result.unwrap();
        let pos0 = order.iter().position(|&x| x == 0).unwrap();
        let pos1 = order.iter().position(|&x| x == 1).unwrap();
        let pos2 = order.iter().position(|&x| x == 2).unwrap();
        
        assert!(pos0 < pos1);
        assert!(pos1 < pos2);
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = DirectedGraph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        
        assert!(g.has_cycle());
    }

    #[test]
    fn test_undirected_graph() {
        let mut g = UndirectedGraph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(0, 2);
        
        assert_eq!(g.vertex_count(), 3);
        assert_eq!(g.edge_count(), 3);
        assert!(g.is_connected());
    }

    #[test]
    fn test_bipartite() {
        let mut g = UndirectedGraph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        
        assert!(g.is_bipartite());
        
        g.add_edge(0, 2);
        assert!(!g.is_bipartite());
    }

    #[test]
    fn test_strongly_connected_components() {
        let mut g = DirectedGraph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        g.add_edge(3, 4);
        
        let sccs = g.strongly_connected_components();
        assert_eq!(sccs.len(), 2);
    }
}
