use std::collections::{VecDeque, BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

#[derive(Debug, Clone, Default)]
pub struct Graph {
    n: usize,
    adj: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new(n: usize) -> Self { Self { n, adj: vec![Vec::new(); n] } }
    pub fn add_edge(&mut self, u: usize, v: usize) { self.adj[u].push(v); self.adj[v].push(u); }
    pub fn add_directed_edge(&mut self, u: usize, v: usize) { self.adj[u].push(v); }
    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }
    pub fn neighbors(&self, u: usize) -> &[usize] { &self.adj[u] }
    pub fn degree(&self, u: usize) -> usize { self.adj[u].len() }

    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.n);
        let mut visited = vec![false; self.n];
        let mut q = VecDeque::from([start]);
        visited[start] = true;
        while let Some(u) = q.pop_front() {
            order.push(u);
            for &v in &self.adj[u] {
                if !visited[v] { visited[v] = true; q.push_back(v); }
            }
        }
        order
    }

    pub fn dfs(&self, start: usize) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.n);
        let mut visited = vec![false; self.n];
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            if visited[u] { continue; }
            visited[u] = true;
            order.push(u);
            for &v in self.adj[u].iter().rev() {
                if !visited[v] { stack.push(v); }
            }
        }
        order
    }

    pub fn shortest_path(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        if start == end { return Some(vec![start]); }
        let mut parent = vec![None; self.n];
        let mut visited = vec![false; self.n];
        let mut q = VecDeque::from([start]);
        visited[start] = true;
        while let Some(u) = q.pop_front() {
            for &v in &self.adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = Some(u);
                    if v == end {
                        let mut path = vec![v];
                        let mut cur = u;
                        while cur != start { path.push(cur); cur = parent[cur]?; }
                        path.push(start);
                        path.reverse();
                        return Some(path);
                    }
                    q.push_back(v);
                }
            }
        }
        None
    }

    pub fn is_connected(&self) -> bool { self.n == 0 || self.bfs(0).len() == self.n }

    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.n];
        let mut components = Vec::new();
        for i in 0..self.n {
            if !visited[i] {
                let mut comp = Vec::new();
                let mut q = VecDeque::from([i]);
                visited[i] = true;
                while let Some(u) = q.pop_front() {
                    comp.push(u);
                    for &v in &self.adj[u] {
                        if !visited[v] { visited[v] = true; q.push_back(v); }
                    }
                }
                components.push(comp);
            }
        }
        components
    }

    pub fn has_cycle(&self) -> bool {
        let mut visited = vec![false; self.n];
        for i in 0..self.n {
            if !visited[i] {
                let mut stack = vec![(i, None::<usize>)];
                while let Some((u, parent)) = stack.pop() {
                    if visited[u] { return true; }
                    visited[u] = true;
                    for &v in &self.adj[u] {
                        if Some(v) != parent { stack.push((v, Some(u))); }
                    }
                }
            }
        }
        false
    }

    pub fn is_bipartite(&self) -> bool {
        let mut color = vec![None; self.n];
        for i in 0..self.n {
            if color[i].is_some() { continue; }
            color[i] = Some(0u8);
            let mut q = VecDeque::from([i]);
            while let Some(u) = q.pop_front() {
                for &v in &self.adj[u] {
                    if color[v] == Some(color[u].unwrap()) { return false; }
                    if color[v].is_none() {
                        color[v] = Some(1 - color[u].unwrap());
                        q.push_back(v);
                    }
                }
            }
        }
        true
    }
}

#[derive(Debug, Clone, Default)]
pub struct WeightedGraph {
    n: usize,
    adj: Vec<Vec<(usize, f64)>>,
}

impl WeightedGraph {
    pub fn new(n: usize) -> Self { Self { n, adj: vec![Vec::new(); n] } }
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) { self.adj[u].push((v, w)); }
    pub fn add_undirected_edge(&mut self, u: usize, v: usize, w: f64) { self.adj[u].push((v, w)); self.adj[v].push((u, w)); }
    pub fn len(&self) -> usize { self.n }
    pub fn neighbors(&self, u: usize) -> &[(usize, f64)] { &self.adj[u] }

    pub fn dijkstra(&self, source: usize) -> (Vec<f64>, Vec<Option<usize>>) {
        let mut dist = vec![f64::INFINITY; self.n];
        let mut pred = vec![None; self.n];
        dist[source] = 0.0;
        #[derive(Clone)]
        struct State { dist: f64, vertex: usize }
        impl PartialEq for State { fn eq(&self, o: &Self) -> bool { self.dist == o.dist && self.vertex == o.vertex } }
        impl Eq for State {}
        impl Ord for State { fn cmp(&self, o: &Self) -> Ordering { o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal) } }
        impl PartialOrd for State { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        let mut heap = BinaryHeap::new();
        heap.push(State { dist: 0.0, vertex: source });
        while let Some(State { dist: d, vertex: u }) = heap.pop() {
            if d > dist[u] { continue; }
            for &(v, w) in &self.adj[u] {
                let nd = dist[u] + w;
                if nd < dist[v] { dist[v] = nd; pred[v] = Some(u); heap.push(State { dist: nd, vertex: v }); }
            }
        }
        (dist, pred)
    }

    pub fn bellman_ford(&self, source: usize) -> (Vec<f64>, Vec<Option<usize>>, bool) {
        let mut dist = vec![f64::INFINITY; self.n];
        let mut pred = vec![None; self.n];
        dist[source] = 0.0;
        for _ in 0..self.n - 1 {
            for u in 0..self.n {
                if dist[u] == f64::INFINITY { continue; }
                for &(v, w) in &self.adj[u] {
                    if dist[u] + w < dist[v] { dist[v] = dist[u] + w; pred[v] = Some(u); }
                }
            }
        }
        let mut neg_cycle = false;
        for u in 0..self.n {
            if dist[u] == f64::INFINITY { continue; }
            for &(v, w) in &self.adj[u] {
                if dist[u] + w < dist[v] { neg_cycle = true; break; }
            }
        }
        (dist, pred, neg_cycle)
    }

    pub fn floyd_warshall(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut d = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n { d[i][i] = 0.0; }
        for u in 0..n { for &(v, w) in &self.adj[u] { d[u][v] = w.min(d[u][v]); } }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if d[i][k] + d[k][j] < d[i][j] { d[i][j] = d[i][k] + d[k][j]; }
                }
            }
        }
        d
    }

    pub fn prim(&self) -> Option<Vec<(usize, usize, f64)>> {
        if self.n == 0 { return Some(Vec::new()); }
        let mut visited = vec![false; self.n];
        let mut mst = Vec::new();
        let mut heap = BinaryHeap::new();
        #[derive(Clone)]
        struct Edge { weight: f64, u: usize, v: usize }
        impl PartialEq for Edge { fn eq(&self, o: &Self) -> bool { self.weight == o.weight && self.u == o.u && self.v == o.v } }
        impl Eq for Edge {}
        impl Ord for Edge { fn cmp(&self, o: &Self) -> Ordering { o.weight.partial_cmp(&self.weight).unwrap_or(Ordering::Equal) } }
        impl PartialOrd for Edge { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        visited[0] = true;
        for &(v, w) in &self.adj[0] { heap.push(Edge { weight: w, u: 0, v }); }
        while let Some(Edge { weight, u, v }) = heap.pop() {
            if visited[v] { continue; }
            visited[v] = true;
            mst.push((u, v, weight));
            for &(next, w) in &self.adj[v] {
                if !visited[next] { heap.push(Edge { weight: w, u: v, v: next }); }
            }
        }
        if visited.iter().all(|&v| v) { Some(mst) } else { None }
    }

    pub fn kruskal(&self) -> Vec<(usize, usize, f64)> {
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..self.n { for &(v, w) in &self.adj[u] { if u < v { edges.push((w, u, v)); } } }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let mut parent: Vec<usize> = (0..self.n).collect();
        let mut rank = vec![0; self.n];
        let mut mst = Vec::new();
        for (w, u, v) in edges {
            let (mut pu, mut pv) = (u, v);
            while parent[pu] != pu { parent[pu] = parent[parent[pu]]; pu = parent[pu]; }
            while parent[pv] != pv { parent[pv] = parent[parent[pv]]; pv = parent[pv]; }
            if pu != pv {
                if rank[pu] < rank[pv] { parent[pu] = pv; } else if rank[pu] > rank[pv] { parent[pv] = pu; } else { parent[pv] = pu; rank[pu] += 1; }
                mst.push((u, v, w));
            }
        }
        mst
    }
}

#[derive(Debug, Clone, Default)]
pub struct DirectedGraph {
    n: usize,
    adj: Vec<Vec<usize>>,
}

impl DirectedGraph {
    pub fn new(n: usize) -> Self { Self { n, adj: vec![Vec::new(); n] } }
    pub fn add_edge(&mut self, u: usize, v: usize) { self.adj[u].push(v); }
    pub fn len(&self) -> usize { self.n }

    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut in_degree = vec![0; self.n];
        for u in 0..self.n { for &v in &self.adj[u] { in_degree[v] += 1; } }
        let mut queue: VecDeque<usize> = in_degree.iter().enumerate().filter(|&(_, &d)| d == 0).map(|(i, _)| i).collect();
        let mut result = Vec::new();
        while let Some(u) = queue.pop_front() {
            result.push(u);
            for &v in &self.adj[u] { in_degree[v] -= 1; if in_degree[v] == 0 { queue.push_back(v); } }
        }
        if result.len() == self.n { Some(result) } else { None }
    }

    pub fn scc(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.n];
        let mut order = Vec::new();
        fn dfs1(adj: &[Vec<usize>], u: usize, visited: &mut [bool], order: &mut Vec<usize>) {
            visited[u] = true;
            for &v in &adj[u] { if !visited[v] { dfs1(adj, v, visited, order); } }
            order.push(u);
        }
        for i in 0..self.n { if !visited[i] { dfs1(&self.adj, i, &mut visited, &mut order); } }
        let mut radj = vec![Vec::new(); self.n];
        for u in 0..self.n { for &v in &self.adj[u] { radj[v].push(u); } }
        let mut visited2 = vec![false; self.n];
        let mut components = Vec::new();
        for &u in order.iter().rev() {
            if visited2[u] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![u];
            visited2[u] = true;
            while let Some(v) = stack.pop() {
                comp.push(v);
                for &w in &radj[v] { if !visited2[w] { visited2[w] = true; stack.push(w); } }
            }
            components.push(comp);
        }
        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversal() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1); g.add_edge(1, 2);
        assert_eq!(g.bfs(0).len(), 3);
        assert!(g.is_connected());
    }

    #[test]
    fn shortest() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 3);
        assert_eq!(g.shortest_path(0, 3), Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn dijkstra_test() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0); g.add_edge(0, 2, 4.0); g.add_edge(1, 2, 2.0);
        let (dist, _) = g.dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn components() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1); g.add_edge(2, 3);
        assert_eq!(g.connected_components().len(), 2);
    }

    #[test]
    fn bipartite() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 3);
        assert!(g.is_bipartite());
    }

    #[test]
    fn topo() {
        let mut g = DirectedGraph::new(3);
        g.add_edge(0, 1); g.add_edge(1, 2);
        assert!(g.topological_sort().is_some());
    }
}
