//! Random graphs: Erdős-Rényi (G(n,p) and G(n,m)), Barabási-Albert
//! preferential attachment, the configuration model, and bond percolation.

use crate::rng::Rng;

/// Undirected simple graph with `n` vertices, edges stored as `(u, v)` with
/// `u < v`.
#[must_use]
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<(usize, usize)>,
}

impl Graph {
    /// Empty graph on `n` vertices.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
        }
    }

    /// Add the undirected edge `(u, v)` (ignored if it already exists).
    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u == v || u >= self.n || v >= self.n {
            return;
        }
        let (a, b) = (u.min(v), u.max(v));
        if !self.edges.contains(&(a, b)) {
            self.edges.push((a, b));
        }
    }

    /// Whether the edge `(u, v)` exists.
    #[must_use]
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        let (a, b) = (u.min(v), u.max(v));
        self.edges.contains(&(a, b))
    }

    /// Number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Degree of vertex `v`.
    #[must_use]
    pub fn degree(&self, v: usize) -> usize {
        self.adjacency()[v].len()
    }

    /// Adjacency lists (one per vertex).
    #[must_use]
    pub fn adjacency(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.n];
        for &(a, b) in &self.edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        adj
    }

    /// Average degree.
    #[must_use]
    pub fn average_degree(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        2.0 * self.edges.len() as f64 / self.n as f64
    }

    /// Degree distribution (pmf over degrees `0..max_degree+1`).
    #[must_use]
    pub fn degree_distribution(&self) -> Vec<f64> {
        let adj = self.adjacency();
        let max_deg = adj.iter().map(|n| n.len()).max().unwrap_or(0);
        let mut hist = vec![0.0; max_deg + 1];
        for v in &adj {
            hist[v.len()] += 1.0;
        }
        for h in &mut hist {
            *h /= self.n as f64;
        }
        hist
    }

    /// Number of connected components (breadth-first search).
    #[must_use]
    pub fn connected_components(&self) -> usize {
        let adj = self.adjacency();
        let mut seen = vec![false; self.n];
        let mut count = 0;
        for start in 0..self.n {
            if seen[start] {
                continue;
            }
            count += 1;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(v) = stack.pop() {
                for &w in &adj[v] {
                    if !seen[w] {
                        seen[w] = true;
                        stack.push(w);
                    }
                }
            }
        }
        count
    }

    /// Size of the largest connected component.
    #[must_use]
    pub fn largest_component_size(&self) -> usize {
        let adj = self.adjacency();
        let mut seen = vec![false; self.n];
        let mut best = 0;
        for start in 0..self.n {
            if seen[start] {
                continue;
            }
            let mut size = 0;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(v) = stack.pop() {
                size += 1;
                for &w in &adj[v] {
                    if !seen[w] {
                        seen[w] = true;
                        stack.push(w);
                    }
                }
            }
            best = best.max(size);
        }
        best
    }
}

/// Erdős-Rényi `G(n, p)`: every pair is an edge independently with
/// probability `p`.
#[must_use]
pub fn erdos_renyi(n: usize, p: f64, rng: &mut Rng) -> Graph {
    let mut g = Graph::new(n);
    for i in 0..n {
        for j in i + 1..n {
            if rng.uniform() < p {
                g.add_edge(i, j);
            }
        }
    }
    g
}

/// Erdős-Rényi `G(n, m)`: exactly `m` distinct edges chosen uniformly.
#[must_use]
pub fn erdos_renyi_edges(n: usize, m: usize, rng: &mut Rng) -> Graph {
    let max_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
    let m = m.min(max_edges);
    let mut g = Graph::new(n);
    let mut attempts = 0;
    while g.edge_count() < m && attempts < m * 100 + 10_000 {
        let i = rng.below(n as u64) as usize;
        let j = rng.below(n as u64) as usize;
        if i != j {
            g.add_edge(i, j);
        }
        attempts += 1;
    }
    g
}

/// Barabási-Albert model: start from a path on `m` nodes, then attach each
/// new node with `m` edges, each target chosen with probability proportional
/// to current degree (preferential attachment, "rich get richer").
///
/// # Errors
/// Returns an error if `n <= m` or `m == 0`.
pub fn barabasi_albert(n: usize, m: usize, rng: &mut Rng) -> Result<Graph, String> {
    if m == 0 {
        return Err("barabasi_albert needs m >= 1".into());
    }
    if n <= m {
        return Err("barabasi_albert needs n > m".into());
    }
    let mut g = Graph::new(n);
    for i in 0..m - 1 {
        g.add_edge(i, i + 1);
    }
    let mut stubs: Vec<usize> = Vec::new();
    for i in 0..m - 1 {
        stubs.push(i);
        stubs.push(i + 1);
    }
    for new_node in m..n {
        let mut chosen: Vec<usize> = Vec::with_capacity(m);
        while chosen.len() < m {
            let candidate = stubs[rng.below(stubs.len() as u64) as usize];
            if !chosen.contains(&candidate) {
                chosen.push(candidate);
            }
        }
        for &c in &chosen {
            g.add_edge(new_node, c);
            stubs.push(new_node);
            stubs.push(c);
        }
    }
    Ok(g)
}

/// Configuration model: a random graph with the prescribed degree sequence,
/// sampled uniformly over stub pairings (self-loops and duplicate edges are
/// rejected and retried).
///
/// # Errors
/// Returns an error if the degree sum is odd or the pairing cannot be
/// completed (e.g. a star with a huge center, or an invalid sequence).
pub fn configuration_model(degrees: &[usize], rng: &mut Rng) -> Result<Graph, String> {
    let n = degrees.len();
    let sum: usize = degrees.iter().sum();
    if n == 0 {
        return Err("configuration model needs at least one vertex".into());
    }
    if sum % 2 != 0 {
        return Err("degree sequence must have even sum".into());
    }
    let mut stubs: Vec<usize> = Vec::with_capacity(sum);
    for (v, &d) in degrees.iter().enumerate() {
        if d >= n {
            return Err("degree exceeds the number of vertices".into());
        }
        for _ in 0..d {
            stubs.push(v);
        }
    }
    let mut g = Graph::new(n);
    let mut failures = 0usize;
    while !stubs.is_empty() {
        if stubs.len() < 2 {
            return Err("stub pairing stalled".into());
        }
        let i = rng.below(stubs.len() as u64) as usize;
        let mut j = rng.below(stubs.len() as u64) as usize;
        if i == j {
            j = (j + 1) % stubs.len();
        }
        let u = stubs[i];
        let v = stubs[j];
        let (lo, hi) = (i.min(j), i.max(j));
        stubs.swap_remove(hi);
        stubs.swap_remove(lo);
        if u != v && !g.has_edge(u, v) {
            g.add_edge(u, v);
            failures = 0;
        } else {
            stubs.push(u);
            stubs.push(v);
            failures += 1;
            if failures > 1_000 * sum.max(1) {
                return Err("unable to complete stub pairing".into());
            }
        }
    }
    Ok(g)
}

/// Bond percolation: keep each edge independently with probability `p`.
#[must_use]
pub fn percolate(g: &Graph, p: f64, rng: &mut Rng) -> Graph {
    let mut out = Graph::new(g.n);
    for &(a, b) in &g.edges {
        if rng.uniform() < p {
            out.add_edge(a, b);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erdos_renyi_edge_count_matches_p() {
        let n = 500;
        let p = 0.05;
        let mut rng = Rng::new(1);
        let g = erdos_renyi(n, p, &mut rng);
        let expected = p * n as f64 * (n as f64 - 1.0) / 2.0;
        let got = g.edge_count() as f64;
        assert!(
            (got - expected).abs() < 0.1 * expected,
            "edges {got} vs {expected}"
        );
        // Giant component present at p above the threshold 1/n.
        let largest = g.largest_component_size();
        assert!(largest as f64 > 0.9 * n as f64, "largest {largest}");
        assert!(g.connected_components() < 20);
        // Average degree ~ p(n-1).
        assert!((g.average_degree() - p * (n as f64 - 1.0)).abs() < 1.0);
    }

    #[test]
    fn erdos_renyi_edges_exact_count() {
        let mut rng = Rng::new(2);
        let g = erdos_renyi_edges(100, 300, &mut rng);
        assert_eq!(g.edge_count(), 300);
    }

    #[test]
    fn subcritical_er_has_many_components() {
        let n = 200;
        let mut rng = Rng::new(3);
        let g = erdos_renyi(n, 0.003, &mut rng);
        assert!(g.largest_component_size() < 30, "largest {}", g.largest_component_size());
        assert!(g.connected_components() > 20);
    }

    #[test]
    fn barabasi_albert_is_connected_and_heavy_tailed() {
        let mut rng = Rng::new(4);
        let (n, m) = (400, 2);
        let g = barabasi_albert(n, m, &mut rng).unwrap();
        assert_eq!(g.edge_count(), (n - 1) * m);
        assert_eq!(g.connected_components(), 1);
        let dist = g.degree_distribution();
        let max_deg = dist.len() - 1;
        let mean = g.average_degree();
        assert!(max_deg as f64 > 4.0 * mean, "max degree {max_deg} vs mean {mean}");
        // Degree-1 mass is large (scale-free).
        assert!(dist[1] > 0.1, "P(deg=1) {}", dist[1]);
        assert!(barabasi_albert(5, 5, &mut rng).is_err());
    }

    #[test]
    fn configuration_model_respects_degrees() {
        let mut rng = Rng::new(5);
        let degrees = vec![2usize, 2, 2, 2, 2, 2];
        let g = configuration_model(&degrees, &mut rng).unwrap();
        let adj = g.adjacency();
        for (v, &d) in degrees.iter().enumerate() {
            assert_eq!(adj[v].len(), d, "degree of {v}");
        }
        assert_eq!(g.edge_count(), 6);
        // Odd sum rejected.
        assert!(configuration_model(&[2, 3], &mut rng).is_err());
        // Star with isolated center: 4 leaves of degree 1, center degree 3.
        let degrees = vec![3usize, 1, 1, 1];
        let g2 = configuration_model(&degrees, &mut rng).unwrap();
        assert_eq!(g2.edge_count(), 3);
        let adj2 = g2.adjacency();
        assert_eq!(adj2[0].len(), 3);
    }

    #[test]
    fn percolation_limits() {
        let mut rng = Rng::new(6);
        let g = erdos_renyi(100, 0.1, &mut rng);
        let kept = percolate(&g, 1.0, &mut rng);
        assert_eq!(kept.edge_count(), g.edge_count());
        let none = percolate(&g, 0.0, &mut rng);
        assert_eq!(none.edge_count(), 0);
        assert_eq!(none.connected_components(), 100);
        // Half-percolated graph has between 0 and full edges.
        let half = percolate(&g, 0.5, &mut rng);
        assert!(half.edge_count() > 0 && half.edge_count() < g.edge_count());
    }

    #[test]
    fn graph_edge_helpers() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 0); // duplicate ignored
        g.add_edge(2, 3);
        assert_eq!(g.edge_count(), 2);
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 0));
        assert!(!g.has_edge(0, 2));
        assert_eq!(g.degree(1), 1);
        let dist = g.degree_distribution();
        assert_eq!(dist[0], 0.5); // two isolated vertices
        assert_eq!(dist[1], 0.5);
    }
}
