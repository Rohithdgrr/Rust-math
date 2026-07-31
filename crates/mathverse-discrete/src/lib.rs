//! Discrete mathematics: boolean logic, set operations, functions,
//! combinatorics, graph theory, number theory, recurrence relations.

use std::collections::HashSet;

pub mod combinatorics;
pub mod graph;
pub mod number_theory;
pub mod recurrence;

pub use combinatorics::Combinatorics;
pub use graph::{DirectedGraph, UndirectedGraph};
pub use number_theory::NumberTheory;
pub use recurrence::RecurrenceRelations;

/// All `2^n` truth assignments, one `Vec<bool>` per row (bit `b` = variable `b`).
pub fn truth_table(n: usize) -> Vec<Vec<bool>> {
    (0..(1usize << n))
        .map(|i| (0..n).map(|b| i & (1 << b) != 0).collect())
        .collect()
}

pub fn implies(a: bool, b: bool) -> bool {
    !a || b
}
pub fn iff(a: bool, b: bool) -> bool {
    a == b
}
pub fn xor(a: bool, b: bool) -> bool {
    a != b
}
pub fn nand(a: bool, b: bool) -> bool {
    !(a && b)
}
pub fn nor(a: bool, b: bool) -> bool {
    !(a || b)
}

/// Union of two sorted sets.
pub fn set_union(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let av = a.get(i).copied();
        let bv = b.get(j).copied();
        match (av, bv) {
            (Some(x), Some(y)) if x == y => {
                out.push(x);
                i += 1;
                j += 1;
            }
            (Some(x), Some(y)) if x < y => {
                out.push(x);
                i += 1;
            }
            (Some(_), Some(y)) => {
                out.push(y);
                j += 1;
            }
            (Some(x), None) => {
                out.push(x);
                i += 1;
            }
            (None, Some(y)) => {
                out.push(y);
                j += 1;
            }
            (None, None) => break,
        }
    }
    out
}

/// Intersection of two sorted sets.
pub fn set_intersection(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Elements of `a` not in `b`.
pub fn set_difference(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() {
        if j >= b.len() || a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            i += 1;
        }
    }
    out
}

/// `a ⊆ b` (sorted inputs).
pub fn is_subset(a: &[u64], b: &[u64]) -> bool {
    let mut j = 0;
    for &x in a {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        if j >= b.len() || b[j] != x {
            return false;
        }
    }
    true
}

/// `f ∘ g`: `(f ∘ g)(x) = f(g(x))`.
pub fn compose<'a>(f: &'a dyn Fn(f64) -> f64, g: &'a dyn Fn(f64) -> f64) -> impl Fn(f64) -> f64 + 'a {
    move |x| f(g(x))
}

/// Whether `f` maps distinct domain elements to distinct images.
pub fn is_injective(f: &dyn Fn(usize) -> usize, domain: &[usize]) -> bool {
    let mut seen = HashSet::new();
    domain.iter().all(|&x| seen.insert(f(x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logic() {
        assert_eq!(truth_table(2).len(), 4);
        assert_eq!(truth_table(2)[0], vec![false, false]);
        assert_eq!(truth_table(2)[3], vec![true, true]);
        assert!(!implies(true, false));
        assert!(implies(false, true));
        assert!(xor(true, false) && !xor(true, true));
        assert!(iff(true, true) && !iff(true, false));
        assert!(!nand(true, true));
        assert!(!nor(false, true));
    }

    #[test]
    fn sets() {
        let a = [1, 2, 3, 5];
        let b = [2, 3, 4];
        assert_eq!(set_union(&a, &b), vec![1, 2, 3, 4, 5]);
        assert_eq!(set_intersection(&a, &b), vec![2, 3]);
        assert_eq!(set_difference(&a, &b), vec![1, 5]);
        assert!(is_subset(&[2, 3], &a));
        assert!(!is_subset(&[2, 4], &a));
        assert!(is_subset(&[], &a));
    }

    #[test]
    fn functions() {
        let f = |x: f64| x + 1.0;
        let g = |x: f64| x * 2.0;
        assert_eq!(compose(&f, &g)(3.0), 7.0);
        let inj = |x: usize| x * 2;
        assert!(is_injective(&inj, &[1, 2, 3]));
        let not_inj = |x: usize| x % 2;
        assert!(!is_injective(&not_inj, &[1, 2, 3]));
    }
}
