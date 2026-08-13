//! Connected component labeling (the `cv2.connectedComponents` equivalent).
//!
//! Foreground pixels are those with value `> 0.5`. Labels are 1-based;
//! background is `0`. The label vector is row-major, matching [`crate::Image`]
//! data order.

use crate::Image;

/// Union-find (disjoint set) with path compression and union by rank.
struct DisjointSet {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // Path compression.
        let mut cur = x;
        while self.parent[cur as usize] != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (ra, rb) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else {
            (ra, rb)
        };
        self.parent[rb as usize] = ra;
        if self.rank[ra as usize] == self.rank[rb as usize] {
            self.rank[ra as usize] += 1;
        }
    }
}

/// Labels connected components of a binary image.
///
/// `connectivity` must be `4` (edge neighbors) or `8` (edge + corner
/// neighbors). Returns `(labels, count)` where `count` excludes the
/// background label `0`. Equivalent to `cv2.connectedComponents(img, 8)`.
///
/// # Panics
///
/// Panics if `connectivity` is not 4 or 8.
pub fn connected_components(img: &Image, connectivity: usize) -> (Vec<u32>, u32) {
    assert!(connectivity == 4 || connectivity == 8, "connectivity must be 4 or 8");
    let (w, h) = (img.w, img.h);
    let n = w * h;
    let mut labels = vec![0u32; n];
    if n == 0 {
        return (labels, 0);
    }

    let mut dsu = DisjointSet::new(n);
    let mut next_label: u32 = 1;

    // First pass: provisional labels + equivalence recording.
    for y in 0..h {
        for x in 0..w {
            if img.data[y * w + x] <= 0.5 {
                continue;
            }
            // Gather neighbor labels (top-left, top, top-right, left).
            let mut neighbor_labels = Vec::with_capacity(4);
            let offsets: &[(i64, i64)] = if connectivity == 8 {
                &[(-1, -1), (0, -1), (1, -1), (-1, 0)]
            } else {
                &[(0, -1), (-1, 0)]
            };
            for (dx, dy) in offsets {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                    let l = labels[ny as usize * w + nx as usize];
                    if l != 0 {
                        neighbor_labels.push(l);
                    }
                }
            }
            if neighbor_labels.is_empty() {
                labels[y * w + x] = next_label;
                next_label += 1;
            } else {
                let min_label = *neighbor_labels.iter().min().unwrap();
                labels[y * w + x] = min_label;
                for &l in &neighbor_labels {
                    dsu.union(min_label, l);
                }
            }
        }
    }

    // Second pass: resolve equivalences and renumber 1..=count.
    let mut remap = vec![0u32; next_label as usize];
    let mut count: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let l = labels[y * w + x];
            if l == 0 {
                continue;
            }
            let root = dsu.find(l);
            if remap[root as usize] == 0 {
                count += 1;
                remap[root as usize] = count;
            }
            labels[y * w + x] = remap[root as usize];
        }
    }
    (labels, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_separate_blobs_4conn() {
        let mut img = Image::new(10, 10);
        // Blob A at (1..3, 1..3).
        for y in 1..3 {
            for x in 1..3 {
                img.set(x, y, 1.0);
            }
        }
        // Blob B at (6..8, 6..8), separated by a gap.
        for y in 6..8 {
            for x in 6..8 {
                img.set(x, y, 1.0);
            }
        }
        let (labels, count) = connected_components(&img, 4);
        assert_eq!(count, 2);
        let l1 = labels[1 * 10 + 1];
        let l2 = labels[6 * 10 + 6];
        assert!(l1 != 0 && l2 != 0 && l1 != l2);
        // Background stays 0.
        assert_eq!(labels[0], 0);
        assert_eq!(labels[9 * 10 + 9], 0);
    }

    #[test]
    fn diagonal_connects_only_in_8() {
        let mut img = Image::new(3, 3);
        img.set(0, 0, 1.0);
        img.set(2, 2, 1.0);
        let (_, c4) = connected_components(&img, 4);
        assert_eq!(c4, 2);
        // 8-connectivity: make them touch diagonally.
        img.set(1, 1, 1.0);
        let (_, c8) = connected_components(&img, 8);
        assert_eq!(c8, 1);
    }

    #[test]
    fn single_blob_has_one_component() {
        let mut img = Image::new(6, 6);
        for y in 1..5 {
            for x in 1..5 {
                img.set(x, y, 1.0);
            }
        }
        let (_, count) = connected_components(&img, 8);
        assert_eq!(count, 1);
    }

    #[test]
    fn empty_image() {
        let img = Image::new(4, 4);
        let (labels, count) = connected_components(&img, 4);
        assert_eq!(count, 0);
        assert!(labels.iter().all(|&l| l == 0));
    }
}
