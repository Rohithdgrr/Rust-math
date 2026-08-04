//! Categorical axis support for string-label axes.

use std::collections::HashMap;

/// A mapping between string labels and numeric positions.
#[derive(Debug, Clone, Default)]
pub struct CategoryMap {
    /// Label-to-index mapping.
    labels: Vec<String>,
    /// Reverse mapping: index -> label.
    index_map: HashMap<String, usize>,
}

impl CategoryMap {
    /// Create a new empty category map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a category map from an iterator of labels.
    pub fn from_labels(labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut map = Self::new();
        for label in labels {
            map.add_label(label);
        }
        map
    }

    /// Add a label and return its numeric position.
    pub fn add_label(&mut self, label: impl Into<String>) -> f64 {
        let label = label.into();
        if let Some(&idx) = self.index_map.get(&label) {
            idx as f64
        } else {
            let idx = self.labels.len();
            self.index_map.insert(label.clone(), idx);
            self.labels.push(label);
            idx as f64
        }
    }

    /// Get the numeric position for a label, or `None` if not found.
    pub fn position_of(&self, label: &str) -> Option<f64> {
        self.index_map.get(label).map(|&i| i as f64)
    }

    /// Get the label at a numeric position, or `None` if out of bounds.
    pub fn label_at(&self, position: f64) -> Option<&str> {
        let idx = position.round() as usize;
        self.labels.get(idx).map(|s| s.as_str())
    }

    /// Get all labels in order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Get the number of categories.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Check if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Generate tick positions and labels for all categories.
    pub fn ticks(&self) -> Vec<(f64, String)> {
        self.labels
            .iter()
            .enumerate()
            .map(|(i, label)| (i as f64, label.clone()))
            .collect()
    }
}

/// Configuration for categorical axis rendering.
#[derive(Debug, Clone)]
pub struct CategoricalAxis {
    /// The category map for this axis.
    pub categories: CategoryMap,
    /// Rotation angle for labels (in degrees). 0 = horizontal, 90 = vertical.
    pub label_rotation: f64,
    /// Font size for category labels.
    pub font_size: f64,
    /// Whether to show grid lines at category positions.
    pub show_grid: bool,
}

impl CategoricalAxis {
    /// Create a new categorical axis from a category map.
    pub fn new(categories: CategoryMap) -> Self {
        Self {
            categories,
            label_rotation: 0.0,
            font_size: 11.0,
            show_grid: true,
        }
    }

    /// Set label rotation in degrees.
    pub fn with_label_rotation(mut self, degrees: f64) -> Self {
        self.label_rotation = degrees;
        self
    }

    /// Set font size for labels.
    pub fn with_font_size(mut self, size: f64) -> Self {
        self.font_size = size;
        self
    }

    /// Set whether to show grid lines.
    pub fn with_grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_map_basic() {
        let mut map = CategoryMap::new();
        assert_eq!(map.add_label("A"), 0.0);
        assert_eq!(map.add_label("B"), 1.0);
        assert_eq!(map.add_label("C"), 2.0);
        assert_eq!(map.add_label("A"), 0.0); // duplicate
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn category_map_from_labels() {
        let map = CategoryMap::from_labels(vec!["X", "Y", "Z"]);
        assert_eq!(map.position_of("X"), Some(0.0));
        assert_eq!(map.position_of("Y"), Some(1.0));
        assert_eq!(map.position_of("Z"), Some(2.0));
        assert_eq!(map.label_at(0.0), Some("X"));
        assert_eq!(map.label_at(1.0), Some("Y"));
        assert_eq!(map.label_at(2.0), Some("Z"));
    }

    #[test]
    fn category_ticks() {
        let map = CategoryMap::from_labels(vec!["A", "B", "C"]);
        let ticks = map.ticks();
        assert_eq!(ticks.len(), 3);
        assert_eq!(ticks[0], (0.0, "A".to_string()));
        assert_eq!(ticks[2], (2.0, "C".to_string()));
    }
}
