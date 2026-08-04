//! ML plot visualizations via `mathverse-machine-learning`.
//!
//! Provides:
//! - Confusion matrix heatmap
//! - ROC curve plot

use mathverse_machine_learning::model_selection::{confusion_matrix, roc_curve};

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::svg::SvgPlot;

/// Render a confusion matrix as a heatmap SVG.
///
/// `pred` and `target` are integer class labels (0..num_classes).
/// `num_classes` is the total number of classes.
pub fn render_confusion_matrix(
    pred: &[f64],
    target: &[f64],
    num_classes: usize,
    config: PlotConfig,
) -> PlotResult<String> {
    if pred.len() != target.len() {
        return Err(PlotError::InvalidData(
            "pred and target must have the same length".into(),
        ));
    }
    if num_classes == 0 {
        return Err(PlotError::InvalidData(
            "num_classes must be positive".into(),
        ));
    }

    let cm = confusion_matrix(pred, target, num_classes);

    let mut plot = SvgPlot::new(config);
    plot.add_heatmap("confusion_matrix", cm, crate::color::viridis)?;

    Ok(plot.generate())
}

/// Render an ROC curve as an SVG line plot.
///
/// `scores` are the predicted scores (higher = more positive).
/// `labels` are the true binary labels (0.0 or 1.0).
pub fn render_roc_curve(
    scores: &[f64],
    labels: &[f64],
    config: PlotConfig,
) -> PlotResult<String> {
    if scores.len() != labels.len() {
        return Err(PlotError::InvalidData(
            "scores and labels must have the same length".into(),
        ));
    }

    let points = roc_curve(scores, labels);

    let data_points: Vec<crate::DataPoint> = points
        .iter()
        .map(|&(fpr, tpr)| crate::DataPoint::new(fpr, tpr))
        .collect();

    let mut plot = SvgPlot::new(config);
    plot.add_series(crate::DataSeries::new("ROC".to_string(), data_points));

    Ok(plot.generate())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confusion_matrix_renders_svg() {
        let pred = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let target = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let config = PlotConfig::new()
            .with_title("Confusion Matrix")
            .with_x_label("Predicted")
            .with_y_label("Actual");

        let svg = render_confusion_matrix(&pred, &target, 2, config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn roc_curve_renders_svg() {
        let scores = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let labels = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let config = PlotConfig::new()
            .with_title("ROC Curve")
            .with_x_label("False Positive Rate")
            .with_y_label("True Positive Rate");

        let svg = render_roc_curve(&scores, &labels, config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn roc_curve_rejects_mismatched_lengths() {
        let scores = vec![0.9, 0.8, 0.7];
        let labels = vec![1.0, 0.0];
        let config = PlotConfig::new();

        let result = render_roc_curve(&scores, &labels, config);
        assert!(result.is_err());
    }

    #[test]
    fn confusion_matrix_rejects_mismatched_lengths() {
        let pred = vec![0.0, 1.0];
        let target = vec![0.0];
        let config = PlotConfig::new();

        let result = render_confusion_matrix(&pred, &target, 2, config);
        assert!(result.is_err());
    }
}