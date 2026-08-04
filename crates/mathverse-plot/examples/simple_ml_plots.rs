//! Simple ML plots example: confusion matrix and ROC curve.

use mathverse_plot::{PlotConfig, render_confusion_matrix, render_roc_curve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Confusion matrix
    let pred = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let target = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];

    let config = PlotConfig::new()
        .with_title("Confusion Matrix")
        .with_x_label("Predicted")
        .with_y_label("Actual");

    let svg = render_confusion_matrix(&pred, &target, 2, config)?;
    std::fs::write("confusion_matrix.svg", &svg)?;
    println!("Wrote confusion_matrix.svg ({} bytes)", svg.len());

    // Example 2: ROC curve
    let scores = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.95, 0.85, 0.6, 0.4];
    let labels = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];

    let config = PlotConfig::new()
        .with_title("ROC Curve")
        .with_x_label("False Positive Rate")
        .with_y_label("True Positive Rate");

    let svg = render_roc_curve(&scores, &labels, config)?;
    std::fs::write("roc_curve.svg", &svg)?;
    println!("Wrote roc_curve.svg ({} bytes)", svg.len());

    Ok(())
}
