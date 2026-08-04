//! Annotation system for text, arrows, and shapes at data coordinates.

use crate::common::DataPoint;
use crate::style::Color;

/// An arrow pointing from one data point to another.
#[derive(Debug, Clone)]
pub struct Arrow {
    /// Start position in data coordinates.
    pub from: DataPoint,
    /// End position in data coordinates.
    pub to: DataPoint,
    /// Arrow color.
    pub color: Color,
    /// Stroke width.
    pub width: f64,
    /// Arrowhead size (pixels).
    pub head_size: f64,
    /// Dash pattern (None = solid).
    pub dash: Option<String>,
}

impl Arrow {
    /// Create a new arrow.
    pub fn new(from: DataPoint, to: DataPoint) -> Self {
        Self {
            from,
            to,
            color: Color::BLACK,
            width: 1.5,
            head_size: 8.0,
            dash: None,
        }
    }

    /// Set the color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the stroke width.
    pub fn with_width(mut self, width: f64) -> Self {
        self.width = width;
        self
    }

    /// Set the arrowhead size.
    pub fn with_head_size(mut self, size: f64) -> Self {
        self.head_size = size;
        self
    }

    /// Set the dash pattern.
    pub fn with_dash(mut self, dash: impl Into<String>) -> Self {
        self.dash = Some(dash.into());
        self
    }
}

/// A text annotation at a data coordinate.
#[derive(Debug, Clone)]
pub struct TextAnnotation {
    /// Position in data coordinates.
    pub position: DataPoint,
    /// Text content.
    pub text: String,
    /// Font size.
    pub font_size: f64,
    /// Font weight (normal, bold).
    pub font_weight: String,
    /// Text color.
    pub color: Color,
    /// Background color (None = transparent).
    pub background: Option<Color>,
    /// Horizontal offset from the position (pixels).
    pub x_offset: f64,
    /// Vertical offset from the position (pixels).
    pub y_offset: f64,
    /// Text anchor (start, middle, end).
    pub anchor: TextAnchor,
}

impl TextAnnotation {
    /// Create a new text annotation.
    pub fn new(position: DataPoint, text: impl Into<String>) -> Self {
        Self {
            position,
            text: text.into(),
            font_size: 12.0,
            font_weight: "normal".to_string(),
            color: Color::BLACK,
            background: None,
            x_offset: 0.0,
            y_offset: 0.0,
            anchor: TextAnchor::Start,
        }
    }

    /// Set the font size.
    pub fn with_font_size(mut self, size: f64) -> Self {
        self.font_size = size;
        self
    }

    /// Set the font weight.
    pub fn with_bold(mut self) -> Self {
        self.font_weight = "bold".to_string();
        self
    }

    /// Set the text color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the background color.
    pub fn with_background(mut self, color: Color) -> Self {
        self.background = Some(color);
        self
    }

    /// Set the offset from the position.
    pub fn with_offset(mut self, x: f64, y: f64) -> Self {
        self.x_offset = x;
        self.y_offset = y;
        self
    }

    /// Set the text anchor.
    pub fn with_anchor(mut self, anchor: TextAnchor) -> Self {
        self.anchor = anchor;
        self
    }
}

/// Text anchor alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAnchor {
    Start,
    Middle,
    End,
}

/// A rectangle shape in data coordinates.
#[derive(Debug, Clone)]
pub struct Rectangle {
    /// Top-left corner in data coordinates.
    pub origin: DataPoint,
    /// Width in data units.
    pub width: f64,
    /// Height in data units.
    pub height: f64,
    /// Fill color (None = transparent).
    pub fill: Option<Color>,
    /// Stroke color.
    pub stroke: Color,
    /// Stroke width.
    pub stroke_width: f64,
    /// Corner radius for rounded corners.
    pub rx: f64,
}

impl Rectangle {
    /// Create a new rectangle.
    pub fn new(origin: DataPoint, width: f64, height: f64) -> Self {
        Self {
            origin,
            width,
            height,
            fill: None,
            stroke: Color::BLACK,
            stroke_width: 1.0,
            rx: 0.0,
        }
    }

    /// Set the fill color.
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set the stroke color.
    pub fn with_stroke(mut self, color: Color) -> Self {
        self.stroke = color;
        self
    }

    /// Set rounded corners.
    pub fn with_rounded_corners(mut self, radius: f64) -> Self {
        self.rx = radius;
        self
    }
}

/// A horizontal or vertical line across the plot.
#[derive(Debug, Clone)]
pub struct ReferenceLine {
    /// Orientation.
    pub orientation: LineOrientation,
    /// Position in data coordinates (x for horizontal, y for vertical).
    pub position: f64,
    /// Line color.
    pub color: Color,
    /// Stroke width.
    pub width: f64,
    /// Dash pattern.
    pub dash: Option<String>,
    /// Optional label.
    pub label: Option<String>,
}

/// Orientation for reference lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineOrientation {
    Horizontal,
    Vertical,
}

impl ReferenceLine {
    /// Create a horizontal reference line.
    pub fn horizontal(position: f64) -> Self {
        Self {
            orientation: LineOrientation::Horizontal,
            position,
            color: Color::GRAY,
            width: 1.0,
            dash: Some("5,5".to_string()),
            label: None,
        }
    }

    /// Create a vertical reference line.
    pub fn vertical(position: f64) -> Self {
        Self {
            orientation: LineOrientation::Vertical,
            position,
            color: Color::GRAY,
            width: 1.0,
            dash: Some("5,5".to_string()),
            label: None,
        }
    }

    /// Set the color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the width.
    pub fn with_width(mut self, width: f64) -> Self {
        self.width = width;
        self
    }

    /// Set the dash pattern.
    pub fn with_dash(mut self, dash: impl Into<String>) -> Self {
        self.dash = Some(dash.into());
        self
    }

    /// Set the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A collection of annotations.
#[derive(Debug, Clone, Default)]
pub struct Annotations {
    /// Text annotations.
    pub texts: Vec<TextAnnotation>,
    /// Arrows.
    pub arrows: Vec<Arrow>,
    /// Rectangles.
    pub rectangles: Vec<Rectangle>,
    /// Reference lines.
    pub lines: Vec<ReferenceLine>,
}

impl Annotations {
    /// Create a new empty annotations collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a text annotation.
    pub fn add_text(mut self, text: TextAnnotation) -> Self {
        self.texts.push(text);
        self
    }

    /// Add an arrow.
    pub fn add_arrow(mut self, arrow: Arrow) -> Self {
        self.arrows.push(arrow);
        self
    }

    /// Add a rectangle.
    pub fn add_rectangle(mut self, rect: Rectangle) -> Self {
        self.rectangles.push(rect);
        self
    }

    /// Add a reference line.
    pub fn add_line(mut self, line: ReferenceLine) -> Self {
        self.lines.push(line);
        self
    }

    /// Check if there are any annotations.
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
            && self.arrows.is_empty()
            && self.rectangles.is_empty()
            && self.lines.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_basic() {
        let arrow = Arrow::new(DataPoint::new(0.0, 0.0), DataPoint::new(1.0, 1.0))
            .with_color(Color::RED)
            .with_width(2.0);
        assert_eq!(arrow.from.x, 0.0);
        assert_eq!(arrow.to.x, 1.0);
    }

    #[test]
    fn text_annotation() {
        let text = TextAnnotation::new(DataPoint::new(5.0, 10.0), "Hello")
            .with_bold()
            .with_font_size(14.0);
        assert_eq!(text.text, "Hello");
        assert_eq!(text.font_weight, "bold");
    }

    #[test]
    fn reference_line() {
        let line = ReferenceLine::horizontal(5.0)
            .with_color(Color::RED)
            .with_label("Mean");
        assert_eq!(line.position, 5.0);
        assert_eq!(line.label, Some("Mean".to_string()));
    }

    #[test]
    fn annotations_collection() {
        let ann = Annotations::new()
            .add_text(TextAnnotation::new(DataPoint::new(0.0, 0.0), "A"))
            .add_arrow(Arrow::new(DataPoint::new(0.0, 0.0), DataPoint::new(1.0, 1.0)))
            .add_line(ReferenceLine::horizontal(5.0));
        assert!(!ann.is_empty());
    }
}
