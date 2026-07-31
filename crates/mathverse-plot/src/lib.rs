//! Plotting with SVG, HTML, and terminal output backends
//! 
//! This crate provides:
//! - SVG plotting backend
//! - HTML plotting backend
//! - Terminal (ASCII) plotting backend
//! - Common plot types (line, scatter, bar, histogram)

pub mod svg;
pub mod html;
pub mod terminal;
pub mod common;
pub mod style;

pub use svg::*;
pub use html::*;
pub use terminal::*;
pub use common::*;
pub use style::*;
