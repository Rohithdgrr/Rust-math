//! # mathverse-views
//!
//! Zero-copy borrowed views for MathVerse vectors and matrices.
//!
//! `VecView` and `MatView` borrow data from existing containers without
//! cloning. This enables subvector, submatrix, row, column, and diagonal
//! extraction as lightweight view objects.
//!
//! # Examples
//!
//! ```
//! use mathverse_views::{VecView, MatView};
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let view = VecView::new(&data);
//! assert_eq!(view.len(), 6);
//!
//! // Subvector view
//! let sub = view.slice(1..4);
//! assert_eq!(sub[0], 2.0);
//! assert_eq!(sub[2], 4.0);
//!
//! // Matrix view over row-major data
//! let mat = MatView::new(&data, 2, 3);
//! assert_eq!(mat.get(0, 2), 3.0);
//! assert_eq!(mat.get(1, 0), 4.0);
//! ```

extern crate alloc;

pub mod vec_view;
pub mod mat_view;

pub use vec_view::VecView;
pub use mat_view::MatView;
