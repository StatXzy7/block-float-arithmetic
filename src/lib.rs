//! # qlp4 — blocked low-precision (4-bit) numeric primitives
//!
//! This library quantizes vectors and matrices with an explicit 16-entry code
//! table and lets the exact clipping residuals participate again in dot
//! products and matrix multiplication. It has **no third-party dependencies**
//! and runs on a plain Linux CPU.
//!
//! See [`code_table`] for the code table and nibble-packing rules,
//! [`QTensor::encode`] for block quantization, and [`dot`] / [`matmul`] for the
//! kernels. The serialization layout is documented in the crate README.

pub mod code_table;
mod error;
pub mod linalg;
pub mod tensor;

pub use error::{QError, QResult};
pub use linalg::{dot, error_metrics, matmul, ref_dot, ref_matmul, ErrorMetrics, Matrix};
pub use tensor::{clip_abs, QTensor, SparseEntry};

/// Byte accounting for one serialized [`QTensor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageStats {
    /// Format header bytes (including the optional threshold field).
    pub header_bytes: u64,
    /// The `packed_len` and `residual_count` length fields.
    pub length_bytes: u64,
    /// One little-endian `f64` scale per block.
    pub scale_bytes: u64,
    /// Packed nibble bytes, including any zero-padding nibble.
    pub nibble_bytes: u64,
    /// Sparse residual bytes (`u64` index + `f64` value per entry).
    pub residual_bytes: u64,
    /// Total on-disk size; equals `QTensor::to_bytes().len()`.
    pub total_bytes: u64,
    /// Number of stored non-zero residuals.
    pub residual_count: u64,
}

#[cfg(test)]
mod tests;
