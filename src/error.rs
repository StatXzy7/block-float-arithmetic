//! Error types for qlp4.

use std::fmt;

/// All fallible operations in this crate return [`QError`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QError {
    /// An input element or scale was NaN or infinite.
    NonFinite,
    /// `block_len` was zero.
    InvalidBlockLen,
    /// A clipping threshold was NaN, infinite, or negative.
    InvalidThreshold,
    /// A matrix shape was zero or the flat data length did not match it.
    ShapeMismatch,
    /// A serialized buffer was shorter than expected.
    Truncated,
    /// A serialized buffer contained trailing bytes.
    TrailingData,
    /// A magic number or version byte was unknown.
    BadHeader,
    /// A decoded scale was NaN, infinite or negative.
    InvalidScale,
    /// A packed nibble byte had a non-zero padding (high) nibble.
    InvalidPadding,
    /// A zero-scale block contained non-zero codes, or a non-zero-scale
    /// block had all-zero codes (broken block/scale invariant).
    CorruptBlock,
    /// Sparse residuals were malformed: index out of range, non-increasing
    /// indices, a non-finite value, or a residual below the expected minimum.
    CorruptResiduals,
}

impl fmt::Display for QError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            QError::NonFinite => "non-finite value encountered",
            QError::InvalidBlockLen => "block length must be non-zero",
            QError::InvalidThreshold => "threshold must be finite and non-negative",
            QError::ShapeMismatch => "matrix shape conflict",
            QError::Truncated => "truncated serialized data",
            QError::TrailingData => "trailing bytes in serialized data",
            QError::BadHeader => "unknown magic or format version",
            QError::InvalidScale => "illegal block scale",
            QError::InvalidPadding => "non-zero padding nibble",
            QError::CorruptBlock => "block codes inconsistent with scale",
            QError::CorruptResiduals => "malformed sparse residuals",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for QError {}

/// Crate result alias.
pub type QResult<T> = Result<T, QError>;
