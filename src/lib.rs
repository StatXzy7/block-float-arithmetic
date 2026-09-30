//! # nibble4
//!
//! Block-wise 4-bit (half-byte) numerics for studying how a four-bit
//! representation affects dot products and matrix multiplication on a plain
//! Linux CPU.
//!
//! ## The explicit 16-entry code table
//!
//! A nibble is `[sign | m2 m1 m0]`. The low three bits index an unsigned
//! magnitude table:
//!
//! | index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
//! |-------|---|---|---|---|---|---|---|---|
//! | value | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
//!
//! The high bit is the sign: codes `0x1..=0x7` are positive, codes
//! `0x9..=0xF` are the corresponding negatives. Code `0x0` is positive zero
//! and code `0x8` is the readable negative zero: decoding `0x8` yields `-0.0`,
//! but the encoder never emits it (it writes `-0.0` as `0x0`).
//!
//! ## Block quantization
//!
//! Values are split into blocks of configurable length. Each block is scaled
//! so that its largest absolute magnitude maps to code magnitude `6`
//! (index 7): `scale = max_abs / 6`. Rounding picks the nearest code value; on
//! an exact tie the entry with the smaller absolute value is chosen
//! (round-half-toward-zero). An all-zero block has a deterministic scale of
//! `0.0` and all-zero codes.
//!
//! ## Nibble packing
//!
//! Two codes share a byte. [`LOW_NIBBLE_FIRST`] documents the nibble order:
//! element `i` occupies the low nibble when `i` is even (relative to the
//! vector start), otherwise the high nibble. A trailing odd code is padded
//! with `0x0` in the high nibble, recorded in [`ODD_TAIL_PAD`].
//!
//! ## Clipping and sparse compensation
//!
//! An absolute-value threshold clips inputs before quantization. The residual
//! `r = x - clipped(x)` is stored sparsely (index/value pairs) only for
//! clipped elements. The residual is a real operand in dot products and matrix
//! multiplication, not merely an exported field: use `use_compensation = true`
//! to add it back and `false` to ignore it. Both modes share the exact same
//! encoded input and parameters, so comparisons isolate compensation's effect.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// Magnitude table indexed by the low three bits of a nibble.
pub const MAGNITUDES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Nibble order: element `i` goes in the low nibble iff `i % 2 == 0`.
pub const LOW_NIBBLE_FIRST: bool = true;

/// Padding value placed in the unused high nibble of an odd-length tail.
pub const ODD_TAIL_PAD: u8 = 0x0;

/// Magic prefix of the serialized byte container.
pub const MAGIC: &[u8; 4] = b"QNP1";

const HEADER_LEN: usize = 21; // magic(4) + n(4) + block_len(4) + threshold(8) + flags(1)

/// Bit in the flags byte: residual pairs are present.
const FLAG_COMPENSATED: u8 = 0x01;

/// All errors produced by the library. On failure no partial decoded result is
/// ever returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QError {
    /// An input scalar was NaN or infinite.
    NonFinite,
    /// `block_len` was zero (or did not fit in `u32`).
    BadBlockLen,
    /// The threshold was NaN or negative.
    BadThreshold,
    /// A serialized buffer ended before its declared layout.
    Truncated,
    /// A serialized buffer had an unknown magic or flags.
    Corrupted,
    /// A decoded scale was NaN, negative, infinite or inconsistent.
    BadScale,
    /// Matrix/vector shapes were incompatible for the operation.
    ShapeMismatch,
}

impl fmt::Display for QError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            QError::NonFinite => "non-finite value in input",
            QError::BadBlockLen => "block length must be a non-zero u32",
            QError::BadThreshold => "threshold must be finite and non-negative",
            QError::Truncated => "buffer is truncated",
            QError::Corrupted => "buffer is corrupted",
            QError::BadScale => "scale is illegal or inconsistent",
            QError::ShapeMismatch => "matrix or vector shape mismatch",
        };
        f.write_str(msg)
    }
}

impl Error for QError {}

/// Result alias used throughout the library.
pub type QResult<T> = Result<T, QError>;

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Decode one nibble to its (possibly signed zero) code value.
///
/// `0x8` decodes to `-0.0`; every other nibble reads as its table entry.
pub fn code_to_value(code: u8) -> f64 {
    let code = code & 0x0F;
    let mag = MAGNITUDES[(code & 0x07) as usize];
    if code & 0x08 != 0 {
        -mag
    } else {
        mag
    }
}

/// Find the magnitude-table index closest to `x >= 0`.
///
/// On an exact tie the smaller index (smaller magnitude) wins. Values at or
/// above `6` map to index 7.
fn nearest_index(x: f64) -> usize {
    let mut best = 0usize;
    let mut best_dist = (x - MAGNITUDES[0]).abs();
    for idx in 1..8 {
        let dist = (x - MAGNITUDES[idx]).abs();
        // Strict comparison preserves the first (smaller) index on a tie.
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }
    best
}

/// Quantize one finite scalar with a known non-negative block scale.
///
/// Returns `(code, dequantized)` where `dequantized = sign * magnitude * scale`.
/// Negative zero and exact zero encode as `0x0`.
fn quantize_scalar(x: f64, scale: f64) -> (u8, f64) {
    if x == 0.0 {
        return (0x0, 0.0);
    }
    let sign = if x < 0.0 { 0x08 } else { 0x00 };
    let norm = if scale > 0.0 { (x.abs() / scale).min(6.0) } else { 6.0 };
    let idx = nearest_index(norm);
    let code = sign | (idx as u8);
    (code, code_to_value(code) * scale)
}

/// Clip `x` to the symmetric absolute threshold.
fn clip_scalar(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        threshold
    } else if x < -threshold {
        -threshold
    } else {
        x
    }
}

/// Pack a slice of nibbles (low four bits of each byte) into bytes.
///
/// Even indices occupy the low nibble ([`LOW_NIBBLE_FIRST`]); an odd final
/// element is padded with [`ODD_TAIL_PAD`] in the high nibble.
pub fn pack_nibbles(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(div_ceil(codes.len(), 2));
    let mut chunks = codes.chunks_exact(2);
    for pair in &mut chunks {
        out.push((pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4));
    }
    if let Some(tail) = chunks.remainder().first() {
        out.push((tail & 0x0F) | (ODD_TAIL_PAD << 4));
    }
    out
}

/// Unpack nibble bytes back into one nibble per byte.
///
/// `count` is the number of real elements. The high nibble of the final byte
/// must equal [`ODD_TAIL_PAD`] when `count` is odd; otherwise the data is
/// rejected as corrupt.
pub fn unpack_nibbles(bytes: &[u8], count: usize) -> QResult<Vec<u8>> {
    let need = div_ceil(count, 2);
    if bytes.len() != need {
        return Err(QError::Corrupted);
    }
    let mut out = Vec::with_capacity(count);
    for (i, &byte) in bytes.iter().enumerate() {
        out.push(byte & 0x0F);
        if 2 * i + 1 < count {
            out.push((byte >> 4) & 0x0F);
        } else if (byte >> 4) & 0x0F != ODD_TAIL_PAD {
            return Err(QError::Corrupted);
        }
    }
    Ok(out)
}

/// Quantization configuration: block length and absolute clipping threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantConfig {
    /// Number of consecutive values sharing one scale (must be non-zero).
    pub block_len: u32,
    /// Absolute clipping threshold; `f64::INFINITY` disables clipping.
    pub threshold: f64,
}

impl QuantConfig {
    /// Construct a configuration, rejecting illegal parameters eagerly.
    pub fn new(block_len: u32, threshold: f64) -> QResult<Self> {
        if block_len == 0 {
            return Err(QError::BadBlockLen);
        }
        if threshold.is_nan() || threshold < 0.0 {
            return Err(QError::BadThreshold);
        }
        Ok(Self { block_len, threshold })
    }

    /// Configuration without clipping (`threshold = +infinity`).
    pub fn no_clipping(block_len: u32) -> QResult<Self> {
        Self::new(block_len, f64::INFINITY)
    }

    fn is_clipping(&self) -> bool {
        self.threshold.is_finite()
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> QResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(QError::Truncated)?;
        if end > self.buf.len() {
            return Err(QError::Truncated);
        }
        let part = &self.buf[self.pos..end];
        self.pos = end;
        Ok(part)
    }

    fn u32_le(&mut self) -> QResult<u32> {
        let p = self.take(4)?;
        Ok(u32::from_le_bytes([p[0], p[1], p[2], p[3]]))
    }

    fn u8(&mut self) -> QResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn f64_le(&mut self) -> QResult<f64> {
        let p = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(p);
        Ok(f64::from_le_bytes(a))
    }
}

/// A block-wise 4-bit encoded vector, optionally carrying sparse residuals.
#[derive(Debug, Clone, PartialEq)]
pub struct QVector {
    /// Number of encoded values.
    pub n: u32,
    /// Encoding configuration (block length and clipping threshold).
    pub config: QuantConfig,
    /// One scale per block; all-zero blocks carry the deterministic value `0`.
    pub scales: Vec<f64>,
    /// Packed nibble bytes (see [`pack_nibbles`]).
    pub code_bytes: Vec<u8>,
    /// Sorted indices of clipped elements.
    pub residual_idx: Vec<u32>,
    /// Residuals `x - clipped(x)` aligned with [`QVector::residual_idx`].
    pub residual_val: Vec<f64>,
}

impl QVector {
    /// Encode a finite slice of `f64` values.
    pub fn encode(x: &[f64], config: QuantConfig) -> QResult<Self> {
        if config.block_len == 0 {
            return Err(QError::BadBlockLen);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(QError::NonFinite);
        }
        let n = u32::try_from(x.len()).map_err(|_| QError::BadBlockLen)?;
        let block_len = config.block_len as usize;
        let num_blocks = div_ceil(x.len(), block_len);

        let mut scales = Vec::with_capacity(num_blocks);
        let mut codes = Vec::with_capacity(x.len());
        let mut residual_idx = Vec::new();
        let mut residual_val = Vec::new();

        for (b, block) in x.chunks(block_len).enumerate() {
            let mut max_abs = 0.0f64;
            for (j, &v) in block.iter().enumerate() {
                let c = clip_scalar(v, config.threshold);
                max_abs = max_abs.max(c.abs());
                if config.is_clipping() && v.abs() > config.threshold {
                    residual_idx.push((b * block_len + j) as u32);
                    residual_val.push(v - c);
                }
            }
            let scale = max_abs / 6.0;
            scales.push(scale);
            for &v in block {
                let c = clip_scalar(v, config.threshold);
                let (code, _) = quantize_scalar(c, scale);
                codes.push(code);
            }
        }

        let q = Self {
            n,
            config,
            scales,
            code_bytes: pack_nibbles(&codes),
            residual_idx,
            residual_val,
        };
        q.validate_sizes()?;
        Ok(q)
    }

    /// Number of encoded values.
    pub fn len(&self) -> usize {
        self.n as usize
    }

    /// Whether the vector encodes zero values.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn validate_sizes(&self) -> QResult<()> {
        let n = self.n as usize;
        let block_len = self.config.block_len as usize;
        if self.scales.len() != div_ceil(n, block_len) {
            return Err(QError::Corrupted);
        }
        if self.code_bytes.len() != div_ceil(n, 2) {
            return Err(QError::Corrupted);
        }
        if self.residual_idx.len() != self.residual_val.len() {
            return Err(QError::Corrupted);
        }
        Ok(())
    }

    fn validate(&self) -> QResult<()> {
        self.validate_sizes()?;
        if self.config.block_len == 0 {
            return Err(QError::BadBlockLen);
        }
        if self.config.threshold.is_nan() || self.config.threshold < 0.0 {
            return Err(QError::BadThreshold);
        }
        let n = self.n as usize;
        let block_len = self.config.block_len as usize;

        let codes = unpack_nibbles(&self.code_bytes, n)?;
        for (b, &scale) in self.scales.iter().enumerate() {
            if !scale.is_finite() || scale < 0.0 {
                return Err(QError::BadScale);
            }
            if scale == 0.0 {
                let start = b * block_len;
                for &code in &codes[start..n.min(start + block_len)] {
                    // Zero-scale blocks must be exact zeros; 0x8 is rejected.
                    if code != 0 {
                        return Err(QError::BadScale);
                    }
                }
            }
        }
        let mut prev: Option<u32> = None;
        for (&idx, &r) in self.residual_idx.iter().zip(&self.residual_val) {
            if idx as usize >= n {
                return Err(QError::Corrupted);
            }
            if !r.is_finite() || r == 0.0 {
                return Err(QError::Corrupted);
            }
            if let Some(p) = prev {
                if idx <= p {
                    return Err(QError::Corrupted);
                }
            }
            prev = Some(idx);
        }
        Ok(())
    }

    /// Decode one element, optionally adding the stored residual back.
    pub fn value_at(&self, i: usize, use_compensation: bool) -> f64 {
        let codes = unpack_nibbles(&self.code_bytes, self.len()).expect("validated");
        let block = i / self.config.block_len as usize;
        let mut v = code_to_value(codes[i]) * self.scales[block];
        if use_compensation {
            if let Ok(p) = self.residual_idx.binary_search(&(i as u32)) {
                v += self.residual_val[p];
            }
        }
        v
    }

    /// Decode the whole vector, optionally adding residuals back.
    pub fn decode(&self, use_compensation: bool) -> Vec<f64> {
        let n = self.len();
        let codes = unpack_nibbles(&self.code_bytes, n).expect("validated vector");
        let block_len = self.config.block_len as usize;
        let mut out = Vec::with_capacity(n);
        for (i, &code) in codes.iter().enumerate() {
            let mut v = code_to_value(code) * self.scales[i / block_len];
            if use_compensation {
                if let Ok(p) = self.residual_idx.binary_search(&(i as u32)) {
                    v += self.residual_val[p];
                }
            }
            out.push(v);
        }
        out
    }

    /// Serialize into a self-describing byte container (all-or-nothing parse).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.n.to_le_bytes());
        out.extend_from_slice(&self.config.block_len.to_le_bytes());
        out.extend_from_slice(&self.config.threshold.to_le_bytes());
        let flags = if self.residual_idx.is_empty() { 0 } else { FLAG_COMPENSATED };
        out.push(flags);
        for &s in &self.scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out.extend_from_slice(&self.code_bytes);
        out.extend_from_slice(&(self.residual_idx.len() as u32).to_le_bytes());
        for &idx in &self.residual_idx {
            out.extend_from_slice(&idx.to_le_bytes());
        }
        for &r in &self.residual_val {
            out.extend_from_slice(&r.to_le_bytes());
        }
        out
    }

    /// Parse a byte container, rejecting truncated or corrupted data outright.
    pub fn from_bytes(buf: &[u8]) -> QResult<Self> {
        let mut r = Reader::new(buf);
        let magic = r.take(4)?;
        if magic != MAGIC {
            return Err(QError::Corrupted);
        }
        let n = r.u32_le()?;
        let block_len = r.u32_le()?;
        if block_len == 0 {
            return Err(QError::BadBlockLen);
        }
        let threshold = r.f64_le()?;
        if threshold.is_nan() || threshold < 0.0 {
            return Err(QError::BadThreshold);
        }
        let flags = r.u8()?;
        if flags & !FLAG_COMPENSATED != 0 {
            return Err(QError::Corrupted);
        }
        let config = QuantConfig { block_len, threshold };

        let n_us = n as usize;
        let num_blocks = div_ceil(n_us, block_len as usize);
        let mut scales = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            scales.push(r.f64_le()?);
        }
        let code_len = div_ceil(n_us, 2);
        let code_bytes = r.take(code_len)?.to_vec();
        let residual_count = r.u32_le()? as usize;
        if residual_count > n_us {
            return Err(QError::Corrupted);
        }
        let mut residual_idx = Vec::with_capacity(residual_count);
        for _ in 0..residual_count {
            residual_idx.push(r.u32_le()?);
        }
        let mut residual_val = Vec::with_capacity(residual_count);
        for _ in 0..residual_count {
            residual_val.push(r.f64_le()?);
        }
        if r.pos != buf.len() {
            return Err(QError::Corrupted);
        }

        let q = Self {
            n,
            config,
            scales,
            code_bytes,
            residual_idx,
            residual_val,
        };
        q.validate()?;

        // Residuals only make sense when clipping is enabled and the residual
        // direction/magnitude matches a value that actually crossed threshold.
        if q.residual_idx.is_empty() != ((flags & FLAG_COMPENSATED) == 0) {
            return Err(QError::Corrupted);
        }
        let codes = unpack_nibbles(&q.code_bytes, n_us)?;
        for (&idx, &res) in q.residual_idx.iter().zip(&q.residual_val) {
            if !config.is_clipping() {
                return Err(QError::Corrupted);
            }
            let i = idx as usize;
            let block = i / block_len as usize;
            let y = code_to_value(codes[i]) * q.scales[block];
            let reconstructed = y + res;
            // `6.0 * (threshold / 6.0)` can differ by a rounding ulp, so allow
            // a tolerance vastly larger than an ulp but far below one code step.
            let slack = (6.0 * q.scales[block]).abs() * 1e-9 + 1e-12;
            if !reconstructed.is_finite()
                || reconstructed.abs() + slack <= config.threshold
                || (reconstructed >= 0.0 && res <= 0.0)
                || (reconstructed <= 0.0 && res >= 0.0)
            {
                return Err(QError::Corrupted);
            }
        }
        Ok(q)
    }
}

/// On-disk/in-memory storage footprint accounting (bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StorageBytes {
    /// Total bytes of the serialized container.
    pub total: usize,
    /// Fixed header (magic, length, block length, threshold, flags).
    pub header: usize,
    /// Per-block `f64` scales.
    pub scales: usize,
    /// Packed nibble payload.
    pub codes: usize,
    /// Residual section: count `u32`, per pair `u32 + f64`.
    pub residuals: usize,
}

impl QVector {
    /// Storage breakdown of the serialized form.
    pub fn storage_bytes(&self) -> StorageBytes {
        let residual_pairs = self.residual_idx.len();
        StorageBytes {
            total: HEADER_LEN
                + self.scales.len() * 8
                + self.code_bytes.len()
                + 4
                + residual_pairs * (4 + 8),
            header: HEADER_LEN,
            scales: self.scales.len() * 8,
            codes: self.code_bytes.len(),
            residuals: if residual_pairs == 0 { 4 } else { 4 + residual_pairs * (4 + 8) },
        }
    }
}

fn dot_codes(a: &QVector, b: &QVector, use_compensation: bool) -> f64 {
    let n = a.len();
    let ca = unpack_nibbles(&a.code_bytes, n).expect("validated");
    let cb = unpack_nibbles(&b.code_bytes, n).expect("validated");
    let ba = a.config.block_len as usize;
    let bb = b.config.block_len as usize;
    let mut acc = 0.0f64;
    for i in 0..n {
        let va = code_to_value(ca[i]) * a.scales[i / ba];
        let vb = code_to_value(cb[i]) * b.scales[i / bb];
        let ra = if use_compensation {
            a.residual_idx
                .binary_search(&(i as u32))
                .map(|p| a.residual_val[p])
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let rb = if use_compensation {
            b.residual_idx
                .binary_search(&(i as u32))
                .map(|p| b.residual_val[p])
                .unwrap_or(0.0)
        } else {
            0.0
        };
        acc += (va + ra) * (vb + rb);
    }
    acc
}

/// Dot product of two encoded vectors (equal length required).
///
/// `use_compensation = true` adds stored residuals to both operands; `false`
/// quantizes-only. Both calls use the identical encoded input and parameters.
pub fn dot(a: &QVector, b: &QVector, use_compensation: bool) -> QResult<f64> {
    if a.n != b.n {
        return Err(QError::ShapeMismatch);
    }
    a.validate()?;
    b.validate()?;
    Ok(dot_codes(a, b, use_compensation))
}

/// Pure `f64` reference dot product (rejects non-finite operands).
pub fn reference_dot(x: &[f64], y: &[f64]) -> QResult<f64> {
    if x.len() != y.len() {
        return Err(QError::ShapeMismatch);
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(QError::NonFinite);
    }
    Ok(x.iter().zip(y).map(|(a, b)| a * b).sum())
}

/// Error summary of an approximation against a pure-`f64` reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorMetrics {
    /// Maximum absolute difference `max|approx - reference|`.
    pub max_abs_error: f64,
    /// Mean squared difference over all elements.
    pub mse: f64,
}

/// Compare an approximation slice against a reference slice.
pub fn metrics(reference: &[f64], approx: &[f64]) -> QResult<ErrorMetrics> {
    if reference.len() != approx.len() {
        return Err(QError::ShapeMismatch);
    }
    if reference.iter().chain(approx).any(|v| !v.is_finite()) {
        return Err(QError::NonFinite);
    }
    let mut max_abs = 0.0f64;
    let mut sq = 0.0f64;
    for (r, a) in reference.iter().zip(approx) {
        let d = (a - r).abs();
        max_abs = max_abs.max(d);
        sq += d * d;
    }
    Ok(ErrorMetrics {
        max_abs_error: max_abs,
        mse: sq / reference.len().max(1) as f64,
    })
}

/// Row-major dense matrix of pure `f64` values.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// Row-major data of length `rows * cols`.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a matrix, rejecting inconsistent data length and non-finite values.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> QResult<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(QError::ShapeMismatch);
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(QError::NonFinite);
        }
        Ok(Self { rows, cols, data })
    }

    /// Reference entry `C_ij = sum_k A_ik B_kj` in pure `f64`.
    pub fn reference_mul(a: &Matrix, b: &Matrix) -> QResult<Matrix> {
        if a.cols != b.rows {
            return Err(QError::ShapeMismatch);
        }
        let mut data = vec![0.0f64; a.rows * b.cols];
        for i in 0..a.rows {
            for k in 0..a.cols {
                let aik = a.data[i * a.cols + k];
                for j in 0..b.cols {
                    data[i * b.cols + j] += aik * b.data[k * b.cols + j];
                }
            }
        }
        Matrix::new(a.rows, b.cols, data)
    }
}

/// A matrix encoded as one [`QVector`] over its row-major flattened data.
#[derive(Debug, Clone, PartialEq)]
pub struct QMatrix {
    rows: usize,
    cols: usize,
    inner: QVector,
}

impl QMatrix {
    /// Quantize a dense matrix's flattened values.
    pub fn encode(m: &Matrix, config: QuantConfig) -> QResult<Self> {
        let inner = QVector::encode(&m.data, config)?;
        Ok(Self {
            rows: m.rows,
            cols: m.cols,
            inner,
        })
    }

    /// Row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Column count.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Borrow the underlying encoded vector.
    pub fn as_vector(&self) -> &QVector {
        &self.inner
    }

    /// Decode the matrix, optionally adding residuals back.
    pub fn decode(&self, use_compensation: bool) -> Matrix {
        Matrix::new(self.rows, self.cols, self.inner.decode(use_compensation))
            .expect("validated encoded matrix")
    }

    /// Matrix multiplication; shapes must agree like [`Matrix::reference_mul`].
    pub fn mul(&self, other: &QMatrix, use_compensation: bool) -> QResult<Matrix> {
        if self.cols != other.rows {
            return Err(QError::ShapeMismatch);
        }
        let a = &self.inner;
        let b = &other.inner;
        let ca = unpack_nibbles(&a.code_bytes, a.len())?;
        let cb = unpack_nibbles(&b.code_bytes, b.len())?;
        let ba = a.config.block_len as usize;
        let bb = b.config.block_len as usize;
        let inner_cols = self.cols;
        let out_cols = other.cols;
        let mut data = vec![0.0f64; self.rows * out_cols];
        for i in 0..self.rows {
            for k in 0..inner_cols {
                let pa = i * inner_cols + k;
                let va = code_to_value(ca[pa]) * a.scales[pa / ba]
                    + residual_at(a, pa, use_compensation);
                for j in 0..out_cols {
                    let pb = k * out_cols + j;
                    let vb = code_to_value(cb[pb]) * b.scales[pb / bb]
                        + residual_at(b, pb, use_compensation);
                    data[i * out_cols + j] += va * vb;
                }
            }
        }
        Matrix::new(self.rows, out_cols, data)
    }
}

fn residual_at(q: &QVector, i: usize, use_compensation: bool) -> f64 {
    if use_compensation {
        q.residual_idx
            .binary_search(&(i as u32))
            .map(|p| q.residual_val[p])
            .unwrap_or(0.0)
    } else {
        0.0
    }
}
