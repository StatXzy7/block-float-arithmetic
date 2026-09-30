//! Pure-double reference kernels and 4-bit kernels (optionally compensated by
//! the exact clipping residuals).

use crate::error::{QError, QResult};
use crate::tensor::QTensor;

/// Maximum absolute error and mean squared error between two vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorMetrics {
    pub max_abs: f64,
    pub mse: f64,
}

/// Compares an approximate vector against a pure-double reference.
pub fn error_metrics(reference: &[f64], approx: &[f64]) -> QResult<ErrorMetrics> {
    if reference.len() != approx.len() {
        return Err(QError::ShapeMismatch);
    }
    let (mut sum_sq, mut max_abs) = (0.0f64, 0.0f64);
    for (r, a) in reference.iter().zip(approx) {
        let e = (r - a).abs();
        if e > max_abs {
            max_abs = e;
        }
        sum_sq += e * e;
    }
    let mse = if reference.is_empty() {
        0.0
    } else {
        sum_sq / reference.len() as f64
    };
    Ok(ErrorMetrics { max_abs, mse })
}

/// Pure-double reference dot product.
pub fn ref_dot(a: &[f64], b: &[f64]) -> QResult<f64> {
    if a.len() != b.len() {
        return Err(QError::ShapeMismatch);
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

/// Pure-double reference matrix multiplication (`C = A * B`), row-major.
pub fn ref_matmul(
    a_rows: usize,
    a_cols: usize,
    a: &[f64],
    b_rows: usize,
    b_cols: usize,
    b: &[f64],
) -> QResult<Vec<f64>> {
    if a_rows == 0 || a_cols == 0 || b_rows == 0 || b_cols == 0 {
        return Err(QError::ShapeMismatch);
    }
    if a.len() != a_rows * a_cols || b.len() != b_rows * b_cols || a_cols != b_rows {
        return Err(QError::ShapeMismatch);
    }
    let mut c = vec![0.0f64; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut acc = 0.0f64;
            for k in 0..a_cols {
                acc += a[i * a_cols + k] * b[k * b_cols + j];
            }
            c[i * b_cols + j] = acc;
        }
    }
    Ok(c)
}

/// Dot product of two quantized vectors.
///
/// With `compensate == false` only decoded 4-bit codes participate. With
/// `compensate == true` the exact stored residuals of both operands are added
/// in, so the result equals a direct pure-double dot product of the originals.
pub fn dot(a: &QTensor, b: &QTensor, compensate: bool) -> QResult<f64> {
    if a.len() != b.len() {
        return Err(QError::ShapeMismatch);
    }
    let da = a.dequant();
    let db = b.dequant();
    let mut acc: f64 = da.iter().zip(&db).map(|(x, y)| x * y).sum();

    if compensate {
        let ra = a.residuals();
        let rb = b.residuals();
        let (mut ia, mut ib) = (0usize, 0usize);
        while ia < ra.len() || ib < rb.len() {
            let ka = ra.get(ia).map(|&(k, _)| k);
            let kb = rb.get(ib).map(|&(k, _)| k);
            match (ka, kb) {
                (Some(i), Some(j)) if i == j => {
                    let (_, rai) = ra[ia];
                    let (_, rbj) = rb[ib];
                    acc += rai * db[i as usize]
                        + da[j as usize] * rbj
                        + rai * rbj;
                    ia += 1;
                    ib += 1;
                }
                (Some(i), _) if kb.map_or(true, |j| i < j) => {
                    acc += ra[ia].1 * db[i as usize];
                    ia += 1;
                }
                (_, Some(j)) => {
                    acc += da[j as usize] * rb[ib].1;
                    ib += 1;
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(acc)
}

/// A row-major matrix stored as independently quantized rows *and* columns.
///
/// Quantizing rows lets them act as the left operand of a multiplication and
/// quantizing columns lets the same matrix act as the right operand; every
/// block is independently scaled in both orientations.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    row_tensors: Vec<QTensor>,
    col_tensors: Vec<QTensor>,
}

impl Matrix {
    /// Quantizes a row-major `rows * cols` matrix.
    pub fn quantize(
        rows: usize,
        cols: usize,
        data: &[f64],
        block_len: u64,
        threshold: Option<f64>,
    ) -> QResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(QError::ShapeMismatch);
        }
        if data.len() != rows * cols {
            return Err(QError::ShapeMismatch);
        }
        let mut row_tensors = Vec::with_capacity(rows);
        for i in 0..rows {
            row_tensors.push(QTensor::encode(&data[i * cols..(i + 1) * cols], block_len, threshold)?);
        }
        let mut col_tensors = Vec::with_capacity(cols);
        for j in 0..cols {
            let col: Vec<f64> = (0..rows).map(|i| data[i * cols + j]).collect();
            col_tensors.push(QTensor::encode(&col, block_len, threshold)?);
        }
        Ok(Self {
            rows,
            cols,
            row_tensors,
            col_tensors,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row_tensor(&self, i: usize) -> &QTensor {
        &self.row_tensors[i]
    }

    pub fn col_tensor(&self, j: usize) -> &QTensor {
        &self.col_tensors[j]
    }

    /// Actual bytes used by both row and column encodings.
    pub fn storage_total(&self) -> u64 {
        self.row_tensors
            .iter()
            .chain(&self.col_tensors)
            .map(|t| t.storage().total_bytes)
            .sum()
    }
}

/// Matrix multiplication. Shape conflicts return [`QError::ShapeMismatch`].
pub fn matmul(a: &Matrix, b: &Matrix, compensate: bool) -> QResult<Vec<f64>> {
    if a.cols != b.rows {
        return Err(QError::ShapeMismatch);
    }
    let mut c = vec![0.0f64; a.rows * b.cols];
    for i in 0..a.rows {
        for j in 0..b.cols {
            c[i * b.cols + j] = dot(&a.row_tensors[i], &b.col_tensors[j], compensate)?;
        }
    }
    Ok(c)
}
