//! Integration tests exercising only the public API.

use qlp4::{
    dot, error_metrics, matmul, ref_dot, ref_matmul, Matrix, QError, QTensor,
};

/// Same input/parameters, compensation off vs on: storage is identical, only
/// arithmetic accuracy changes.
#[test]
fn compensation_switch_changes_error_not_storage() {
    let x = vec![9.0, 2.0, -3.0, 4.0, 0.5, -6.0];
    let y = vec![1.0, -2.0, 3.0, 0.5, 4.0, 1.0];
    let qx = QTensor::encode(&x, 4, Some(6.0)).unwrap();
    let qy = QTensor::encode(&y, 4, Some(6.0)).unwrap();
    let reference = ref_dot(&x, &y).unwrap();

    let off = dot(&qx, &qy, false).unwrap();
    let on = dot(&qx, &qy, true).unwrap();
    assert_eq!(on, reference);
    assert!(off != reference);
    // Storage does not depend on the runtime compensation flag.
    assert_eq!(qx.storage().total_bytes, qx.to_bytes().len() as u64);
    assert!(qx.storage().residual_count >= 1);
}

#[test]
fn matmul_matches_independent_reference_and_hardcoded_values() {
    // A: 3x4, B: 4x2. Outliers 8 exceed the threshold and survive only via
    // residuals. Block length 4 exercises full blocks in both orientations.
    let a = vec![
        6.0, 2.0, 0.5, 8.0, //
        -4.0, 3.0, 6.0, -1.5, //
        0.0, 1.5, -6.0, 2.0,
    ];
    let b = vec![
        1.0, -2.0, //
        3.0, 0.5, //
        -4.0, 6.0, //
        6.0, -1.0,
    ];
    let ma = Matrix::quantize(3, 4, &a, 4, Some(6.0)).unwrap();
    let mb = Matrix::quantize(4, 2, &b, 4, Some(6.0)).unwrap();

    let reference = ref_matmul(3, 4, &a, 4, 2, &b).unwrap();
    let hardcoded = vec![58.0, -16.0, -28.0, 47.0, 40.5, -37.25];
    assert_eq!(reference, hardcoded);

    let exact = matmul(&ma, &mb, true).unwrap();
    let approx = matmul(&ma, &mb, false).unwrap();
    assert_eq!(exact, reference, "compensated matmul must be exact");

    let exact_err = error_metrics(&reference, &exact).unwrap();
    assert_eq!((exact_err.max_abs, exact_err.mse), (0.0, 0.0));

    let approx_err = error_metrics(&reference, &approx).unwrap();
    assert_eq!(approx_err.max_abs, 12.0); // 2 (residual) * 6 (B[3][0])
    assert!((approx_err.mse - (144.0 + 4.0) / 6.0).abs() < 1e-12);
    // Entries untouched by residuals stay exact.
    assert_eq!(approx[2..], reference[2..]);

    let bytes = ma.storage_total() + mb.storage_total();
    assert!(bytes > 0);
}

#[test]
fn matmul_rejects_shape_conflicts() {
    let a = Matrix::quantize(2, 3, &[1.0; 6], 3, None).unwrap();
    let b = Matrix::quantize(2, 2, &[1.0; 4], 2, None).unwrap();
    assert_eq!(matmul(&a, &b, true), Err(QError::ShapeMismatch));
    assert_eq!(
        Matrix::quantize(2, 3, &[1.0; 5], 3, None),
        Err(QError::ShapeMismatch)
    );
}

#[test]
fn tail_blocks_and_odd_columns_roundtrip_and_multiply() {
    // Non-divisor block length (3) over 7 elements: tail block + odd nibbles.
    let x = vec![0.0, 0.5, -1.5, 4.0, -6.0, 3.0, 2.0];
    let q = QTensor::encode(&x, 3, None).unwrap();
    let bytes = q.to_bytes();
    assert_eq!(QTensor::from_bytes(&bytes).unwrap().reconstruct(), x);

    // 3x3 matrices, block length 2: every block scales to 1 (its maximum is
    // 6) and all values are table entries, so both rows and columns are exact.
    let m = vec![6.0, -3.0, 0.0, 3.0, 6.0, 0.0, -6.0, 0.0, 3.0];
    let left = Matrix::quantize(3, 3, &m, 2, None).unwrap();
    let right = Matrix::quantize(3, 3, &m, 2, None).unwrap();
    let c = matmul(&left, &right, true).unwrap();
    let r = ref_matmul(3, 3, &m, 3, 3, &m).unwrap();
    assert_eq!(c, r);
    assert_eq!(error_metrics(&r, &c).unwrap().max_abs, 0.0);
}
