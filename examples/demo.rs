//! Small end-to-end demonstration:
//!
//!     cargo run --example demo
//!
//! It quantizes vectors and a matrix with the explicit 4-bit code table,
//! clips outliers with stored residuals, and compares kernels with
//! compensation off vs on under identical inputs and parameters.

use qlp4::{dot, error_metrics, matmul, ref_dot, ref_matmul, Matrix, QTensor};

fn main() {
    let block_len = 4u64;
    let threshold = Some(6.0f64);

    println!("== blocked 4-bit dot product (block_len={block_len}, threshold={threshold:?}) ==");
    let a = [9.0, 2.0, -3.0, 4.0, 0.5, -6.0, 6.0, -8.0];
    let b = [1.0, -2.0, 3.0, 0.5, 4.0, 0.0, -2.0, 6.0];

    let qa = QTensor::encode(&a, block_len, threshold).unwrap();
    let qb = QTensor::encode(&b, block_len, threshold).unwrap();

    println!("scales(a)      = {:?}", qa.scales());
    println!("residuals(a)   = {:?}", qa.residuals());
    println!("dequant(a)     = {:?}", qa.dequant());
    println!("reconstruct(a) = {:?}", qa.reconstruct());

    let reference = ref_dot(&a, &b).unwrap();
    for compensate in [false, true] {
        let got = dot(&qa, &qb, compensate).unwrap();
        let va = qa.dequant();
        let vb = qb.dequant();
        let err = (got - reference).abs();
        let mse_a = error_metrics(&a, &va).unwrap();
        let mse_b = error_metrics(&b, &vb).unwrap();
        println!(
            "dot compensate={compensate:5}: value={got:8.4} abs_err={err:8.4}  \
             (decode vec err a: max={:.3} mse={:.4}; b: max={:.3} mse={:.4})",
            mse_a.max_abs, mse_a.mse, mse_b.max_abs, mse_b.mse
        );
    }
    println!("reference dot  = {reference:.4}");

    let s = qa.storage();
    println!(
        "storage(a)     = {} bytes total (header {}, scales {}, nibbles {}, residuals {} x 16B)",
        s.total_bytes, s.header_bytes, s.scale_bytes, s.nibble_bytes, s.residual_count
    );

    println!("\n== 3x3 matrix multiplication, compensation off vs on ==");
    let m = [9.0, -2.0, 2.0, 3.0, 7.0, -4.0, 6.0, 0.0, 2.5];
    let ma = Matrix::quantize(3, 3, &m, 2, threshold).unwrap();
    let mb = Matrix::quantize(3, 3, &m, 2, threshold).unwrap();
    let ref_c = ref_matmul(3, 3, &m, 3, 3, &m).unwrap();

    for compensate in [false, true] {
        let c = matmul(&ma, &mb, compensate).unwrap();
        let e = error_metrics(&ref_c, &c).unwrap();
        println!("compensate={compensate:5}: C={c:6.3?}");
        println!("             max_abs_err={:.6} mse={:.6}", e.max_abs, e.mse);
    }
    println!("reference        : C={ref_c:6.3?}");
    println!("matrix storage   = {} bytes (rows + columns)", ma.storage_total() + mb.storage_total());
}
