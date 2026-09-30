//! Small runnable demonstration: block 4-bit quantization, clipping with
//! sparse residual compensation, dot products, matrix multiplication and the
//! distinction between storage footprint and compute error.
//!
//! Run with: `cargo run --example demo`

use nibble4::{
    dot, metrics, pack_nibbles, reference_dot, unpack_nibbles, Matrix, QMatrix, QVector,
    QuantConfig,
};

fn print_vector_report(name: &str, x: &[f64], q: &QVector) {
    println!("--- {name} ---");
    println!("input       : {x:?}");
    println!(
        "scales      : {:?} (all-zero block => deterministic 0.0)",
        q.scales
    );
    let raw: Vec<u8> = q.code_bytes.clone();
    println!(
        "code bytes  : {raw:02X?}; nibbles: {:02X?}",
        unpack_nibbles(&q.code_bytes, q.len()).unwrap()
    );
    let off = q.decode(false);
    let on = q.decode(true);
    println!("decoded  off: {off:?}");
    println!("decoded  on : {on:?}");
    println!(
        "residuals   : idx {:?} val {:?}",
        q.residual_idx, q.residual_val
    );
    let m_off = metrics(x, &off).unwrap();
    let m_on = metrics(x, &on).unwrap();
    println!(
        "error   off : max={:.6} mse={:.6}",
        m_off.max_abs_error, m_off.mse
    );
    println!(
        "error   on  : max={:.6} mse={:.6}",
        m_on.max_abs_error, m_on.mse
    );
    let st = q.storage_bytes();
    println!(
        "storage     : total={} header={} scales={} codes={} residuals={}",
        st.total, st.header, st.scales, st.codes, st.residuals
    );
    let naive = std::mem::size_of::<f64>() * x.len();
    println!("f64 baseline: {naive} bytes for {n} values", n = x.len());
    println!();
}

fn main() {
    println!("== nibble4 small demonstration ==\n");

    // Nibble packing rule, shown directly.
    let demo_codes = [0x0u8, 0x7, 0x3, 0x1, 0x5];
    let packed = pack_nibbles(&demo_codes);
    println!(
        "pack {demo_codes:02X?} -> {packed:02X?} (low nibble first, tail padded 0)\n"
    );

    let config = QuantConfig::new(4, 4.0).unwrap();

    let x = [5.0, -2.0, 0.5, 0.0, 9.0, -1.0, 3.0, -6.0, 1.5];
    let y = [1.0, 4.0, -2.0, 0.0, 0.5, 8.0, -3.0, 2.0, -1.0];
    let qx = QVector::encode(&x, config).unwrap();
    let qy = QVector::encode(&y, config).unwrap();
    print_vector_report("x", &x, &qx);
    print_vector_report("y", &y, &qy);

    let reference = reference_dot(&x, &y).unwrap();
    let off = dot(&qx, &qy, false).unwrap();
    let on = dot(&qx, &qy, true).unwrap();
    println!("--- dot product (same inputs/params, compensation toggled) ---");
    println!("reference f64: {reference}");
    println!("quantized off: {off:?}  err {:.3e}", (off - reference).abs());
    println!("compensated on: {on:?} err {:.3e}\n", (on - reference).abs());

    // Matrix multiplication vs the pure-f64 independent reference.
    let a = Matrix::new(2, 3, vec![8.0, -1.0, 0.0, 2.0, 5.0, -7.0]).unwrap();
    let b = Matrix::new(3, 2, vec![1.5, 0.0, -4.0, 3.0, 0.5, -2.0]).unwrap();
    let qa = QMatrix::encode(&a, config).unwrap();
    let qb = QMatrix::encode(&b, config).unwrap();
    let refm = Matrix::reference_mul(&a, &b).unwrap();
    let moff = qa.mul(&qb, false).unwrap();
    let mon = qa.mul(&qb, true).unwrap();
    println!("--- matrix multiply 2x3 * 3x2 ---");
    println!("reference : {:?}", refm.data);
    println!("quant off : {:?}", moff.data);
    println!("compensate: {:?}", mon.data);
    let em = metrics(&refm.data, &mon.data).unwrap();
    println!("compensated entrywise max error: {:.3e}", em.max_abs_error);
}
