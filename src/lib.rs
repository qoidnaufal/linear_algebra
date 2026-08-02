pub mod matrix;
pub mod heap;
pub mod vector;
pub mod macros;
pub mod lu;
pub mod woodbury;
pub mod simd;
pub mod sparse;
mod traits;

pub const fn fast_exp_f32(x: f32) -> f32 {
    let i = (12102203.0 * x + 1064866805.0) as i32;
    f32::from_bits(i as u32)
}

pub const fn fast_exp_f64(x: f64) -> f64 {
    let n = (x * std::f64::consts::LOG2_E).round();
    let r = x - n * std::f64::consts::LN_2;

    let r2 = r * r;
    let r4 = r2 * r2;
    let p = (1.0 + r * (1.0 + r2 * (1.0/6.0 + r2 * (1.0/120.0))))
          + r2 * (0.5 + r4 * (1.0/720.0));

    let pow2n = f64::from_bits(((n as i64 + 1023) as u64) << 52);
    pow2n * p
}

pub fn softplus(x: f64, limit: f64) -> f64 {
    if x > limit {
        x
    } else if x < -limit {
        1e-12
    } else {
        x.exp().ln_1p()
    }
}

pub fn sigmoid(x: f64, limit: f64) -> f64 {
    if x > limit {
        1.0
    } else if x < -limit {
        1e-12
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn softplus_sigmoid(x: f64, limit: f64) -> (f64, f64) {
    if x > limit {
        (x, 1.0)
    } else if x < -limit {
        (1e-12, 1e-12)
    } else {
        let e = x.exp();
        (e.ln_1p(), e / (1.0 + e))
    }
}

#[inline]
pub fn safe_exp(x: f64, limit: f64) -> (f64, f64) {
    if x > limit {
        let e = limit.exp();
        (e * (1.0 + (x - limit)), e)
    } else {
        let e = x.exp();
        (e, e)
    }
}
