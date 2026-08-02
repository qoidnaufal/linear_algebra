use core::arch::aarch64::*;

pub trait NeonContainer {
    fn load_slice(src: &[f64]) -> Self;
    fn dup_f64(value: f64) -> Self;
}

impl NeonContainer for float64x2_t {
    #[inline(always)]
    fn load_slice(src: &[f64]) -> Self {
        unsafe { vld1q_f64(src.as_ptr()) }
    }

    #[inline(always)]
    fn dup_f64(value: f64) -> Self {
        unsafe { vdupq_n_f64(value) }
    }
}

impl NeonContainer for float64x2x2_t {
    #[inline(always)]
    fn load_slice(src: &[f64]) -> Self {
        unsafe { vld1q_f64_x2(src.as_ptr()) }
    }

    #[inline(always)]
    fn dup_f64(value: f64) -> Self {
        vdupq_n_f64x4(value)
    }
}

impl NeonContainer for float64x2x3_t {
    #[inline(always)]
    fn load_slice(src: &[f64]) -> Self {
        unsafe { vld1q_f64_x3(src.as_ptr()) }
    }

    #[inline(always)]
    fn dup_f64(value: f64) -> Self {
        vdupq_n_f64x6(value)
    }
}

#[inline(always)]
pub fn vdupq_n_f64x4(s: f64) -> float64x2x2_t {
    unsafe {
        let v = vdupq_n_f64(s);
        float64x2x2_t(v, v)
    }
}

#[inline(always)]
pub fn vdupq_n_f64x6(s: f64) -> float64x2x3_t {
    unsafe {
        let v = vdupq_n_f64(s);
        float64x2x3_t(v, v, v)
    }
}

#[inline(always)]
pub fn vmulq_f64x4(a: float64x2x2_t, b: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vmulq_f64(a.0, b.0), vmulq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vmulq_f64x6(a: float64x2x3_t, b: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vmulq_f64(a.0, b.0), vmulq_f64(a.1, b.1), vmulq_f64(a.2, b.2)) }
}

#[inline(always)]
pub fn vmulq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vmulq_f64(a.0, b), vmulq_f64(a.1, b), vmulq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vmulq_f64x4x2(a: float64x2x2_t, b: float64x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vmulq_f64(a.0, b), vmulq_f64(a.1, b)) }
}

#[inline(always)]
pub fn vdivq_f64x4(a: float64x2x2_t, b: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vdivq_f64(a.0, b.0), vdivq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vdivq_f64x6(a: float64x2x3_t, b: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vdivq_f64(a.0, b.0), vdivq_f64(a.1, b.1), vdivq_f64(a.2, b.2)) }
}

#[inline(always)]
pub fn vdivq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vdivq_f64(a.0, b), vdivq_f64(a.1, b), vdivq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vdivq_f64x2x4(a: float64x2_t, b: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vdivq_f64(a, b.0), vdivq_f64(a, b.1)) }
}

#[inline(always)]
pub fn vdivq_f64x2x6(a: float64x2_t, b: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vdivq_f64(a, b.0), vdivq_f64(a, b.1), vdivq_f64(a, b.2)) }
}

#[inline(always)]
pub fn vsqrtq_f64x4(x: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vsqrtq_f64(x.0), vsqrtq_f64(x.1)) }
}

#[inline(always)]
pub fn vsqrtq_f64x6(x: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vsqrtq_f64(x.0), vsqrtq_f64(x.1), vsqrtq_f64(x.2)) }
}

#[inline(always)]
pub fn vfmaq_f64x4(a: float64x2x2_t, b: float64x2x2_t, c: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vfmaq_f64(a.0, b.0, c.0), vfmaq_f64(a.1, b.1, c.1)) }
}

#[inline(always)]
pub fn vfmaq_f64x6(a: float64x2x3_t, b: float64x2x3_t, c: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vfmaq_f64(a.0, b.0, c.0), vfmaq_f64(a.1, b.1, c.1), vfmaq_f64(a.2, b.2, c.2)) }
}

#[inline(always)]
pub fn vfmsq_f64x4(a: float64x2x2_t, b: float64x2x2_t, c: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vfmsq_f64(a.0, b.0, c.0), vfmsq_f64(a.1, b.1, c.1)) }
}

#[inline(always)]
pub fn vfmsq_f64x6(a: float64x2x3_t, b: float64x2x3_t, c: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vfmsq_f64(a.0, b.0, c.0), vfmsq_f64(a.1, b.1, c.1), vfmsq_f64(a.2, b.2, c.2)) }
}

#[inline(always)]
pub fn vfmsq_f64x2x6(a: float64x2_t, b: float64x2x3_t, c: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vfmsq_f64(a, b.0, c.0), vfmsq_f64(a, b.1, c.1), vfmsq_f64(a, b.2, c.2)) }
}

#[inline(always)]
pub fn vaddq_f64x4(a: float64x2x2_t, b: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vaddq_f64(a.0, b.0), vaddq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vaddq_f64x6(a: float64x2x3_t, b: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vaddq_f64(a.0, b.0), vaddq_f64(a.1, b.1), vaddq_f64(a.2, b.2)) }
}

#[inline(always)]
pub fn vaddq_f64x4x2(a: float64x2x2_t, b: float64x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vaddq_f64(a.0, b), vaddq_f64(a.1, b)) }
}

#[inline(always)]
pub fn vaddq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vaddq_f64(a.0, b), vaddq_f64(a.1, b), vaddq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vsubq_f64x4(a: float64x2x2_t, b: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vsubq_f64(a.0, b.0), vsubq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vsubq_f64x4x2(a: float64x2x2_t, b: float64x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vsubq_f64(a.0, b), vsubq_f64(a.1, b)) }
}

#[inline(always)]
pub fn vsubq_f64x6(a: float64x2x3_t, b: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vsubq_f64(a.0, b.0), vsubq_f64(a.1, b.1), vsubq_f64(a.2, b.2)) }
}

#[inline(always)]
pub fn vsubq_f64x2x6(a: float64x2_t, b: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vsubq_f64(a, b.0), vsubq_f64(a, b.1), vsubq_f64(a, b.2)) }
}

#[inline(always)]
pub fn vsubq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vsubq_f64(a.0, b), vsubq_f64(a.1, b), vsubq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vmaxq_f64x4x4(a: float64x2x2_t, b: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vmaxq_f64(a.0, b.0), vmaxq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vmaxq_f64x4x2(a: float64x2x2_t, b: float64x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vmaxq_f64(a.0, b), vmaxq_f64(a.1, b)) }
}

#[inline(always)]
pub fn vmaxq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vmaxq_f64(a.0, b), vmaxq_f64(a.1, b), vmaxq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vmaxnmq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vmaxnmq_f64(a.0, b), vmaxnmq_f64(a.1, b), vmaxnmq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vminq_f64x4x4(a: float64x2x2_t, b: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vminq_f64(a.0, b.0), vminq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vminq_f64x4x2(a: float64x2x2_t, b: float64x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vminq_f64(a.0, b), vminq_f64(a.1, b)) }
}

#[inline(always)]
pub fn vminq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vminq_f64(a.0, b), vminq_f64(a.1, b), vminq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vminnmq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vminnmq_f64(a.0, b), vminnmq_f64(a.1, b), vminnmq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vclamp_f64x6x2x2(x: float64x2x3_t, min: float64x2_t, max: float64x2_t) -> float64x2x3_t {
    vmaxq_f64x6x2(vminq_f64x6x2(x, max), min)
}

#[inline(always)]
pub fn vclampnmq_f64x6x2x2(x: float64x2x3_t, min: float64x2_t, max: float64x2_t) -> float64x2x3_t {
    vmaxnmq_f64x6x2(vminnmq_f64x6x2(x, max), min)
}

#[inline(always)]
pub fn vabdq_f64x4(a: float64x2x2_t, b: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vabdq_f64(a.0, b.0), vabdq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vabsq_f64x4(x: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vabsq_f64(x.0), vabsq_f64(x.1)) }
}

#[inline(always)]
pub fn vabsq_f64x6(x: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vabsq_f64(x.0), vabsq_f64(x.1), vabsq_f64(x.2)) }
}

#[inline(always)]
pub fn vnegq_f64x4(x: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vnegq_f64(x.0), vnegq_f64(x.1)) }
}

#[inline(always)]
pub fn vnegq_f64x6(x: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vnegq_f64(x.0), vnegq_f64(x.1), vnegq_f64(x.2)) }
}

#[inline(always)]
pub fn vcltq_f64x4(a: float64x2x2_t, b: float64x2x2_t) -> uint64x2x2_t {
    unsafe { uint64x2x2_t(vcltq_f64(a.0, b.0), vcltq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vcltq_f64x6(a: float64x2x3_t, b: float64x2x3_t) -> uint64x2x3_t {
    unsafe { uint64x2x3_t(vcltq_f64(a.0, b.0), vcltq_f64(a.1, b.1), vcltq_f64(a.2, b.2)) }
}

#[inline(always)]
pub fn vcltq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> uint64x2x3_t {
    unsafe { uint64x2x3_t(vcltq_f64(a.0, b), vcltq_f64(a.1, b), vcltq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vcgtq_f64x4(a: float64x2x2_t, b: float64x2x2_t) -> uint64x2x2_t {
    unsafe { uint64x2x2_t(vcgtq_f64(a.0, b.0), vcgtq_f64(a.1, b.1)) }
}

#[inline(always)]
pub fn vcgtq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> uint64x2x3_t {
    unsafe { uint64x2x3_t(vcgtq_f64(a.0, b), vcgtq_f64(a.1, b), vcgtq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vcgtq_f64x4x2(a: float64x2x2_t, b: float64x2_t) -> uint64x2x2_t {
    unsafe { uint64x2x2_t(vcgtq_f64(a.0, b), vcgtq_f64(a.1, b)) }
}

#[inline(always)]
pub fn vcgeq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> uint64x2x3_t {
    unsafe { uint64x2x3_t(vcgeq_f64(a.0, b), vcgeq_f64(a.1, b), vcgeq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vcleq_f64x6x2(a: float64x2x3_t, b: float64x2_t) -> uint64x2x3_t {
    unsafe { uint64x2x3_t(vcleq_f64(a.0, b), vcleq_f64(a.1, b), vcleq_f64(a.2, b)) }
}

#[inline(always)]
pub fn vbslq_f64x4(a: uint64x2x2_t, b: float64x2x2_t, c: float64x2x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vbslq_f64(a.0, b.0, c.0), vbslq_f64(a.1, b.1, c.1)) }
}

#[inline(always)]
pub fn vbslq_f64x6(a: uint64x2x3_t, b: float64x2x3_t, c: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vbslq_f64(a.0, b.0, c.0), vbslq_f64(a.1, b.1, c.1), vbslq_f64(a.2, b.2, c.2)) }
}

#[inline(always)]
pub fn vbslq_f64x4x2(a: uint64x2x2_t, b: float64x2x2_t, c: float64x2_t) -> float64x2x2_t {
    unsafe { float64x2x2_t(vbslq_f64(a.0, b.0, c), vbslq_f64(a.1, b.1, c)) }
}

#[inline(always)]
pub fn vbslq_f64x6x2(a: uint64x2x3_t, b: float64x2x3_t, c: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vbslq_f64(a.0, b.0, c), vbslq_f64(a.1, b.1, c), vbslq_f64(a.2, b.2, c)) }
}

#[inline(always)]
pub fn vbslq_f64x6x2x2(a: uint64x2x3_t, b: float64x2_t, c: float64x2_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vbslq_f64(a.0, b, c), vbslq_f64(a.1, b, c), vbslq_f64(a.2, b, c)) }
}

#[inline(always)]
pub fn vbslq_f64x2x6(a: uint64x2x3_t, b: float64x2_t, c: float64x2x3_t) -> float64x2x3_t {
    unsafe { float64x2x3_t(vbslq_f64(a.0, b, c.0), vbslq_f64(a.1, b, c.1), vbslq_f64(a.2, b, c.2)) }
}

#[inline(always)]
pub fn vandq_u64x4(a: uint64x2x2_t, b: uint64x2x2_t) -> uint64x2x2_t {
    unsafe { uint64x2x2_t(vandq_u64(a.0, b.0), vandq_u64(a.1, b.1)) }
}

#[inline(always)]
pub fn vandq_u64x6(a: uint64x2x3_t, b: uint64x2x3_t) -> uint64x2x3_t {
    unsafe { uint64x2x3_t(vandq_u64(a.0, b.0), vandq_u64(a.1, b.1), vandq_u64(a.2, b.2)) }
}

#[inline(always)]
pub fn vcgezq_f64x6(x: float64x2x3_t) -> uint64x2x3_t {
    unsafe { uint64x2x3_t(vcgezq_f64(x.0), vcgezq_f64(x.1), vcgezq_f64(x.2)) }
}

#[inline(always)]
pub fn vsignq_f64x6(x: float64x2x3_t) -> float64x2x3_t {
    unsafe {
        let sign_mask = vdupq_n_u64(0x8000_0000_0000_0000);
        let ones = vdupq_n_f64(1.0);

        float64x2x3_t(
            vbslq_f64(sign_mask, x.0, ones),
            vbslq_f64(sign_mask, x.1, ones),
            vbslq_f64(sign_mask, x.2, ones),
        )
    }
}

#[inline(always)]
pub fn vreinterpretq_f64_u64_x6(a: uint64x2x3_t) -> float64x2x3_t {
    unsafe {
        float64x2x3_t(
            vreinterpretq_f64_u64(a.0),
            vreinterpretq_f64_u64(a.1),
            vreinterpretq_f64_u64(a.2),
        )
    }
}

#[inline(always)]
pub fn transmute_f64x2(x: float64x2_t) -> [f64; 2] {
    unsafe {
        let mut arr = [0f64; 2];
        vst1q_f64(arr.as_mut_ptr(), x);
        arr
    }
}

#[inline(always)]
pub fn transmute_f64x4(x: float64x2x2_t) -> [f64; 4] {
    unsafe {
        let mut arr = [0f64; 4];
        vst1q_f64_x2(arr.as_mut_ptr(), x);
        arr
    }
}

#[inline(always)]
pub fn transmute_f64x6(x: float64x2x3_t) -> [f64; 6] {
    unsafe {
        let mut arr = [0f64; 6];
        vst1q_f64_x3(arr.as_mut_ptr(), x);
        arr
    }
}

#[inline(always)]
pub fn transmute_u64x2(x: uint64x2_t) -> [u64; 2] {
    unsafe {
        let mut arr = [0u64; 2];
        vst1q_u64(arr.as_mut_ptr(), x);
        arr
    }
}

#[inline(always)]
pub fn transmute_u64x4(x: uint64x2x2_t) -> [u64; 4] {
    unsafe {
        let mut arr = [0u64; 4];
        vst1q_u64_x2(arr.as_mut_ptr(), x);
        arr
    }
}

struct HiLo {
    exp_hi: float64x2_t,
    exp_lo: float64x2_t,
    log2e: float64x2_t,
    ln2_hi: float64x2_t,
    ln2_lo: float64x2_t,
}

#[inline(always)]
fn hi_lo() -> HiLo {
    unsafe {
        HiLo {
            exp_hi: vdupq_n_f64(709.782712893384),
            exp_lo: vdupq_n_f64(-708.396418532264),

            log2e: vdupq_n_f64(1.4426950408889634074),
            ln2_hi: vdupq_n_f64(0.69314718055994528623),
            ln2_lo: vdupq_n_f64(2.3190428994635850965e-17),
        }
    }
}

/// 11 degree polynomial
/// Max Error is ~5.7e-15
#[inline]
pub fn exp_x2(mut x: float64x2_t) -> float64x2_t {
    unsafe {
        let HiLo { exp_hi, exp_lo, log2e, ln2_hi, ln2_lo } = hi_lo();

        let c0  = vdupq_n_f64(1.0);
        let c1  = vdupq_n_f64(1.0);
        let c2  = vdupq_n_f64(0.5);
        let c3  = vdupq_n_f64(0.16666666666666666667);
        let c4  = vdupq_n_f64(0.04166666666666666667);
        let c5  = vdupq_n_f64(0.00833333333333333333);
        let c6  = vdupq_n_f64(0.00138888888888888889);
        let c7  = vdupq_n_f64(0.00019841269841269841);
        let c8  = vdupq_n_f64(0.00002480158730158730);
        let c9  = vdupq_n_f64(0.00000275573192239859);
        let c10 = vdupq_n_f64(0.00000027557319223986);
        let c11 = vdupq_n_f64(0.00000002505210838544);

        x = vminq_f64(x, exp_hi);
        x = vmaxq_f64(x, exp_lo);

        let mut fx = vfmaq_f64(vdupq_n_f64(0.0), x, log2e);
        let n = vcvtaq_s64_f64(fx); 
        fx = vcvtq_f64_s64(n);

        let mut r = vfmsq_f64(x, fx, ln2_hi);
        r = vfmsq_f64(r, fx, ln2_lo);

        let mut y = vfmaq_f64(c10, r, c11);
        y = vfmaq_f64(c9, r, y);
        y = vfmaq_f64(c8, r, y);
        y = vfmaq_f64(c7, r, y);
        y = vfmaq_f64(c6, r, y);
        y = vfmaq_f64(c5, r, y);
        y = vfmaq_f64(c4, r, y);
        y = vfmaq_f64(c3, r, y);
        y = vfmaq_f64(c2, r, y);
        y = vfmaq_f64(c1, r, y);
        y = vfmaq_f64(c0, r, y);

        let pow2n = vshlq_n_s64(vaddq_s64(n, vdupq_n_s64(1023)), 52);
        vmulq_f64(y, vreinterpretq_f64_s64(pow2n))
    }
}

#[inline(always)]
pub fn exp_x4(x: float64x2x2_t) -> float64x2x2_t {
    float64x2x2_t(exp_x2(x.0), exp_x2(x.1))
}

/// 6 degree polynomial
#[inline(always)]
pub fn simd_fast_exp_x2(mut x: float64x2_t) -> float64x2_t {
    unsafe {
        let HiLo { exp_hi, exp_lo, log2e, ln2_hi, ln2_lo } = hi_lo();

        let c0 = vdupq_n_f64(1.0);
        let c1 = vdupq_n_f64(1.0);
        let c2 = vdupq_n_f64(0.5);
        let c3 = vdupq_n_f64(0.166666666666666667);
        let c4 = vdupq_n_f64(0.041666666666666666);
        let c5 = vdupq_n_f64(0.008333333333333333);
        let c6 = vdupq_n_f64(0.001388888888888888);

        x = vminq_f64(x, exp_hi);
        x = vmaxq_f64(x, exp_lo);

        let mut fx = vfmaq_f64(vdupq_n_f64(0.0), x, log2e);
        let n = vcvtaq_s64_f64(fx); 
        fx = vcvtq_f64_s64(n);

        let mut r = vfmsq_f64(x, fx, ln2_hi);
        r = vfmsq_f64(r, fx, ln2_lo);

        // Evaluate polynomial using Horner's Method:
        // c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
        let mut y = vfmaq_f64(c5, r, c6);
        y = vfmaq_f64(c4, r, y);
        y = vfmaq_f64(c3, r, y);
        y = vfmaq_f64(c2, r, y);
        y = vfmaq_f64(c1, r, y);
        y = vfmaq_f64(c0, r, y);

        let pow2n = vshlq_n_s64(vaddq_s64(n, vdupq_n_s64(1023)), 52);
        vmulq_f64(y, vreinterpretq_f64_s64(pow2n))
    }
}

struct LnConstants {
    ln2_hi: float64x2_t,
    ln2_lo: float64x2_t,
    sqrt2: float64x2_t,
}

#[inline(always)]
fn ln_constants() -> LnConstants {
    unsafe {
        LnConstants {
            ln2_hi: vdupq_n_f64(0.69314718055994528623),
            ln2_lo: vdupq_n_f64(2.3190428994635850965e-17),
            sqrt2: vdupq_n_f64(1.4142135623730950488),
        }
    }
}

#[inline]
pub fn ln_x2(x: float64x2_t) -> float64x2_t {
    unsafe {
        let LnConstants { ln2_hi, ln2_lo, sqrt2 } = ln_constants();

        let zero = vdupq_n_f64(0.0);
        let inf = vdupq_n_f64(f64::INFINITY);
        
        let is_lt_zero = vcltq_f64(x, zero);
        let is_zero = vceqq_f64(x, zero);
        let is_inf = vceqq_f64(x, inf);

        // Extract Exponent and Mantissa using bit manipulation
        let x_u = vreinterpretq_u64_f64(x);
        
        // Extract integer exponent: e = (x_bits >> 52) - 1023
        let e_int = vsubq_s64(
            vreinterpretq_s64_u64(vshrq_n_u64(x_u, 52)),
            vdupq_n_s64(1023),
        );

        // Force mantissa bits to represent a float in [1.0, 2.0)
        let mantissa_mask = vdupq_n_u64(0x000FFFFFFFFFFFFF);
        let exp_bias = vdupq_n_u64(0x3FF0000000000000);
        let m_u = vorrq_u64(vandq_u64(x_u, mantissa_mask), exp_bias);
        let mut m = vreinterpretq_f64_u64(m_u);

        // Map m from [1.0, 2.0) to [sqrt(2)/2, sqrt(2)) to speed up convergence.
        let mask_gt_sqrt2 = vcgtq_f64(m, sqrt2);

        // If m > sqrt(2), we halve m and add 1 to the exponent
        m = vbslq_f64(mask_gt_sqrt2, vmulq_f64(m, vdupq_n_f64(0.5)), m);
        
        // SIMD trick: mask_gt_sqrt2 is all 1s (-1 as s64) when true. 
        // Subtracting -1 adds 1 to the exponent.
        let e_adj = vsubq_s64(e_int, vreinterpretq_s64_u64(mask_gt_sqrt2));
        let e_float = vcvtq_f64_s64(e_adj);

        // Compute Substitution: s = (m - 1) / (m + 1)
        let one = vdupq_n_f64(1.0);
        let s = vdivq_f64(vsubq_f64(m, one), vaddq_f64(m, one));
        let z = vmulq_f64(s, s); // z = s^2

        // Polynomial Evaluation (Horner's Method)
        // Taylor series coefficients for ln((1+s)/(1-s))
        let c1 = vdupq_n_f64(0.6666666666666666); // 2/3
        let c2 = vdupq_n_f64(0.4000000000000000); // 2/5
        let c3 = vdupq_n_f64(0.2857142857142857); // 2/7
        let c4 = vdupq_n_f64(0.2222222222222222); // 2/9
        let c5 = vdupq_n_f64(0.1818181818181818); // 2/11
        let c6 = vdupq_n_f64(0.1538461538461538); // 2/13
        let c7 = vdupq_n_f64(0.1333333333333333); // 2/15
        let c8 = vdupq_n_f64(0.1176470588235294); // 2/17
        let c9 = vdupq_n_f64(0.1052631578947368); // 2/19

        let mut r = vfmaq_f64(c8, c9, z);
        r = vfmaq_f64(c7, r, z);
        r = vfmaq_f64(c6, r, z);
        r = vfmaq_f64(c5, r, z);
        r = vfmaq_f64(c4, r, z);
        r = vfmaq_f64(c3, r, z);
        r = vfmaq_f64(c2, r, z);
        r = vfmaq_f64(c1, r, z);

        // ln(m) = 2s + s * z * R(z)
        let mut ln_m = vmulq_f64(s, vdupq_n_f64(2.0));
        ln_m = vfmaq_f64(ln_m, vmulq_f64(s, z), r);

        // ln(x) = e * ln(2) + ln(m)
        let mut res = vmulq_f64(e_float, ln2_hi);
        res = vfmaq_f64(res, e_float, ln2_lo);
        res = vaddq_f64(res, ln_m);

        // Apply Edge Cases Masks
        let nan = vdupq_n_f64(f64::NAN);
        let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);

        res = vbslq_f64(is_inf, inf, res);
        res = vbslq_f64(is_zero, neg_inf, res);
        res = vbslq_f64(is_lt_zero, nan, res);

        res
    }
}

#[inline(always)]
pub fn ln1p_x2(x: float64x2_t) -> float64x2_t {
    unsafe {
        let one = vdupq_n_f64(1.0);
        let u = vaddq_f64(one, x);
        let ln_u = ln_x2(u);
        let u_m1 = vsubq_f64(u, one);
        let is_degenerate = vceqq_f64(u_m1, vdupq_n_f64(0.0));
        let ratio = vdivq_f64(vmulq_f64(ln_u, x), u_m1);
        vbslq_f64(is_degenerate, x, ratio)
    }
}

#[inline(always)]
pub fn ln1p_x4(x: float64x2x2_t) -> float64x2x2_t {
    float64x2x2_t(ln1p_x2(x.0), ln1p_x2(x.1))
}

#[inline(always)]
pub fn softplus_x2(x: float64x2_t, limit: f64, floor: f64) -> float64x2_t {
    unsafe {
        let exp_limit = vdupq_n_f64(limit);
        let neg_exp_limit = vdupq_n_f64(-limit);
        let floor = vdupq_n_f64(floor);

        let mask_gt = vcgtq_f64(x, exp_limit);
        let mask_lt = vcltq_f64(x, neg_exp_limit);

        let e = exp_x2(x);
        let softplus = ln1p_x2(e);
        let lower_select = vbslq_f64(mask_lt, floor, softplus);
        vbslq_f64(mask_gt, x, lower_select)
    }
}

#[inline(always)]
pub fn sigmoid_x2(x: float64x2_t, limit: f64, floor: f64) -> float64x2_t {
    unsafe {
        let exp_limit = vdupq_n_f64(limit);
        let neg_exp_limit = vdupq_n_f64(-limit);
        let one = vdupq_n_f64(1.0);
        let floor = vdupq_n_f64(floor);

        let mask_gt = vcgtq_f64(x, exp_limit);
        let mask_lt = vcltq_f64(x, neg_exp_limit);

        let e = exp_x2(x);
        let sigmoid = vdivq_f64(e, vaddq_f64(one, e));
        let lower_select = vbslq_f64(mask_lt, floor, sigmoid);
        vbslq_f64(mask_gt, one, lower_select)
    }
}

#[inline]
pub fn softplus_sigmoid_x2(
    x: float64x2_t,
    floor: float64x2_t,
    limit: f64,
) -> (float64x2_t, float64x2_t) {
    unsafe {
        let exp_limit = vdupq_n_f64(limit);
        let neg_exp_limit = vdupq_n_f64(-limit);
        let one = vdupq_n_f64(1.0);

        let mask_hi = vcgtq_f64(x, exp_limit);
        let mask_lo = vcltq_f64(x, neg_exp_limit);
        let e = exp_x2(x);

        let softplus = {
            let softplus = ln1p_x2(e);
            let select_lo = vbslq_f64(mask_lo, floor, softplus);
            vbslq_f64(mask_hi, x, select_lo)
        };

        let sigmoid = {
            let sigmoid = vdivq_f64(e, vaddq_f64(one, e));
            let select_lo = vbslq_f64(mask_lo, floor, sigmoid);
            vbslq_f64(mask_hi, one, select_lo)
        };

        (softplus, sigmoid)
    }
}

#[inline(always)]
pub fn softplus_sigmoid_x4(
    x: float64x2x2_t,
    floor: float64x2_t,
    limit: f64,
) -> (float64x2x2_t, float64x2x2_t) {
    let (softplus0, sigmoid0) = softplus_sigmoid_x2(x.0, floor, limit);
    let (softplus1, sigmoid1) = softplus_sigmoid_x2(x.1, floor, limit);
    (float64x2x2_t(softplus0, softplus1), float64x2x2_t(sigmoid0, sigmoid1))
}

#[inline(always)]
pub fn softplus_sigmoid_x6(
    x: float64x2x3_t,
    floor: float64x2_t,
    limit: f64,
) -> (float64x2x3_t, float64x2x3_t) {
    let (softplus0, sigmoid0) = softplus_sigmoid_x2(x.0, floor, limit);
    let (softplus1, sigmoid1) = softplus_sigmoid_x2(x.1, floor, limit);
    let (softplus2, sigmoid2) = softplus_sigmoid_x2(x.2, floor, limit);
    (float64x2x3_t(softplus0, softplus1, softplus2), float64x2x3_t(sigmoid0, sigmoid1, sigmoid2))
}

#[inline]
pub fn powf_x2(x: float64x2_t, y: float64x2_t) -> float64x2_t {
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let one = vdupq_n_f64(1.0);

        let abs_x = vabsq_f64(x);
        let ln_ax = ln_x2(abs_x);
        let y_ln_ax = vmulq_f64(y, ln_ax);
        let res_mag = exp_x2(y_ln_ax);

        let is_neg_x = vcltq_f64(x, zero);

        let y_i = vcvtaq_s64_f64(y);
        let y_rounded_f = vcvtq_f64_s64(y_i);
        let is_y_integer = vceqq_f64(y, y_rounded_f);
        let is_y_odd = vtstq_s64(y_i, vdupq_n_s64(1));

        let sign_flip = vandq_u64(vandq_u64(is_neg_x, is_y_integer), is_y_odd);
        let undefined = vbicq_u64(is_neg_x, is_y_integer);

        let mut res = vbslq_f64(sign_flip, vnegq_f64(res_mag), res_mag);
        res = vbslq_f64(undefined, vdupq_n_f64(f64::NAN), res);

        let is_y_zero = vceqq_f64(y, zero);
        let is_x_one = vceqq_f64(x, one);
        let force_one = vorrq_u64(is_y_zero, is_x_one);
        res = vbslq_f64(force_one, one, res);

        res
    }
}

#[inline(always)]
pub fn powf_x4(x: float64x2x2_t, y: float64x2x2_t) -> float64x2x2_t {
    float64x2x2_t(powf_x2(x.0, y.0), powf_x2(x.1, y.1))
}

#[inline(always)]
pub fn powf_x6(x: float64x2x3_t, y: float64x2x3_t) -> float64x2x3_t {
    float64x2x3_t(powf_x2(x.0, y.0), powf_x2(x.1, y.1), powf_x2(x.2, y.2))
}

#[inline(always)]
pub fn powf_x6x2(x: float64x2x3_t, y: float64x2_t) -> float64x2x3_t {
    float64x2x3_t(powf_x2(x.0, y), powf_x2(x.1, y), powf_x2(x.2, y))
}

#[inline]
pub fn atan_f64x2(x: float64x2_t) -> float64x2_t {
    unsafe {
        // atan(0.5), atan(1.0), atan(1.5), atan(inf), split into hi/lo parts
        // for extra precision when reconstructing the final result.
        const ATAN_HI: [f64; 4] = [
            4.63647609000806093515e-01,
            7.85398163397448278999e-01,
            9.82793723247329054082e-01,
            1.57079632679489655800e+00,
        ];
        const ATAN_LO: [f64; 4] = [
            2.26987774529616870924e-17,
            3.06161699786838301793e-17,
            1.39033110312309984516e-17,
            6.12323399573676603587e-17,
        ];
        // Minimax coefficients for atan(x) on |x| <= 0.4375 (odd polynomial,
        // indices 0..10 correspond to x^1, x^3, x^5, ... x^21).
        const AT: [f64; 11] = [
            3.33333333333329318027e-01,
            -1.99999999998764832476e-01,
            1.42857142725034663711e-01,
            -1.11111104054623557880e-01,
            9.09088713343650656196e-02,
            -7.69187620504482999495e-02,
            6.66107313738753120669e-02,
            -5.83357013379057348645e-02,
            4.97687799461593236017e-02,
            -3.65315727442169155270e-02,
            1.62858201153657823623e-02,
        ];
 
        let zero = vdupq_n_f64(0.0);
        let one = vdupq_n_f64(1.0);
        let two = vdupq_n_f64(2.0);
        let onehalf = vdupq_n_f64(1.5);
 
        let neg_mask = vcltq_f64(x, zero);
        let ax = vabsq_f64(x);
 
        let is_small = vcltq_f64(ax, vdupq_n_f64(0.4375)); // |x| < 0.4375 -> use x directly
        let is_a = vcltq_f64(ax, vdupq_n_f64(0.6875));      // 0.4375 <= |x| < 0.6875
        let is_b = vcltq_f64(ax, vdupq_n_f64(1.1875));      // |x| < 1.1875
        let is_c = vcltq_f64(ax, vdupq_n_f64(2.4375));      // 1.1875 <= |x| < 2.4375
 
        // Reduced-argument candidates, computed unconditionally for all
        // lanes; whichever ones aren't selected below are simply discarded
        // (this can transiently divide by zero / produce inf in an unused
        // lane, which is harmless — SIMD lanes don't interact).
        let r1 = vdivq_f64(vsubq_f64(ax, one), vaddq_f64(ax, one));
        let r0 = vdivq_f64(vsubq_f64(vmulq_f64(two, ax), one), vaddq_f64(two, ax));
        let r2 = vdivq_f64(vsubq_f64(ax, onehalf), vaddq_f64(one, vmulq_f64(onehalf, ax)));
        let r3 = vdivq_f64(vdupq_n_f64(-1.0), ax);
 
        let low = vbslq_f64(is_a, r1, r0);
        let high = vbslq_f64(is_c, r2, r3);
        let reduced = vbslq_f64(is_b, low, high);
        let xr = vbslq_f64(is_small, ax, reduced);
 
        // hi/lo constant selection, mirroring the same region layout.
        let hi_low = vbslq_f64(is_a, vdupq_n_f64(ATAN_HI[1]), vdupq_n_f64(ATAN_HI[0]));
        let hi_high = vbslq_f64(is_c, vdupq_n_f64(ATAN_HI[2]), vdupq_n_f64(ATAN_HI[3]));
        let hi_sel = vbslq_f64(is_b, hi_low, hi_high);
 
        let lo_low = vbslq_f64(is_a, vdupq_n_f64(ATAN_LO[1]), vdupq_n_f64(ATAN_LO[0]));
        let lo_high = vbslq_f64(is_c, vdupq_n_f64(ATAN_LO[2]), vdupq_n_f64(ATAN_LO[3]));
        let lo_sel = vbslq_f64(is_b, lo_low, lo_high);
 
        // Minimax polynomial in w = xr^4, evaluated via Horner + fma.
        let z = vmulq_f64(xr, xr);
        let w = vmulq_f64(z, z);
 
        let mut p = vdupq_n_f64(AT[10]);
        p = vfmaq_f64(vdupq_n_f64(AT[8]), w, p);
        p = vfmaq_f64(vdupq_n_f64(AT[6]), w, p);
        p = vfmaq_f64(vdupq_n_f64(AT[4]), w, p);
        p = vfmaq_f64(vdupq_n_f64(AT[2]), w, p);
        p = vfmaq_f64(vdupq_n_f64(AT[0]), w, p);
        let s1 = vmulq_f64(z, p);
 
        let mut q = vdupq_n_f64(AT[9]);
        q = vfmaq_f64(vdupq_n_f64(AT[7]), w, q);
        q = vfmaq_f64(vdupq_n_f64(AT[5]), w, q);
        q = vfmaq_f64(vdupq_n_f64(AT[3]), w, q);
        q = vfmaq_f64(vdupq_n_f64(AT[1]), w, q);
        let s2 = vmulq_f64(w, q);
 
        let s = vaddq_f64(s1, s2);
 
        // Small-x result: xr - xr*s
        let result_small = vsubq_f64(xr, vmulq_f64(xr, s));
 
        // General result: hi - ((xr*s - lo) - xr)
        let t = vsubq_f64(vmulq_f64(xr, s), lo_sel);
        let t = vsubq_f64(t, xr);
        let result_general = vsubq_f64(hi_sel, t);
 
        let result = vbslq_f64(is_small, result_small, result_general);
        vbslq_f64(neg_mask, vnegq_f64(result), result)
    }
}

#[inline(always)]
pub fn atan_f64x4(x: float64x2x2_t) -> float64x2x2_t {
    float64x2x2_t(atan_f64x2(x.0), atan_f64x2(x.1))
}

#[inline(always)]
pub fn safe_exp_x2(x: float64x2_t, limit: float64x2_t) -> (float64x2_t, float64x2_t) {
    unsafe {
        let mask = vcgtq_f64(x, limit);
        let arg = vbslq_f64(mask, limit, x);
        let e = exp_x2(arg);
        let one = vdupq_n_f64(1.0);
        let e_crit = vbslq_f64(
            mask,
            vmulq_f64(e, vaddq_f64(one, vsubq_f64(x, limit))),
            e
        );
        (e_crit, e)
    }
}

#[inline(always)]
pub fn safe_exp_x4x2(x: float64x2x2_t, limit: float64x2_t) -> (float64x2x2_t, float64x2x2_t) {
    let (e0, de0) = safe_exp_x2(x.0, limit);
    let (e1, de1) = safe_exp_x2(x.1, limit);
    (float64x2x2_t(e0, e1), float64x2x2_t(de0, de1))
}
