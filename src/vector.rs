use crate::traits::Container;

#[repr(align(32))]
#[derive(Clone, Copy)]
pub struct VecF<const N: usize> {
    pub data: [f64; N]
}

impl<const N: usize> Default for VecF<N> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const N: usize> VecF<N> {
    pub const ZERO: Self = Self { data: [0f64; N] };

    pub const fn zero() -> Self {
        Self::ZERO
    }

    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::ZERO
    }

    #[inline(always)]
    pub fn ptr(&self, offset: usize) -> *const f64 {
        unsafe { self.data.as_ptr().add(offset) }
    }

    #[inline(always)]
    pub fn ptr_mut(&mut self, offset: usize) -> *mut f64 {
        unsafe { self.data.as_mut_ptr().add(offset) }
    }

    #[inline(always)]
    pub const fn swap(&mut self, a: usize, b: usize) {
        self.data.swap(a, b)
    }

    pub fn map<R>(&self, f: impl FnMut(f64) -> R) -> [R; N] {
        self.data.map(f)
    }

    pub fn iter(&self) -> core::slice::Iter<'_, f64> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, f64> {
        self.data.iter_mut()
    }

    pub fn chunks_exact(&self, size: usize) -> core::slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(size)
    }

    pub fn chunks_exact_mut(&mut self, size: usize) -> core::slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(size)
    }

    pub fn scalar_mul(&self, rhs: f64) -> Self {
        let mut out = Self::ZERO;
        out.iter_mut()
            .zip(&self.data)
            .for_each(|(val, src)| *val = src * rhs);
        out
    }

    #[inline]
    /// res = self * scalar - rhs
    pub fn scalar_mul_sub(&self, scalar: f64, rhs: &Self) -> Self {
        let mut res = Self::ZERO;
        self.scalar_mul_sub_into(scalar, rhs, &mut res);
        res
    }
}

// =============================================================================
// Neon impl
// =============================================================================

#[cfg(target_feature = "neon")]
use core::arch::aarch64::*;

#[cfg(target_feature = "neon")]
impl<const N: usize> VecF<N> {
    #[inline]
    /// res = self - mat * b
    pub fn sub_mat_vec_mul_into<const COL: usize, MAT>(
        &self,
        mat: &MAT,
        b: &VecF<COL>,
        res: &mut Self,
    )
    where
        MAT: Container<N, COL>
    {
        for i in 0..N {
            let offset = i * COL;
            let mut acc0 = unsafe { vdupq_n_f64(0.0) };
            let mut acc1 = unsafe { vdupq_n_f64(0.0) };

            let mut j = 0;
            while j + 4 <= COL {
                unsafe {
                    let m_vec = vld1q_f64_x2(mat.ptr(offset + j));
                    let b_vec = vld1q_f64_x2(b.ptr(j));
                    acc0 = vfmaq_f64(acc0, m_vec.0, b_vec.0);
                    acc1 = vfmaq_f64(acc1, m_vec.1, b_vec.1);
                }
                j += 4;
            }
            let mut acc = unsafe { vaddq_f64(acc0, acc1) };
            if j + 2 <= COL {
                unsafe {
                    let m_vec = vld1q_f64(mat.ptr(offset + j));
                    let b_vec = vld1q_f64(b.ptr(j));
                    acc = vfmaq_f64(acc, m_vec, b_vec);
                }
                j += 2;
            }
            let mut dot = unsafe { vaddvq_f64(acc) };
            if j < COL {
                dot += mat[offset + j] * b[j];
            }
            res[i] = self[i] - dot;
        }
    }

    #[inline]
    /// res = self * scalar - rhs
    pub fn scalar_mul_sub_into(&self, scalar: f64, rhs: &Self, res: &mut Self) {

        let s_v = unsafe { vdupq_n_f64(scalar) };
        let mut i = 0;
        while i + 4 <= N {
            unsafe {
                let l_v = vld1q_f64_x2(self.ptr(i));
                let r_v = vld1q_f64_x2(rhs.ptr(i));
                vst1q_f64_x2(
                    res.ptr_mut(i),
                    float64x2x2_t(
                        vfmaq_f64(vnegq_f64(r_v.0), l_v.0, s_v),
                        vfmaq_f64(vnegq_f64(r_v.1), l_v.1, s_v),
                    )
                );
            }
            i += 4
        }
        while i + 2 <= N {
            unsafe {
                let l_v = vld1q_f64(self.data.as_ptr().add(i));
                let r_v = vld1q_f64(rhs.data.as_ptr().add(i));
                vst1q_f64(
                    res.data.as_mut_ptr().add(i),
                    vfmaq_f64(vnegq_f64(r_v), l_v, s_v),
                );
            }
            i += 2
        }
        if i < N {
            res[i] = self[i] * scalar - rhs[i]
        }
    }

    #[inline]
    pub fn vec_cross_mul(&self, rhs: &Self, res: &mut Self) {
        let mut i = 0;
        while i + 4 <= N {
            unsafe {
                let l_vec = vld1q_f64_x2(self.ptr(i));
                let r_vec = vld1q_f64_x2(rhs.ptr(i));
                vst1q_f64_x2(
                    res.ptr_mut(i),
                    float64x2x2_t(
                        vmulq_f64(l_vec.0, r_vec.0),
                        vmulq_f64(l_vec.1, r_vec.1),
                    )
                );
            }
            i += 4;
        }
        if i + 2 <= N {
            unsafe {
                let l_vec = vld1q_f64(self.ptr(i));
                let r_vec = vld1q_f64(rhs.ptr(i));
                vst1q_f64(res.ptr_mut(i), vmulq_f64(l_vec, r_vec));
            }
            i += 2;
        }
        if i < N {
            res[i] = self[i] * rhs[i];
        }
    }
}

#[cfg(not(target_feature = "neon"))]
impl<const N: usize> VecF<N> {
    #[inline]
    /// res = self - mat * b
    pub fn sub_mat_vec_mul_into<const COL: usize, MAT>(
        &self,
        mat: &MAT,
        b: &VecF<COL>,
        res: &mut Self,
    )
    where
        MAT: Container<N, COL>
    {
        for i in 0..N {
            let offset = i * COL;
            let mut acc = 0.0;
            for j in 0..COL {
                acc += mat[offset + j] * b[j];
            }
            res[i] = self[i] - acc;
        }
    }

    #[inline]
    /// res = self * scalar - rhs
    pub fn scalar_mul_sub_into(&self, scalar: f64, rhs: &Self, res: &mut Self) {
        for i in 0..N {
            res[i] = self[i] * scalar - rhs[i]
        }
    }

    #[inline]
    pub fn vec_cross_mul(&self, rhs: &Self, res: &mut Self) {
        for i in 0..N {
            res[i] = self[i] * rhs[i];
        }
    }
}

// =============================================================================
// PartialEq
// =============================================================================

impl<const N: usize> PartialEq for VecF<N> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

// =============================================================================
// Index & IndexMut
// =============================================================================

impl<const N: usize> core::ops::Index<usize> for VecF<N> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<const N: usize> core::ops::IndexMut<usize> for VecF<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<const N: usize> core::ops::Index<core::ops::Range<usize>> for VecF<N> {
    type Output = [f64];

    fn index(&self, index: core::ops::Range<usize>) -> &Self::Output {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<const N: usize> core::ops::IndexMut<core::ops::Range<usize>> for VecF<N> {
    fn index_mut(&mut self, index: core::ops::Range<usize>) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

// =============================================================================
// IntoIterator
// =============================================================================

impl<'a, const N: usize> IntoIterator for &'a VecF<N> {
    type Item = &'a f64;
    type IntoIter = core::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a mut VecF<N> {
    type Item = &'a mut f64;
    type IntoIter = core::slice::IterMut<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<const N: usize> FromIterator<f64> for VecF<N> {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut this = Self::ZERO;
        this.iter_mut().zip(iter).for_each(|(a, b)| *a = b);
        this
    }
}

// =============================================================================
// Debug
// =============================================================================

impl<const N: usize> core::fmt::Debug for VecF<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        self.data.iter().try_for_each(|v| write!(f, " {:.2e} ", v))?;
        write!(f, "]")
    }
}
