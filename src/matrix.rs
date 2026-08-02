#[cfg(target_feature = "neon")]
use core::arch::aarch64::*;
use crate::vector::VecF;
use crate::traits::Container;

// =============================================================================
// Matrix
// =============================================================================

/// A row-major matrix with `ROW` rows and `COL` columns, stored in flat array.
/// `LEN` must equal `ROW * COL`. `COL` defaults to `ROW` for square matrices.
///
/// # Examples
///
/// ```ignore
/// // Square 7x7
/// const N: usize = 7;
/// const LEN: usize = N * N;
/// type Mat7 = Matrix2<LEN, N>;
/// let m = Mat7::IDENTITY;
///
/// // Rectangular 3x5
/// const ROW: usize = 3;
/// const COL: usize = 5;
/// const LEN: usize = ROW * COL;
/// type Mat3x5 = Matrix2<LEN, ROW, COL>;
/// let z = Mat3x5::ZERO;
/// ```
#[repr(align(32))]
#[derive(Clone, Copy)]
pub struct Matrix<const LEN: usize, const ROW: usize, const COL: usize = ROW> {
    /// Row-major storage.
    pub data: [f64; LEN],
}

impl<const LEN: usize, const ROW: usize, const COL: usize>
Container<ROW, COL> for Matrix<LEN, ROW, COL> {
    #[inline(always)]
    fn ptr(&self, offset: usize) -> *const f64 {
        unsafe { self.data.as_ptr().add(offset) }
    }

    #[inline(always)]
    fn ptr_mut(&mut self, offset: usize) -> *mut f64 {
        unsafe { self.data.as_mut_ptr().add(offset) }
    }
}

// =============================================================================
// General impl (any ROW x COL)
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> Matrix<LEN, ROW, COL> {
    pub const ZERO: Self = Self { data: [0.0; LEN] };

    #[inline(always)]
    pub fn ptr(&self, offset: usize) -> *const f64 {
        unsafe { self.data.as_ptr().add(offset) }
    }

    #[inline(always)]
    pub fn ptr_mut(&mut self, offset: usize) -> *mut f64 {
        unsafe { self.data.as_mut_ptr().add(offset) }
    }

    #[inline(always)]
    pub const fn split_at_mut(&mut self, mid: usize) -> (&mut [f64], &mut [f64]) {
        unsafe { self.data.split_at_mut_unchecked(mid) }
    }

    pub const fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }

    pub fn rows(&self) -> core::slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(COL)
    }

    pub fn rows_mut(&mut self) -> core::slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(COL)
    }

    pub const fn expand<
        const LEN2: usize,
        const ROW2: usize,
        const COL2: usize
    >(&self) -> Matrix<LEN2, ROW2, COL2> {
        let mut new = Matrix::ZERO;
        let mut i = 0;
        while i < ROW {
            let src_offset = i * COL;
            let dst_offset = i * COL2;
            unsafe {
                let src = self.data.as_ptr().add(src_offset);
                let dst = new.data.as_mut_ptr().add(dst_offset);
                core::ptr::copy_nonoverlapping(src, dst, COL);
            }
            i += 1;
        }
        new
    }

    pub fn transpose(&self) -> Matrix<LEN, COL, ROW> {
        let mut out = Matrix::ZERO;
        for r in 0..ROW {
            let r_offset = r * COL;
            for c in 0..COL {
                out[c * ROW + r] = self[r_offset + c];
            }
        }
        out
    }

    /// only use this for large matrix (>= 64x64)
    pub fn transpose_with_tiling(&self) -> Matrix<LEN, COL, ROW> {
        let mut out = Matrix::ZERO;
        for r_tile in (0..ROW).step_by(8) {
            for c_tile in (0..COL).step_by(8) {
                let r_end = (r_tile + 8).min(ROW);
                let c_end = (c_tile + 8).min(COL);
                for r in r_tile..r_end {
                    let r_offset = r * COL;
                    for c in c_tile..c_end {
                        out[c * ROW + r] = self[r_offset + c];
                    }
                }
            }
        }
        out
    }
}

impl Matrix<9, 3, 3> {
    /// 3x3:
    /// │ a b c │
    /// │ p q r │
    /// │ x y z │
    /// 
    /// det = aqz + brx + cpy - xqc - pbz - yra
    pub const fn inverse(&self) -> Option<Self> {
        let m00 = self.data[0];
        let m01 = self.data[1];
        let m02 = self.data[2];

        let m10 = self.data[3];
        let m11 = self.data[4];
        let m12 = self.data[5];

        let m20 = self.data[6];
        let m21 = self.data[7];
        let m22 = self.data[8];

        let c00 = m11 * m22 - m12 * m21;
        let c10 = m12 * m20 - m10 * m22;
        let c20 = m10 * m21 - m11 * m20;

        let det = m00 * c00 + m01 * c10 + m02 * c20;

        if det.abs() < 1e-12 { return None }

        let inv_det = 1.0 / det;

        let c01 = m02 * m21 - m01 * m22;
        let c11 = m00 * m22 - m02 * m20;
        let c21 = m01 * m20 - m00 * m21;

        let c02 = m01 * m12 - m02 * m11;
        let c12 = m02 * m10 - m00 * m12;
        let c22 = m00 * m11 - m01 * m10;

        let inv = Self {
            data: [
                c00 * inv_det, c01 * inv_det, c02 * inv_det,
                c10 * inv_det, c11 * inv_det, c12 * inv_det,
                c20 * inv_det, c21 * inv_det, c22 * inv_det,
            ],
        };

        Some(inv)
    }
}

// =============================================================================
// Arithmetic Neon
// =============================================================================

#[cfg(target_feature = "neon")]
impl<const LEN: usize, const ROW: usize, const COL: usize> Matrix<LEN, ROW, COL> {
    #[inline(always)]
    pub fn mat_mul_into<const COL2: usize, RHS, DST>(
        &self,
        rhs: &RHS,
        res: &mut DST
    )
    where
        RHS: Container<COL, COL2>,
        DST: Container<ROW, COL2>,
    {
        for i in 0..ROW {
            let i_offset = i * COL;
            let i_offset2 = i * COL2;
            let mut j = 0;

            while j + 4 <= COL2 {
                let mut acc0 = unsafe { vdupq_n_f64(0.0) };
                let mut acc1 = unsafe { vdupq_n_f64(0.0) };
                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i_offset + k]);
                        let rhs_vec = vld1q_f64_x2(rhs.ptr(k * COL2 + j));
                        acc0 = vfmaq_f64(acc0, a_vec, rhs_vec.0);
                        acc1 = vfmaq_f64(acc1, a_vec, rhs_vec.1);
                    }
                }
                unsafe { vst1q_f64_x2(res.ptr_mut(i_offset2 + j), float64x2x2_t(acc0, acc1)) }
                j += 4;
            }

            if j + 2 <= COL2 {
                let mut acc = unsafe { vdupq_n_f64(0.0) };
                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i_offset + k]);
                        let rhs_vec = vld1q_f64(rhs.ptr(k * COL2 + j));
                        acc = vfmaq_f64(acc, a_vec, rhs_vec);
                    }
                }
                unsafe { vst1q_f64(res.ptr_mut(i_offset2 + j), acc) }
                j += 2;
            }

            if j < COL2 {
                let mut sum = 0.0;
                for k in 0..COL {
                    sum += self[i_offset + k] * rhs[k * COL2 + j];
                }
                res[i_offset2 + j] = sum;
            }
        }
    }

    #[inline(always)]
    /// res = self + b * c
    pub fn add_mat_mul_into<const COL2: usize, B, C, DST>(
        &self,
        b: &B,
        c: &C,
        res: &mut DST
    )
    where
        B: Container<ROW, COL2>,
        C: Container<COL2, COL>,
        DST: Container<ROW, COL>,
    {
        for i in 0..ROW {
            let i_offset = i * COL;
            let i_offset2 = i * COL2;
            let mut j = 0;
            while j + 4 <= COL {
                unsafe {
                    let mut acc = vld1q_f64_x2(self.ptr(i_offset + j));
                    for k in 0..COL2 {
                        let b_vec = vdupq_n_f64(b[i_offset2 + k]);
                        let c_vec = vld1q_f64_x2(c.ptr(k * COL + j));
                        acc.0 = vfmaq_f64(acc.0, b_vec, c_vec.0);
                        acc.1 = vfmaq_f64(acc.1, b_vec, c_vec.1);
                    }
                    vst1q_f64_x2(res.ptr_mut(i_offset + j), acc);
                }
                j += 4;
            }
            if j + 2 <= COL {
                unsafe {
                    let mut acc = vld1q_f64(self.ptr(i_offset + j));
                    for k in 0..COL2 {
                        let b_vec = vdupq_n_f64(b[i_offset2 + k]);
                        let c_vec = vld1q_f64(c.ptr(k * COL + j));
                        acc = vfmaq_f64(acc, b_vec, c_vec);
                    }
                    vst1q_f64(res.ptr_mut(i_offset + j), acc);
                }
                j += 2;
            }
            if j < COL {
                let mut sum = self[i_offset + j];
                for k in 0..COL2 {
                    sum += b[i_offset2 + k] * c[k * COL + j];
                }
                res[i_offset + j] = sum;
            }
        }
    }

    #[inline(always)]
    /// res = self - b * c
    pub fn sub_mat_mul_into<const COL2: usize, B, C, DST>(
        &self,
        b: &B,
        c: &C,
        res: &mut DST
    )
    where
        B: Container<ROW, COL2>,
        C: Container<COL2, COL>,
        DST: Container<ROW, COL>,
    {
        for i in 0..ROW {
            let i_offset = i * COL;
            let i_offset2 = i * COL2;
            let mut j = 0;
            while j + 4 <= COL {
                unsafe {
                    let mut acc = vld1q_f64_x2(self.ptr(i_offset + j));
                    for k in 0..COL2 {
                        let b_vec = vdupq_n_f64(b[i_offset2 + k]);
                        let c_vec = vld1q_f64_x2(c.ptr(k * COL + j));
                        acc.0 = vfmsq_f64(acc.0, b_vec, c_vec.0);
                        acc.1 = vfmsq_f64(acc.1, b_vec, c_vec.1);
                    }
                    vst1q_f64_x2(res.ptr_mut(i_offset + j), acc);
                }
                j += 4;
            }
            if j + 2 <= COL {
                unsafe {
                    let mut acc = vld1q_f64(self.ptr(i_offset + j));
                    for k in 0..COL2 {
                        let b_vec = vdupq_n_f64(b[i_offset2 + k]);
                        let c_vec = vld1q_f64(c.ptr(k * COL + j));
                        acc = vfmsq_f64(acc, b_vec, c_vec);
                    }
                    vst1q_f64(res.ptr_mut(i_offset + j), acc);
                }
                j += 2;
            }
            if j < COL {
                let mut sum = self[i_offset + j];
                for k in 0..COL2 {
                    sum -= b[i_offset2 + k] * c[k * COL + j];
                }
                res[i_offset + j] = sum;
            }
        }
    }

    #[inline(always)]
    /// res = I - self * b
    pub fn identity_sub_mat_mul_into<const LEN2: usize, B, DST>(
        &self,
        b: &B,
        dst: &mut DST
    )
    where
        B: Container<COL, ROW>,
        DST: Container<ROW, ROW>,
    {
        for i in 0..ROW {
            let i_offset = i * COL;
            let row_offset = i * ROW;
            let mut j = 0;
            let mut identity_row = [0f64; ROW];
            identity_row[i] = 1.0;
            let identity = identity_row.as_ptr();

            while j + 4 <= ROW {
                let mut acc = unsafe { vld1q_f64_x2(identity.add(j)) };
                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i_offset + k]);
                        let b_vec = vld1q_f64_x2(b.ptr(k * ROW + j));
                        acc.0 = vfmsq_f64(acc.0, a_vec, b_vec.0);
                        acc.1 = vfmsq_f64(acc.1, a_vec, b_vec.1);
                    }
                }
                unsafe { vst1q_f64_x2(dst.ptr_mut(row_offset + j), acc) }
                j += 4;
            }

            if j + 2 <= ROW {
                let mut acc = unsafe { vld1q_f64(identity.add(j)) };
                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i_offset + k]);
                        let b_vec = vld1q_f64(b.ptr(k * ROW + j));
                        acc = vfmsq_f64(acc, a_vec, b_vec);
                    }
                }
                unsafe { vst1q_f64(dst.ptr_mut(row_offset + j), acc) }
                j += 2;
            }

            if j < ROW {
                let mut sum = unsafe { *identity.add(j) };
                for k in 0..COL {
                    sum -= self[i_offset + k] * b[k * ROW + j];
                }
                dst[row_offset + j] = sum;
            }
        }
    }

    #[inline(always)]
    pub fn mat_vec_mul_into(&self, rhs: &VecF<COL>, res: &mut VecF<ROW>) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc0 = unsafe { vdupq_n_f64(0.0) };
            let mut acc1 = unsafe { vdupq_n_f64(0.0) };

            let mut j = 0;

            while j + 4 <= COL {
                unsafe {
                    let m_vec = vld1q_f64_x2(self.ptr(offset + j));
                    let r_vec = vld1q_f64_x2(rhs.ptr(j));
                    acc0 = vfmaq_f64(acc0, m_vec.0, r_vec.0);
                    acc1 = vfmaq_f64(acc1, m_vec.1, r_vec.1);
                }
                j += 4;
            }

            let mut acc = unsafe { vaddq_f64(acc0, acc1) };

            if j + 2 <= COL {
                unsafe {
                    let m_vec = vld1q_f64(self.ptr(offset + j));
                    let r_vec = vld1q_f64(rhs.ptr(j));
                    acc = vfmaq_f64(acc, m_vec, r_vec);
                }
                j += 2;
            }

            let mut dot = unsafe { vaddvq_f64(acc) };

            if j < COL {
                dot += self[offset + j] * rhs[j];
            }

            res[i] = dot;
        }
    }

    #[inline(always)]
    pub fn mat_add_into<C, DST>(&self, rhs: &C, res: &mut DST)
    where
        C: Container<ROW, COL>,
        DST: Container<ROW, COL>,
    {
        for row in 0..ROW {
            let offset = row * COL;

            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.ptr(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.ptr(offset + col));
                    vst1q_f64_x2(res.ptr_mut(offset + col), float64x2x2_t(
                        vaddq_f64(lhs_vec.0, rhs_vec.0),
                        vaddq_f64(lhs_vec.1, rhs_vec.1),
                    ));
                }
                col += 4;
            }
            if col + 2 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64(self.ptr(offset + col));
                    let rhs_vec = vld1q_f64(rhs.ptr(offset + col));
                    vst1q_f64(res.ptr_mut(offset + col), vaddq_f64(lhs_vec, rhs_vec));
                }
                col += 2;
            }
            if col < COL {
                res[offset + col] = self[offset + col] + rhs[offset + col];
            }
        }
    }

    #[inline(always)]
    pub fn mat_sub_into<C, DST>(&self, rhs: &C, res: &mut DST)
    where
        C: Container<ROW, COL>,
        DST: Container<ROW, COL>,
    {
        for row in 0..ROW {
            let offset = row * COL;
            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.ptr(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.ptr(offset + col));
                    vst1q_f64_x2(res.ptr_mut(offset + col), float64x2x2_t(
                        vsubq_f64(lhs_vec.0, rhs_vec.0),
                        vsubq_f64(lhs_vec.1, rhs_vec.1),
                    ));
                }
                col += 4;
            }
            if col + 2 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64(self.ptr(offset + col));
                    let rhs_vec = vld1q_f64(rhs.ptr(offset + col));
                    vst1q_f64(res.ptr_mut(offset + col), vsubq_f64(lhs_vec, rhs_vec));
                }
                col += 2;
            }
            if col < COL {
                res[offset + col] = self[offset + col] - rhs[offset + col];
            }
        }
    }
}

// =============================================================================
// Arithmetic Non-Neon
// =============================================================================

#[cfg(not(target_feature = "neon"))]
impl<const LEN: usize, const ROW: usize, const COL: usize> Matrix<LEN, ROW, COL> {
    #[inline(always)]
    pub fn mat_mul_into<const COL2: usize, RHS, DST>(
        &self,
        rhs: &RHS,
        res: &mut DST
    )
    where
        RHS: Container<COL, COL2>,
        DST: Container<ROW, COL2>,
    {
        *result = Matrix::ZERO;
        for i in 0..ROW {
            for k in 0..COL {
                let a_val = self[i * COL + k];
                for j in 0..COL2 {
                    let b_val = rhs[k * COL2 + j];
                    result[i * COL2 + j] += a_val * b_val;
                }
            }
        }
    }

    #[inline(always)]
    /// res = self + b * c
    pub fn add_mat_mul_into<const COL2: usize, B, C, DST>(
        &self,
        b: &B,
        c: &C,
        res: &mut DST
    )
    where
        B: Container<ROW, COL2>,
        C: Container<COL2, COL>,
        DST: Container<ROW, COL>,
    {
        for i in 0..ROW {
            let i_offset = i * COL;
            let i_offset2 = i * COL2;
            for j in 0..COL {
                let mut acc = self[i_offset + j];
                for k in 0..COL2 {
                    acc += b[i_offset2 + k] * c[k * COL + j];
                }
                res[i_offset + j] = acc;
            }
        }
    }

    #[inline(always)]
    /// res = self - b * c
    pub fn sub_mat_mul_into<const COL2: usize, B, C, DST>(
        &self,
        b: &B,
        c: &C,
        res: &mut DST
    )
    where
        B: Container<ROW, COL2>,
        C: Container<COL2, COL>,
        DST: Container<ROW, COL>,
    {
        for i in 0..ROW {
            let i_offset = i * COL;
            let i_offset2 = i * COL2;
            for j in 0..COL {
                let mut acc = self[i_offset + j];
                for k in 0..COL2 {
                    acc -= b[i_offset2 + k] * c[k * COL + j];
                }
                res[i_offset + j] = acc;
            }
        }
    }

    #[inline(always)]
    /// res = I - self * b
    pub fn identity_sub_mat_mul_into<const LEN2: usize, B, DST>(
        &self,
        b: &B,
        dst: &mut DST
    )
    where
        B: Container<COL, ROW>,
        DST: Container<ROW, ROW>,
    {
        for i in 0..ROW {
            let i_offset = i * COL;
            let mut identity = [0f64; ROW];
            identity[i] = 1.0;
            for j in 0..ROW {
                let mut sum = identity[j];
                for k in 0..COL {
                    sum -= self[i_offset + k] * b[k * ROW + j];
                }
                dst[i * ROW + j] = sum;
            }
        }
    }

    #[inline(always)]
    pub fn mat_vec_mul_into(&self, rhs: &VecF<COL>, res: &mut VecF<ROW>) {
        for row in 0..ROW {
            let offset = row * COL;
            let mut acc = 0.0;
            for col in 0..COL {
                acc += mat[offset + col] * rhs[col]
            }
            res[row] = acc
        }
    }

    #[inline(always)]
    pub fn mat_add_into<C, DST>(&self, rhs: &C, res: &mut DST)
    where
        C: Container<ROW, COL>,
        DST: Container<ROW, COL>,
    {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                res[offset + col] = self[offset + col] + rhs[offset + col]
            }
        }
    }

    #[inline(always)]
    pub fn mat_sub_into<C, DST>(&self, rhs: &C, res: &mut DST)
    where
        C: Container<ROW, COL>,
        DST: Container<ROW, COL>,
    {
        let mut result = Self::ZERO;
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                res[offset + col] = self[offset + col] - rhs[offset + col];
            }
        }
    }
}

// =============================================================================
// Identity is only available on square matrix
// =============================================================================

impl<const LEN: usize, const N: usize> Matrix<LEN, N> {
    pub const IDENTITY: Self = {
        let mut this = Self::ZERO;
        let mut i = 0;
        while i < N {
            this.data[i * N + i] = 1.0;
            i += 1;
        }
        this
    };
}

// =============================================================================
// Index / IndexMut
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Index<usize>
    for Matrix<LEN, ROW, COL>
{
    type Output = f64;
    fn index(&self, index: usize) -> &f64 {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::IndexMut<usize>
    for Matrix<LEN, ROW, COL>
{
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize>
    core::ops::Index<core::ops::Range<usize>> for Matrix<LEN, ROW, COL>
{
    type Output = [f64];
    fn index(&self, index: core::ops::Range<usize>) -> &[f64] {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize>
    core::ops::IndexMut<core::ops::Range<usize>> for Matrix<LEN, ROW, COL>
{
    fn index_mut(&mut self, index: core::ops::Range<usize>) -> &mut [f64] {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

// =============================================================================
// IntoIterator
// =============================================================================

impl<'a, const LEN: usize, const ROW: usize, const COL: usize> IntoIterator
    for &'a Matrix<LEN, ROW, COL>
{
    type Item = &'a [f64];
    type IntoIter = core::slice::ChunksExact<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.rows()
    }
}

impl<'a, const LEN: usize, const ROW: usize, const COL: usize> IntoIterator
    for &'a mut Matrix<LEN, ROW, COL>
{
    type Item = &'a mut [f64];
    type IntoIter = core::slice::ChunksExactMut<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.rows_mut()
    }
}

// =============================================================================
// PartialEq
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> PartialEq
    for Matrix<LEN, ROW, COL>
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

// =============================================================================
// Debug: this is highly inefficient but whatever, not important
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> std::fmt::Debug
    for Matrix<LEN, ROW, COL>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Matrix<{ROW}×{COL}>")?;
        let mut max_len = 0;
        let s = self.rows().map(|row| {
            row.iter().map(|val| {
                let space = if val.is_sign_negative() { 0 } else { 1 };
                let st = format!(" {:space$}{val:.2e} ", "");
                let len = st.len();
                if len > max_len { max_len = len }
                st
            })
            .collect::<Box<[_]>>()
        })
        .collect::<Box<[_]>>();
        s.iter().try_for_each(|row| {
            write!(f, " │")?;
            row.iter().try_for_each(|val| {
                let len = max_len - val.len();
                write!(f, "{val}{:len$}", "")
            })?;
            writeln!(f, " │")
        })?;
        Ok(())
    }
}
