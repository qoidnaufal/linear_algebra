use core::ops::Range;
#[cfg(target_feature = "neon")]
use core::arch::aarch64::*;

use crate::vector::VecF;
use crate::traits::Container;

#[repr(align(32))]
pub struct HeapMatrix<const ROW: usize, const COL: usize = ROW> {
    pub data: Box<[f64]>,
}

impl<const ROW: usize, const COL: usize>
Container<ROW, COL> for HeapMatrix<ROW, COL> {
    fn ptr(&self, offset: usize) -> *const f64 {
        unsafe { self.data.as_ptr().add(offset) }
    }

    fn ptr_mut(&mut self, offset: usize) -> *mut f64 {
        unsafe { self.data.as_mut_ptr().add(offset) }
    }
}

impl<const ROW: usize, const COL: usize> HeapMatrix<ROW, COL> {
    pub fn zero() -> Self {
        Self {
            data: vec![0f64; ROW * COL].into_boxed_slice(),
        }
    }

    pub const fn len(&self) -> usize {
        ROW * COL
    }

    pub const fn data_size() -> usize {
        ROW * COL * core::mem::size_of::<f64>()
    }

    pub const fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub const fn as_slice_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn ptr(&self, offset: usize) -> *const f64 {
        unsafe { self.data.as_ptr().add(offset) }
    }

    pub fn ptr_mut(&mut self, offset: usize) -> *mut f64 {
        unsafe { self.data.as_mut_ptr().add(offset) }
    }

    pub fn row_iter(&self) -> core::slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(COL)
    }

    pub fn row_iter_mut(&mut self) -> core::slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(COL)
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [f64], &mut [f64]) {
        unsafe { self.data.split_at_mut_unchecked(mid) }
    }

    pub fn transpose_into(&self, res: &mut HeapMatrix<COL, ROW>) {
        for r_tile in (0..ROW).step_by(8) {
            for c_tile in (0..COL).step_by(8) {
                let r_end = (r_tile + 8).min(ROW);
                let c_end = (c_tile + 8).min(COL);
                for r in r_tile..r_end {
                    let r_offset = r * COL;
                    for c in c_tile..c_end {
                        res[c * ROW + r] = self[r_offset + c];
                    }
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }

    pub fn copy_data(&mut self, src: &Self) {
        self.data.copy_from_slice(&src.data);
    }

    pub fn copy_from_slice(&mut self, src: &[f64]) {
        debug_assert_eq!(src.len(), ROW * COL);
        self.data.copy_from_slice(src);
    }

    pub fn copy_rows(&mut self, src: &Self, start: usize, end: usize) {
        let start = start * COL;
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.ptr(start),
                self.ptr_mut(start),
                end * COL - start
            );
        }
    }

    pub fn copy_block<const ROW2: usize, const COL2: usize>(
        &mut self,
        src: &HeapMatrix<ROW2, COL2>,
        dst_row_start: usize,
        dst_col_start: usize,
        src_row_start: usize,
        src_col_start: usize,
    ) {
        let col_count = (COL - dst_col_start).min(COL2 - src_col_start);
        let row_count = (ROW - dst_row_start).min(ROW2 - src_row_start);

        for i in 0..row_count {
            let dst_row = (i + dst_row_start) * COL + dst_col_start;
            let src_row = (i + src_row_start) * COL2 + src_col_start;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src.ptr(src_row),
                    self.ptr_mut(dst_row),
                    col_count
                );
            }
        }
    }
}

// =============================================================================
// Matrix Arithmetic Neon
// =============================================================================

#[cfg(target_feature = "neon")]
impl<const ROW: usize, const COL: usize> HeapMatrix<ROW, COL> {
    pub fn add_rows<SRC>(&mut self, src: &SRC, start: usize, end: usize)
    where
        SRC: Container<ROW, COL>
    {
        for i in start..end {
            let row = i * COL;
            let mut j = 0;
            while j + 4 <= COL {
                unsafe {
                    let dst_ptr = self.ptr_mut(row + j);
                    let dst_vec = vld1q_f64_x2(dst_ptr);
                    let src_vec = vld1q_f64_x2(src.ptr(row + j));
                    vst1q_f64_x2(dst_ptr, float64x2x2_t(
                        vaddq_f64(dst_vec.0, src_vec.0),
                        vaddq_f64(dst_vec.1, src_vec.1),
                    ));
                }
                j += 4;
            }
            while j < COL {
                self[row + j] += src[row + j];
                j += 1;
            }
        }
    }

    pub fn add_block<const ROW2: usize, const COL2: usize, SRC>(
        &mut self,
        src: &SRC,
        dst_row_start: usize,
        dst_col_start: usize,
        src_row_start: usize,
        src_col_start: usize,
    )
    where
        SRC: Container<ROW2, COL2>
    {
        let col_count = (COL - dst_col_start).min(COL2 - src_col_start);
        let row_count = (ROW - dst_row_start).min(ROW2 - src_row_start);

        for i in 0..row_count {
            let dst_row = (i + dst_row_start) * COL + dst_col_start;
            let src_row = (i + src_row_start) * COL2 + src_col_start;

            let mut j = 0;
            while j + 4 <= col_count {
                unsafe {
                    let dst_ptr = self.ptr_mut(dst_row + j);
                    let dst_vec = vld1q_f64_x2(dst_ptr);
                    let src_vec = vld1q_f64_x2(src.ptr(src_row + j));
                    vst1q_f64_x2(dst_ptr, float64x2x2_t(
                        vaddq_f64(dst_vec.0, src_vec.0),
                        vaddq_f64(dst_vec.1, src_vec.1),
                    ));
                }
                j += 4;
            }
            while j < col_count {
                self[dst_row + j] += src[src_row + j];
                j += 1
            }
        }
    }

    pub fn add_from_slice(&mut self, src: &[f64]) {
        debug_assert_eq!(src.len(), ROW * COL);

        for i in 0..ROW {
            let offset = i * COL;

            let mut j = 0;
            while j + 4 <= COL {
                unsafe {
                    let dst_vec = vld1q_f64_x2(self.ptr(offset + j));
                    let src_vec = vld1q_f64_x2(src.as_ptr().add(offset + j));
                    vst1q_f64_x2(
                        self.ptr_mut(offset + j),
                        float64x2x2_t(
                            vaddq_f64(dst_vec.0, src_vec.0),
                            vaddq_f64(dst_vec.1, src_vec.1),
                        )
                    );
                }
                j += 4;
            }
            if j + 2 <= COL {
                unsafe {
                    let dst_vec = vld1q_f64(self.ptr(offset + j));
                    let src_vec = vld1q_f64(src.as_ptr().add(offset + j));
                    vst1q_f64(self.ptr_mut(offset + j), vaddq_f64(dst_vec, src_vec));
                }
                j += 2;
            }
            if j < COL {
                self[offset + j] += src[offset + j];
            }
        }
    }

    #[inline]
    pub fn mat_mul_into<const COL2: usize, RHS, DST>(&self, rhs: &RHS, dst: &mut DST)
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
                unsafe { vst1q_f64_x2(dst.ptr_mut(i_offset2 + j), float64x2x2_t(acc0, acc1)) }
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
                unsafe { vst1q_f64(dst.ptr_mut(i_offset2 + j), acc) }
                j += 2;
            }

            if j < COL2 {
                let mut sum = 0.0;
                for k in 0..COL {
                    sum += self[i_offset + k] * rhs[k * COL2 + j];
                }
                dst[i_offset2 + j] = sum;
            }
        }
    }

    #[inline]
    /// dst = self * b + c
    pub fn mat_mul_add_into<const COL2: usize, RHS, DST>(&self, b: &RHS, c: &DST, dst: &mut DST)
    where
        RHS: Container<COL, COL2>,
        DST: Container<ROW, COL2>,
    {
        for i in 0..ROW {
            let i_col2 = i * COL2;
            let i_col = i * COL;
            let mut j = 0;

            while j + 4 <= COL2 {
                let mut acc = unsafe { vld1q_f64_x2(c.ptr(i_col2 + j)) };
                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i_col + k]);
                        let b_vec = vld1q_f64_x2(b.ptr(k * COL2 + j));
                        acc.0 = vfmaq_f64(acc.0, a_vec, b_vec.0);
                        acc.1 = vfmaq_f64(acc.1, a_vec, b_vec.1);
                    }
                }
                unsafe { vst1q_f64_x2(dst.ptr_mut(i_col2 + j), acc) }
                j += 4;
            }

            while j + 2 <= COL2 {
                let mut acc = unsafe { vld1q_f64(c.ptr(i_col2 + j)) };
                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i_col + k]);
                        let b_vec = vld1q_f64(b.ptr(k * COL2 + j));
                        acc = vfmaq_f64(acc, a_vec, b_vec);
                    }
                }
                unsafe { vst1q_f64(dst.ptr_mut(i_col2 + j), acc) }
                j += 2;
            }

            if j < COL2 {
                let mut sum = c[i_col2 + j];
                for k in 0..COL {
                    sum += self[i_col + k] * b[k * COL2 + j];
                }
                dst[i * COL2 + j] = sum;
            }
        }
    }

    #[inline]
    /// dst = self * rhs - c
    pub fn mat_mul_sub_into<const COL2: usize, RHS, DST>(&self, rhs: &RHS, c: &DST, dst: &mut DST)
    where
        RHS: Container<COL, COL2>,
        DST: Container<ROW, COL2>,
    {
        for i in 0..ROW {
            let i_col2 = i * COL2;
            let mut j = 0;

            while j + 4 <= COL2 {
                let mut acc = unsafe { vld1q_f64_x2(c.ptr(i_col2 + j)) };
                acc.0 = unsafe { vnegq_f64(acc.0) };
                acc.1 = unsafe { vnegq_f64(acc.1) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i * COL + k]);
                        let b_vec = vld1q_f64_x2(rhs.ptr(k * COL2 + j));
                        acc.0 = vfmaq_f64(acc.0, a_vec, b_vec.0);
                        acc.1 = vfmaq_f64(acc.1, a_vec, b_vec.1);
                    }
                }

                unsafe {
                    vst1q_f64_x2(dst.ptr_mut(i_col2 + j), acc);
                }

                j += 4;
            }

            while j + 2 <= COL2 {
                let mut acc = unsafe { vld1q_f64(c.ptr(i_col2 + j)) };
                acc = unsafe { vnegq_f64(acc) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i * COL + k]);
                        let b_vec = vld1q_f64(rhs.ptr(k * COL2 + j));
                        acc = vfmaq_f64(acc, a_vec, b_vec);
                    }
                }

                unsafe {
                    vst1q_f64(dst.ptr_mut(i_col2 + j), acc);
                }

                j += 2;
            }

            while j < COL2 {
                let mut sum = 0.0;
                for k in 0..COL {
                    sum += self[i * COL + k] * rhs[k * COL2 + j];
                }
                dst[i * COL2 + j] = sum - c[i * COL2 + j];
                j += 1;
            }
        }
    }

    #[inline]
    /// res = self - b * c
    pub fn sub_mat_mul_into<const COL2: usize, B, C, DST>(&self, b: &B, c: &C, res: &mut DST)
    where
        B: Container<ROW, COL2>,
        C: Container<COL2, COL>,
        DST: Container<ROW, COL>,
    {
        for i in 0..ROW {
            let i_col = i * COL;
            let i_col2 = i * COL2;
            let mut j = 0;

            while j + 4 <= COL {
                let mut acc = unsafe { vld1q_f64_x2(self.ptr(i_col + j)) };
                for k in 0..COL2 {
                    unsafe {
                        let b_vec = vdupq_n_f64(b[i_col2 + k]);
                        let c_vec = vld1q_f64_x2(c.ptr(k * COL + j));
                        acc.0 = vfmsq_f64(acc.0, b_vec, c_vec.0);
                        acc.1 = vfmsq_f64(acc.1, b_vec, c_vec.1);
                    }
                }
                unsafe { vst1q_f64_x2(res.ptr_mut(i_col + j), acc) }
                j += 4;
            }

            if j + 2 <= COL {
                let mut acc = unsafe { vld1q_f64(self.ptr(i_col + j)) };
                for k in 0..COL2 {
                    unsafe {
                        let b_vec = vdupq_n_f64(b[i_col2 + k]);
                        let c_vec = vld1q_f64(c.ptr(k * COL + j));
                        acc = vfmsq_f64(acc, b_vec, c_vec);
                    }
                }
                unsafe { vst1q_f64(res.ptr_mut(i_col + j), acc) }
                j += 2;
            }

            if j < COL {
                let mut sum = self[i_col + j];
                for k in 0..COL2 {
                    sum -= b[i_col2 + k] * c[k * COL + j];
                }
                res[i_col + j] = sum;
            }
        }
    }

    #[inline]
    /// res = I - self * b
    pub fn identity_sub_mat_mul_into<RHS, DST>(&self, rhs: &RHS, dst: &mut DST)
    where
        RHS: Container<COL, ROW>,
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
                        let b_vec = vld1q_f64_x2(rhs.ptr(k * ROW + j));
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
                        let b_vec = vld1q_f64(rhs.ptr(k * ROW + j));
                        acc = vfmsq_f64(acc, a_vec, b_vec);
                    }
                }
                unsafe { vst1q_f64(dst.ptr_mut(row_offset + j), acc) }
                j += 2;
            }

            if j < ROW {
                let mut sum = unsafe { *identity.add(j) };
                for k in 0..COL {
                    sum -= self[i_offset + k] * rhs[k * ROW + j];
                }
                dst[row_offset + j] = sum;
            }
        }
    }

    #[inline]
    pub fn mat_add_into<RHS, DST>(&self, rhs: &RHS, res: &mut DST)
    where
        RHS: Container<ROW, COL>,
        DST: Container<ROW, COL>
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
            while col + 2 <= COL {
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

    pub fn mat_add_assign<SRC: Container<ROW, COL>>(&mut self, rhs: &SRC) {
        for row in 0..ROW {
            let offset = row * COL;
            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.ptr(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.ptr(offset + col));
                    vst1q_f64_x2(self.ptr_mut(offset + col), float64x2x2_t(
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
                    vst1q_f64(self.ptr_mut(offset + col), vaddq_f64(lhs_vec, rhs_vec));
                }
                col += 2;
            }
            if col < COL {
                self[offset + col] += rhs[offset + col];
            }
        }
    }

    pub fn mat_sub_into(&self, rhs: &Self, res: &mut Self) {
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

    // self = self - rhs
    pub fn mat_sub_assign<RHS: Container<ROW, COL>>(&mut self, rhs: &RHS) {
        for row in 0..ROW {
            let offset = row * COL;
            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.ptr(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.ptr(offset + col));
                    vst1q_f64_x2(self.ptr_mut(offset + col), float64x2x2_t(
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
                    vst1q_f64(self.ptr_mut(offset + col), vsubq_f64(lhs_vec, rhs_vec));
                }
                col += 2;
            }
            if col < COL {
                self[offset + col] -= rhs[offset + col];
            }
        }
    }

    // self = rhs - self
    pub fn mat_sub_from(&mut self, rhs: &Self) {
        for row in 0..ROW {
            let offset = row * COL;

            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.ptr(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.ptr(offset + col));
                    vst1q_f64_x2(self.ptr_mut(offset + col), float64x2x2_t(
                        vsubq_f64(rhs_vec.0, lhs_vec.0),
                        vsubq_f64(rhs_vec.1, lhs_vec.1),
                    ));
                }
                col += 4;
            }
            if col + 2 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64(self.ptr(offset + col));
                    let rhs_vec = vld1q_f64(rhs.ptr(offset + col));
                    vst1q_f64(self.ptr_mut(offset + col), vsubq_f64(rhs_vec, lhs_vec));
                }
                col += 2;
            }
            if col < COL {
                self[offset + col] = rhs[offset + col] - self[offset + col];
            }
        }
    }

    pub fn scalar_mul_into(&self, scalar: f64, res: &mut Self) {
        let rhs_vec = unsafe { vdupq_n_f64(scalar) };
        for row in 0..ROW {
            let offset = row * COL;
            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.ptr(offset + col));
                    vst1q_f64_x2(res.ptr_mut(offset + col), float64x2x2_t(
                        vmulq_f64(lhs_vec.0, rhs_vec),
                        vmulq_f64(lhs_vec.1, rhs_vec),
                    ));
                }
                col += 4;
            }
            if col + 2 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64(self.ptr(offset + col));
                    vst1q_f64(res.ptr_mut(offset + col), vmulq_f64(lhs_vec, rhs_vec));
                }
                col += 2;
            }
            if col < COL {
                res[offset + col] = self[offset + col] * scalar;
            }
        }
    }

    pub fn scalar_add_into(&self, scalar: f64, res: &mut Self) {
        let rhs_vec = unsafe { vdupq_n_f64(scalar) };
        for row in 0..ROW {
            let offset = row * COL;
            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.ptr(offset + col));
                    vst1q_f64_x2(res.ptr_mut(offset + col), float64x2x2_t(
                        vaddq_f64(lhs_vec.0, rhs_vec),
                        vaddq_f64(lhs_vec.1, rhs_vec),
                    ));
                }
                col += 4;
            }
            if col + 2 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64(self.ptr(offset + col));
                    vst1q_f64(res.ptr_mut(offset + col), vaddq_f64(lhs_vec, rhs_vec));
                }
                col += 2;
            }
            if col < COL {
                res[offset + col] = self[offset + col] + scalar;
            }
        }
    }

    pub fn scalar_sub_into(&self, scalar: f64, res: &mut Self) {
        let rhs_vec = unsafe { vdupq_n_f64(scalar) };
        for row in 0..ROW {
            let offset = row * COL;
            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.ptr(offset + col));
                    vst1q_f64_x2(res.ptr_mut(offset + col), float64x2x2_t(
                        vsubq_f64(lhs_vec.0, rhs_vec),
                        vsubq_f64(lhs_vec.1, rhs_vec),
                    ));
                }
                col += 4;
            }
            if col + 2 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64(self.ptr(offset + col));
                    vst1q_f64(res.ptr_mut(offset + col), vsubq_f64(lhs_vec, rhs_vec));
                }
                col += 2;
            }
            if col < COL {
                res[offset + col] = self[offset + col] - scalar;
            }
        }
    }
}

// =============================================================================
// Matrix Arithmetic Non-Neon
// =============================================================================

#[cfg(not(target_feature = "neon"))]
impl<const ROW: usize, const COL: usize> HeapMatrix<ROW, COL> {
    pub fn add_rows(
        &mut self,
        src: &Self,
        start: usize,
        end: usize,
    ) {
        for i in start..end {
            let row = i * COL;
            for j in 0..COL {
                self[row + j] += src[row + j]
            }
        }
    }

    pub fn add_block<const ROW2: usize, const COL2: usize>(
        &mut self,
        src: &HeapMatrix<ROW2, COL2>,
        dst_row_start: usize,
        dst_col_start: usize,
        src_row_start: usize,
        src_col_start: usize,
    ) {
        let col_count = (COL - dst_col_start).min(COL2 - src_col_start);
        let row_count = (ROW - dst_row_start).min(ROW2 - src_row_start);

        for i in 0..row_count {
            let dst_row = (i + dst_row_start) * COL + dst_col_start;
            let src_row = (i + src_row_start) * COL2 + src_col_start;
            for j in 0..col_count {
                self[dst_row + j] += src[src_row + j]
            }
        }
    }

    pub fn add_from_slice(&mut self, src: &[f64]) {
        debug_assert_eq!(src.len(), ROW * COL);
        for i in 0..ROW * COL {
            self[i] += src[i]
        }
    }

    pub fn mat_mul_into<const COL2: usize>(
        &self,
        rhs: &HeapMatrix<COL, COL2>,
        res: &mut HeapMatrix<ROW, COL2>
    ) {
        for i in 0..ROW {
            let res_offset = i * COL2;
            res.data[res_offset..res_offset + COL2].fill(0.0);

            for k in 0..COL {
                let a_val = self[i * COL + k];
                for j in 0..COL2 {
                    res[res_offset + j] += a_val * rhs[k * COL2 + j];
                }
            }
        }
    }

    /// res = self * rhs + c
    pub fn mat_mul_add_into<const COL2: usize>(
        &self,
        rhs: &HeapMatrix<COL, COL2>,
        c: &HeapMatrix<ROW, COL2>,
        res: &mut HeapMatrix<ROW, COL2>
    ) {
        for i in 0..ROW {
            let res_offset = i * COL2;
            res.data[res_offset..res_offset + COL2]
                .copy_from_slice(&c.data[res_offset..res_offset + COL2]);

            for k in 0..COL {
                let a_val = self[i * COL + k];
                for j in 0..COL2 {
                    res[res_offset + j] += a_val * rhs[k * COL2 + j];
                }
            }
        }
    }

    /// res = self - b * c
    pub fn sub_mat_mul_into<const COL2: usize>(
        &self,
        b: &HeapMatrix<ROW, COL2>,
        c: &HeapMatrix<COL2, COL>,
        res: &mut Self
    ) {
        for i in 0..ROW {
            let res_offset = i * COL;
            res.data[res_offset..res_offset + COL]
                .copy_from_slice(&self.data[res_offset..res_offset + COL]);

            for k in 0..COL2 {
                let b_val = b[i * COL2 + k];
                for j in 0..COL {
                    res[res_offset + j] -= b_val * c[k * COL + j];
                }
            }
        }
    }

    /// res = self * rhs - c
    pub fn mat_mul_sub_into<const COL2: usize>(
        &self,
        rhs: &HeapMatrix<COL, COL2>,
        c: &HeapMatrix<ROW, COL2>,
        res: &mut HeapMatrix<ROW, COL2>
    ) {
        for i in 0..ROW {
            let res_offset = i * COL2;
            for j in 0..COL2 {
                res[res_offset + j] = -c[res_offset + j];
            }
            for k in 0..COL {
                let a_val = self[i * COL + k];
                for j in 0..COL2 {
                    res[res_offset + j] += a_val * rhs[k * COL2 + j];
                }
            }
        }
    }

    pub fn mat_add_into(&self, rhs: &Self, res: &mut Self) {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                res[offset + col] = self[offset + col] + rhs[offset + col]
            }
        }
    }

    pub fn mat_add_assign<RHS: Container<ROW, COL>>(&mut self, rhs: &RHS) {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                self[offset + col] += rhs[offset + col]
            }
        }
    }

    pub fn mat_sub_into(&self, rhs: &Self, res: &mut Self) {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                res[offset + col] = self[offset + col] - rhs[offset + col]
            }
        }
    }

    pub fn mat_sub_assign<RHS: Container<ROW, COL>>(&mut self, rhs: &RHS) {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                self[offset + col] -= rhs[offset + col]
            }
        }
    }

    pub fn mat_sub_from(&mut self, rhs: &Self) {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                self[offset + col] = rhs[offset + col] - self[offset + col]
            }
        }
    }

    pub fn scalar_mul_into(&self, scalar: f64, res: &mut Self) {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                res[offset + col] = self[offset + col] * scalar
            }
        }
    }

    pub fn scalar_add_into(&self, scalar: f64, res: &mut Self) {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                res[offset + col] = self[offset + col] + scalar
            }
        }
    }

    pub fn scalar_sub_into(&self, scalar: f64, res: &mut Self) {
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                res[offset + col] = self[offset + col] - scalar
            }
        }
    }
}

// =============================================================================
// Square M
// =============================================================================

impl<const N: usize> HeapMatrix<N> {
    pub fn identity() -> Self {
        let mut this = Self::zero();
        for i in 0..N {
            this[i * N + i] = 1.0;
        }
        this
    }
}

// =============================================================================
// Matrix-Vector Arithmetic Neon
// =============================================================================

#[cfg(target_feature = "neon")]
impl<const ROW: usize, const COL: usize> HeapMatrix<ROW, COL> {
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

            while j + 2 <= COL {
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

    pub fn mat_vec_mul_scalar_into(&self, rhs: &VecF<COL>, scalar: f64, res: &mut VecF<ROW>) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc0 = unsafe { vdupq_n_f64(0.0) };
            let mut acc1 = unsafe { vdupq_n_f64(0.0) };

            let mut j = 0;

            while j + 4 <= COL {
                unsafe {
                    let m_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + j));
                    let r_vec = vld1q_f64_x2(rhs.data.as_ptr().add(j));
                    acc0 = vfmaq_f64(acc0, m_vec.0, r_vec.0);
                    acc1 = vfmaq_f64(acc1, m_vec.1, r_vec.1);
                }
                j += 4;
            }

            let mut acc = unsafe { vaddq_f64(acc0, acc1) };

            while j + 2 <= COL {
                unsafe {
                    let m_vec = vld1q_f64(self.data.as_ptr().add(offset + j));
                    let r_vec = vld1q_f64(rhs.data.as_ptr().add(j));
                    acc = vfmaq_f64(acc, m_vec, r_vec);
                }
                j += 2;
            }

            let mut dot = unsafe { vaddvq_f64(acc) };

            if j < COL {
                dot += self[offset + j] * rhs[j];
            }

            res[i] = dot * scalar;
        }
    }

    /// res = c + self * rhs
    pub fn mat_vec_mul_add_into(
        &self,
        rhs: &VecF<COL>,
        c: &VecF<ROW>,
        res: &mut VecF<ROW>
    ) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc0 = unsafe { vdupq_n_f64(0.0) };
            let mut acc1 = unsafe { vdupq_n_f64(0.0) };

            let mut j = 0;

            while j + 4 <= COL {
                unsafe {
                    let m_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + j));
                    let r_vec = vld1q_f64_x2(rhs.data.as_ptr().add(j));
                    acc0 = vfmaq_f64(acc0, m_vec.0, r_vec.0);
                    acc1 = vfmaq_f64(acc1, m_vec.1, r_vec.1);
                }
                j += 4;
            }

            let mut dot = unsafe { vaddvq_f64(vaddq_f64(acc0, acc1)) };

            for k in j..COL {
                dot += self[offset + k] * rhs[k];
            }

            res[i] = dot + c[i];
        }
    }

    /// res = self * rhs - c
    pub fn mat_vec_mul_sub_into(
        &self,
        rhs: &VecF<COL>,
        c: &VecF<ROW>,
        res: &mut VecF<ROW>
    ) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc0 = unsafe { vdupq_n_f64(0.0) };
            let mut acc1 = unsafe { vdupq_n_f64(0.0) };

            let mut j = 0;

            while j + 4 <= COL {
                unsafe {
                    let m_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + j));
                    let r_vec = vld1q_f64_x2(rhs.data.as_ptr().add(j));
                    acc0 = vfmaq_f64(acc0, m_vec.0, r_vec.0);
                    acc1 = vfmaq_f64(acc1, m_vec.1, r_vec.1);
                }
                j += 4;
            }

            let mut acc = unsafe { vaddq_f64(acc0, acc1) };

            while j + 2 <= COL {
                unsafe {
                    let m_vec = vld1q_f64(self.data.as_ptr().add(offset + j));
                    let r_vec = vld1q_f64(rhs.data.as_ptr().add(j));
                    acc = vfmaq_f64(acc, m_vec, r_vec);
                }
                j += 2;
            }

            let mut dot = unsafe { vaddvq_f64(acc) };

            if j < COL {
                dot += self[offset + j] * rhs[j];
            }

            res[i] = dot - c[i];
        }
    }
}

// =============================================================================
// Matrix-Vector Arithmetic Non-Neon
// =============================================================================

#[cfg(not(target_feature = "neon"))]
impl<const ROW: usize, const COL: usize> HeapMatrix<ROW, COL> {
    pub fn mat_vec_mul_into(&self, rhs: &VecF<COL>, res: &mut VecF<ROW>) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc = 0.0;
            for j in 0..COL {
                acc += self[offset + j] * rhs[j];
            }
            res[i] = acc;
        }
    }

    pub fn mat_vec_mul_scalar_into(&self, rhs: &VecF<COL>, scalar: f64, res: &mut VecF<ROW>) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc = 0.0;
            for j in 0..COL {
                acc += self[offset + j] * rhs[j];
            }
            res[i] = acc * scalar;
        }
    }

    /// res = c + self * rhs
    pub fn mat_vec_mul_add_into(
        &self,
        rhs: &VecF<COL>,
        c: &VecF<ROW>,
        res: &mut VecF<ROW>
    ) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc = c[i];
            for j in 0..COL {
                acc += self[offset + j] * rhs[j];
            }
            res[i] = acc;
        }
    }

    /// res = self * rhs - c
    pub fn mat_vec_mul_sub_into(
        &self,
        rhs: &VecF<COL>,
        c: &VecF<ROW>,
        res: &mut VecF<ROW>
    ) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc = 0.0;
            for j in 0..COL {
                acc += self[offset + j] * rhs[j];
            }
            res[i] = acc - c[i];
        }
    }
}

// =============================================================================
// Index / IndexMut
// =============================================================================

impl<const ROW: usize, const COL: usize> core::ops::Index<usize> for HeapMatrix<ROW, COL> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<const ROW: usize, const COL: usize> core::ops::IndexMut<usize> for HeapMatrix<ROW, COL> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<const ROW: usize, const COL: usize> core::ops::Index<Range<usize>> for HeapMatrix<ROW, COL> {
    type Output = [f64];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<const ROW: usize, const COL: usize> core::ops::IndexMut<Range<usize>> for HeapMatrix<ROW, COL> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

// =============================================================================
// PartialEq
// =============================================================================

impl<const ROW: usize, const COL: usize> PartialEq for HeapMatrix<ROW, COL> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

// =============================================================================
// IntoIterator
// =============================================================================

impl<'a, const ROW: usize, const COL: usize> IntoIterator for &'a HeapMatrix<ROW, COL> {
    type Item = &'a [f64];
    type IntoIter = core::slice::ChunksExact<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.row_iter()
    }
}

impl<'a, const ROW: usize, const COL: usize> IntoIterator for &'a mut HeapMatrix<ROW, COL> {
    type Item = &'a mut [f64];
    type IntoIter = core::slice::ChunksExactMut<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.row_iter_mut()
    }
}

// =============================================================================
// Default
// =============================================================================

impl<const ROW: usize, const COL: usize> Default for HeapMatrix<ROW, COL> {
    fn default() -> Self {
        Self::zero()
    }
}

// =============================================================================
// fmt
// =============================================================================

impl<const ROW: usize, const COL: usize> core::fmt::Debug for HeapMatrix<ROW, COL> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HeapMatrix<{ROW}×{COL}>")?;
        let mut max_len = 0;
        let s = self.row_iter().map(|row| {
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
