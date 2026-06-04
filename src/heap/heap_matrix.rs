use core::ops::Range;
#[cfg(target_feature = "neon")]
use core::arch::aarch64::*;

pub struct HeapMatrix<const ROW: usize, const COL: usize = ROW> {
    pub data: Box<[f64]>
}

pub type HeapVector<const N: usize> = HeapMatrix<N, 1>;

impl<const ROW: usize, const COL: usize> HeapMatrix<ROW, COL> {
    pub fn zero() -> Self {
        Self {
            data: vec![0f64; ROW * COL].into_boxed_slice()
        }
    }

    pub const fn request_size() -> usize {
        ROW * COL * core::mem::size_of::<f64>()
    }

    pub const fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn row_iter(&self) -> core::slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(COL)
    }

    pub fn row_iter_mut(&mut self) -> core::slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(COL)
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [f64], &mut [f64]) {
        unsafe {
            self.data.split_at_mut_unchecked(mid)
        }
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
        self.data.copy_from_slice(src);
    }

    pub fn copy_rows(&mut self, src: &Self, start: usize, end: usize) {
        let start = start * COL;
        let end = end * COL;
        unsafe {
            core::ptr::copy_nonoverlapping(
                src[start..end].as_ptr(),
                self[start..end].as_mut_ptr(),
                end - start,
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
                    src[src_row..src_row + col_count].as_ptr(),
                    self[dst_row..dst_row + col_count].as_mut_ptr(),
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
    pub fn add_rows(
        &mut self,
        src: &Self,
        start: usize,
        end: usize,
    ) {
        for i in start..end {
            let row = i * COL;
            let mut j = 0;
            while j + 4 <= COL {
                unsafe {
                    let dst_ptr = self.data.as_mut_ptr().add(row + j);
                    let dst_vec = vld1q_f64_x2(dst_ptr);
                    let src_vec = vld1q_f64_x2(src.data.as_ptr().add(row + j));
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

            let mut j = 0;
            while j + 4 <= col_count {
                unsafe {
                    let dst_ptr = self.data.as_mut_ptr().add(dst_row + j);
                    let dst_vec = vld1q_f64_x2(dst_ptr);
                    let src_vec = vld1q_f64_x2(src.data.as_ptr().add(src_row + j));
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
                    let dst_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + j));
                    let src_vec = vld1q_f64_x2(src.as_ptr().add(offset + j));
                    vst1q_f64_x2(
                        self.data.as_mut_ptr().add(offset + j),
                        float64x2x2_t(
                            vaddq_f64(dst_vec.0, src_vec.0),
                            vaddq_f64(dst_vec.1, src_vec.1),
                        )
                    );
                }
                j += 4;
            }
            while j + 2 <= COL {
                unsafe {
                    let dst_vec = vld1q_f64(self.data.as_ptr().add(offset + j));
                    let src_vec = vld1q_f64(src.as_ptr().add(offset + j));
                    vst1q_f64(
                        self.data.as_mut_ptr().add(offset + j),
                        vaddq_f64(dst_vec, src_vec),
                    );
                }
                j += 2;
            }
            if j < COL {
                self[offset + j] += src[offset + j];
            }
        }
    }

    pub fn mat_mul_into<const COL2: usize>(
        &self,
        rhs: &HeapMatrix<COL, COL2>,
        res: &mut HeapMatrix<ROW, COL2>
    ) {
        for i in 0..ROW {
            let mut j = 0;

            while j + 4 <= COL2 {
                let mut acc0 = unsafe { vdupq_n_f64(0.0) };
                let mut acc1 = unsafe { vdupq_n_f64(0.0) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i * COL + k]);
                        let rhs_vec = vld1q_f64_x2(rhs.data.as_ptr().add(k * COL2 + j));
                        acc0 = vfmaq_f64(acc0, a_vec, rhs_vec.0);
                        acc1 = vfmaq_f64(acc1, a_vec, rhs_vec.1);
                    }
                }

                unsafe {
                    vst1q_f64_x2(
                        res.data.as_mut_ptr().add(i * COL2 + j),
                        float64x2x2_t(acc0, acc1),
                    );
                }

                j += 4;
            }

            while j + 2 <= COL2 {
                let mut acc = unsafe { vdupq_n_f64(0.0) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i * COL + k]);
                        let rhs_vec = vld1q_f64(rhs.data.as_ptr().add(k * COL2 + j));
                        acc = vfmaq_f64(acc, a_vec, rhs_vec);
                    }
                }

                unsafe {
                    vst1q_f64(
                        res.data.as_mut_ptr().add(i * COL2 + j),
                        acc
                    );
                }

                j += 2;
            }

            if j < COL2 {
                let mut sum = 0.0;
                for k in 0..COL {
                    sum += self[i * COL + k] * rhs[k * COL2 + j];
                }
                res[i * COL2 + j] = sum;
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
            let i_col2 = i * COL2;
            let mut j = 0;

            while j + 4 <= COL2 {
                let mut acc = unsafe { vld1q_f64_x2(c.data.as_ptr().add(i_col2 + j)) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i * COL + k]);
                        let b_vec = vld1q_f64_x2(rhs.data.as_ptr().add(k * COL2 + j));
                        acc.0 = vfmaq_f64(acc.0, a_vec, b_vec.0);
                        acc.1 = vfmaq_f64(acc.1, a_vec, b_vec.1);
                    }
                }

                unsafe {
                    vst1q_f64_x2(res.data.as_mut_ptr().add(i_col2 + j), acc);
                }

                j += 4;
            }

            while j + 2 <= COL2 {
                let mut acc = unsafe { vld1q_f64(c.data.as_ptr().add(i_col2 + j)) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i * COL + k]);
                        let b_vec = vld1q_f64(rhs.data.as_ptr().add(k * COL2 + j));
                        acc = vfmaq_f64(acc, a_vec, b_vec);
                    }
                }

                unsafe {
                    vst1q_f64(res.data.as_mut_ptr().add(i_col2 + j), acc);
                }

                j += 2;
            }

            if j < COL2 {
                let mut sum = c[i * COL2 + j];
                for k in 0..COL {
                    sum += self[i * COL + k] * rhs[k * COL2 + j];
                }
                res[i * COL2 + j] = sum;
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
            let i_col2 = i * COL2;
            let mut j = 0;

            while j + 4 <= COL2 {
                let mut acc = unsafe { vld1q_f64_x2(c.data.as_ptr().add(i_col2 + j)) };
                acc.0 = unsafe { vnegq_f64(acc.0) };
                acc.1 = unsafe { vnegq_f64(acc.1) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i * COL + k]);
                        let b_vec = vld1q_f64_x2(rhs.data.as_ptr().add(k * COL2 + j));
                        acc.0 = vfmaq_f64(acc.0, a_vec, b_vec.0);
                        acc.1 = vfmaq_f64(acc.1, a_vec, b_vec.1);
                    }
                }

                unsafe {
                    vst1q_f64_x2(res.data.as_mut_ptr().add(i_col2 + j), acc);
                }

                j += 4;
            }

            while j + 2 <= COL2 {
                let mut acc = unsafe { vld1q_f64(c.data.as_ptr().add(i_col2 + j)) };
                acc = unsafe { vnegq_f64(acc) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(self[i * COL + k]);
                        let b_vec = vld1q_f64(rhs.data.as_ptr().add(k * COL2 + j));
                        acc = vfmaq_f64(acc, a_vec, b_vec);
                    }
                }

                unsafe {
                    vst1q_f64(res.data.as_mut_ptr().add(i_col2 + j), acc);
                }

                j += 2;
            }

            while j < COL2 {
                let mut sum = 0.0;
                for k in 0..COL {
                    sum += self[i * COL + k] * rhs[k * COL2 + j];
                }
                res[i * COL2 + j] = sum - c[i * COL2 + j];
                j += 1;
            }
        }
    }

    /// res = c - self * rhs
    pub fn sub_mat_mul_into<const COL2: usize>(
        &self,
        rhs: &HeapMatrix<COL, COL2>,
        c: &HeapMatrix<ROW, COL2>,
        res: &mut HeapMatrix<ROW, COL2>
    ) {
        for i in 0..ROW {
            let i_col2 = i * COL2;
            let mut j = 0;

            while j + 4 <= COL2 {
                let mut acc = unsafe { vld1q_f64_x2(c.data.as_ptr().add(i_col2 + j)) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(-self[i * COL + k]);
                        let b_vec = vld1q_f64_x2(rhs.data.as_ptr().add(k * COL2 + j));
                        acc.0 = vfmaq_f64(acc.0, a_vec, b_vec.0);
                        acc.1 = vfmaq_f64(acc.1, a_vec, b_vec.1);
                    }
                }

                unsafe {
                    vst1q_f64_x2(res.data.as_mut_ptr().add(i_col2 + j), acc);
                }

                j += 4;
            }

            while j + 2 <= COL2 {
                let mut acc = unsafe { vld1q_f64(c.data.as_ptr().add(i_col2 + j)) };

                for k in 0..COL {
                    unsafe {
                        let a_vec = vdupq_n_f64(-self[i * COL + k]);
                        let b_vec = vld1q_f64(rhs.data.as_ptr().add(k * COL2 + j));
                        acc = vfmaq_f64(acc, a_vec, b_vec);
                    }
                }

                unsafe {
                    vst1q_f64(res.data.as_mut_ptr().add(i_col2 + j), acc);
                }

                j += 2;
            }

            while j < COL2 {
                let mut sum = c[i * COL2 + j];
                for k in 0..COL {
                    sum -= self[i * COL + k] * rhs[k * COL2 + j];
                }
                res[i * COL2 + j] = sum;
                j += 1;
            }
        }
    }

    pub fn mat_add_into(&self, rhs: &Self, res: &mut Self) {
        for row in 0..ROW {
            let offset = row * COL;

            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.data.as_ptr().add(offset + col));
                    vst1q_f64_x2(res.data.as_mut_ptr().add(offset + col), float64x2x2_t(
                        vaddq_f64(lhs_vec.0, rhs_vec.0),
                        vaddq_f64(lhs_vec.1, rhs_vec.1),
                    ));
                }
                col += 4;
            }
            while col + 2 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64(self.data.as_ptr().add(offset + col));
                    let rhs_vec = vld1q_f64(rhs.data.as_ptr().add(offset + col));
                    vst1q_f64(
                        res.data.as_mut_ptr().add(offset + col),
                        vaddq_f64(lhs_vec, rhs_vec),
                    );
                }
                col += 2;
            }
            if col < COL {
                res[offset + col] = self[offset + col] + rhs[offset + col];
            }
        }
    }

    pub fn mat_add_assign(&mut self, rhs: &Self) {
        for row in 0..ROW {
            let offset = row * COL;

            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.data.as_ptr().add(offset + col));
                    vst1q_f64_x2(self.data.as_mut_ptr().add(offset + col), float64x2x2_t(
                        vaddq_f64(lhs_vec.0, rhs_vec.0),
                        vaddq_f64(lhs_vec.1, rhs_vec.1),
                    ));
                }
                col += 4;
            }
            while col < COL {
                self[offset + col] += rhs[offset + col];
                col += 1;
            }
        }
    }

    pub fn mat_sub_into(&self, rhs: &Self, res: &mut Self) {
        for row in 0..ROW {
            let offset = row * COL;
            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.data.as_ptr().add(offset + col));
                    vst1q_f64_x2(res.data.as_mut_ptr().add(offset + col), float64x2x2_t(
                        vsubq_f64(lhs_vec.0, rhs_vec.0),
                        vsubq_f64(lhs_vec.1, rhs_vec.1),
                    ));
                }
                col += 4;
            }
            while col < COL {
                res[offset + col] = self[offset + col] - rhs[offset + col];
                col += 1;
            }
        }
    }

    // self = self - rhs
    pub fn mat_sub_assign(&mut self, rhs: &Self) {
        for row in 0..ROW {
            let offset = row * COL;

            let mut col = 0;
            while col + 4 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.data.as_ptr().add(offset + col));
                    vst1q_f64_x2(self.data.as_mut_ptr().add(offset + col), float64x2x2_t(
                        vsubq_f64(lhs_vec.0, rhs_vec.0),
                        vsubq_f64(lhs_vec.1, rhs_vec.1),
                    ));
                }
                col += 4;
            }
            while col + 2 <= COL {
                unsafe {
                    let lhs_vec = vld1q_f64(self.data.as_ptr().add(offset + col));
                    let rhs_vec = vld1q_f64(rhs.data.as_ptr().add(offset + col));
                    vst1q_f64(
                        self.data.as_mut_ptr().add(offset + col),
                        vsubq_f64(lhs_vec, rhs_vec),
                    );
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
                    let lhs_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + col));
                    let rhs_vec = vld1q_f64_x2(rhs.data.as_ptr().add(offset + col));
                    vst1q_f64_x2(self.data.as_mut_ptr().add(offset + col), float64x2x2_t(
                        vsubq_f64(rhs_vec.0, lhs_vec.0),
                        vsubq_f64(rhs_vec.1, lhs_vec.1),
                    ));
                }
                col += 4;
            }
            while col < COL {
                self[offset + col] = rhs[offset + col] - self[offset + col];
                col += 1;
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
                    let lhs_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + col));
                    vst1q_f64_x2(res.data.as_mut_ptr().add(offset + col), float64x2x2_t(
                        vmulq_f64(lhs_vec.0, rhs_vec),
                        vmulq_f64(lhs_vec.1, rhs_vec),
                    ));
                }
                col += 4;
            }
            while col < COL {
                res[offset + col] = self[offset + col] * scalar;
                col += 1;
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
                    let lhs_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + col));
                    vst1q_f64_x2(res.data.as_mut_ptr().add(offset + col), float64x2x2_t(
                        vaddq_f64(lhs_vec.0, rhs_vec),
                        vaddq_f64(lhs_vec.1, rhs_vec),
                    ));
                }
                col += 4;
            }
            while col < COL {
                res[offset + col] = self[offset + col] + scalar;
                col += 1;
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
                    let lhs_vec = vld1q_f64_x2(self.data.as_ptr().add(offset + col));
                    vst1q_f64_x2(res.data.as_mut_ptr().add(offset + col), float64x2x2_t(
                        vsubq_f64(lhs_vec.0, rhs_vec),
                        vsubq_f64(lhs_vec.1, rhs_vec),
                    ));
                }
                col += 4;
            }
            while col < COL {
                res[offset + col] = self[offset + col] - scalar;
                col += 1;
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

    /// res = c - self * rhs
    pub fn sub_mat_mul_into<const COL2: usize>(
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
                    res[res_offset + j] -= a_val * rhs[k * COL2 + j];
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

    pub fn mat_add_assign(&mut self, rhs: &Self) {
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

    pub fn mat_sub_assign(&mut self, rhs: &Self) {
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
    pub fn mat_vec_mul_into(&self, rhs: &HeapVector<COL>, res: &mut HeapVector<ROW>) {
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

            res[i] = dot;
        }
    }

    pub fn mat_vec_mul_scalar_into(&self, rhs: &HeapVector<COL>, scalar: f64, res: &mut HeapVector<ROW>) {
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
        rhs: &HeapVector<COL>,
        c: &HeapVector<ROW>,
        res: &mut HeapVector<ROW>
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
        rhs: &HeapVector<COL>,
        c: &HeapVector<ROW>,
        res: &mut HeapVector<ROW>
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

    /// res = c - self * rhs
    pub fn sub_mat_vec_mul_into(
        &self,
        rhs: &HeapVector<COL>,
        c: &HeapVector<ROW>,
        res: &mut HeapVector<ROW>
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

            res[i] = c[i] - dot;
        }
    }
}

// =============================================================================
// Matrix-Vector Arithmetic Non-Neon
// =============================================================================

#[cfg(not(target_feature = "neon"))]
impl<const ROW: usize, const COL: usize> HeapMatrix<ROW, COL> {
    pub fn mat_vec_mul_into(&self, rhs: &HeapVector<COL>, res: &mut HeapVector<ROW>) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc = 0.0;
            for j in 0..COL {
                acc += self[offset + j] * rhs[j];
            }
            res[i] = acc;
        }
    }

    pub fn mat_vec_mul_scalar_into(&self, rhs: &HeapVector<COL>, scalar: f64, res: &mut HeapVector<ROW>) {
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
        rhs: &HeapVector<COL>,
        c: &HeapVector<ROW>,
        res: &mut HeapVector<ROW>
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
        rhs: &HeapVector<COL>,
        c: &HeapVector<ROW>,
        res: &mut HeapVector<ROW>
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

    /// res = c - self * rhs
    pub fn sub_mat_vec_mul_into(
        &self,
        rhs: &HeapVector<COL>,
        c: &HeapVector<ROW>,
        res: &mut HeapVector<ROW>
    ) {
        for i in 0..ROW {
            let offset = i * COL;
            let mut acc = 0.0;
            for j in 0..COL {
                acc += self[offset + j] * rhs[j];
            }
            res[i] = c[i] - acc;
        }
    }
}

// =============================================================================
// Vector impl
// =============================================================================

impl<const N: usize> HeapVector<N> {
    pub fn sequential() -> Self {
        let data = (0..N).map(|i| i as f64).collect::<Box<[_]>>();
        Self {
            data
        }
    }

    pub fn iter(&self) -> core::slice::Iter<'_, f64> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, f64> {
        self.data.iter_mut()
    }

    pub fn chunks_exact(&self, chunk_size: usize) -> core::slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(chunk_size)
    }

    pub fn chunks_exact_mut(&mut self, chunk_size: usize) -> core::slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(chunk_size)
    }
}

// =============================================================================
// Vector Arithmetic Neon
// =============================================================================

#[cfg(target_feature = "neon")]
impl<const N: usize> HeapVector<N> {
    pub fn vec_mul(&self, rhs: &Self) -> f64 {
        let mut i = 0;
        let mut acc0 = unsafe { vdupq_n_f64(0.0) };
        let mut acc1 = unsafe { vdupq_n_f64(0.0) };
        while i + 4 <= N {
            unsafe {
                let l_vec = vld1q_f64_x2(self.data.as_ptr().add(i));
                let r_vec = vld1q_f64_x2(rhs.data.as_ptr().add(i));
                acc0 = vfmaq_f64(acc0, l_vec.0, r_vec.0);
                acc1 = vfmaq_f64(acc1, l_vec.1, r_vec.1);
            }
            i += 4;
        }
        let mut res = unsafe { vaddvq_f64(vaddq_f64(acc0, acc1)) };
        while i < N {
            res += self[i] * rhs[i];
            i += 1;
        }
        res
    }

    pub fn vec_s_mul(&self, rhs: &Self, res: &mut Self) {
        let mut i = 0;
        while i + 4 <= N {
            unsafe {
                let l_vec = vld1q_f64_x2(self.data.as_ptr().add(i));
                let r_vec = vld1q_f64_x2(rhs.data.as_ptr().add(i));
                vst1q_f64_x2(
                    res.data.as_mut_ptr().add(i),
                    float64x2x2_t(
                        vmulq_f64(l_vec.0, r_vec.0),
                        vmulq_f64(l_vec.1, r_vec.1),
                    )
                );
            }
            i += 4;
        }
        while i + 2 <= N {
            unsafe {
                let l_vec = vld1q_f64(self.data.as_ptr().add(i));
                let r_vec = vld1q_f64(rhs.data.as_ptr().add(i));
                vst1q_f64(
                    res.data.as_mut_ptr().add(i),
                    vmulq_f64(l_vec, r_vec)
                );
            }
            i += 2;
        }
        if i < N {
            res[i] = self[i] * rhs[i];
        }
    }

    pub fn vec_sub_assign(&mut self, rhs: &Self) {
        let mut i = 0;
        while i + 4 <= N {
            unsafe {
                let l_vec = vld1q_f64_x2(self.data.as_ptr().add(i));
                let r_vec = vld1q_f64_x2(rhs.data.as_ptr().add(i));
                vst1q_f64_x2(self.data.as_mut_ptr().add(i), float64x2x2_t(
                    vsubq_f64(l_vec.0, r_vec.0),
                    vsubq_f64(l_vec.1, r_vec.1)
                ));
            }
            i += 4;
        }
        while i + 2 <= N {
            unsafe {
                let l_vec = vld1q_f64(self.data.as_ptr().add(i));
                let r_vec = vld1q_f64(rhs.data.as_ptr().add(i));
                vst1q_f64(
                    self.data.as_mut_ptr().add(i),
                    vsubq_f64(l_vec, r_vec),
                );
            }
            i += 2;
        }
        if i < N {
            self[i] -= rhs[i];
        }
    }
}

// =============================================================================
// Vector Arithmetic Non-Neon
// =============================================================================

#[cfg(not(target_feature = "neon"))]
impl<const N: usize> HeapVector<N> {
    pub fn vec_mul(&self, rhs: &Self) -> f64 {
        self.data.iter()
            .zip(&rhs.data)
            .map(|(l, r)| l * r)
            .sum()
    }

    pub fn vec_s_mul(&self, rhs: &Self, res: &mut Self) {
        for i in 0..N {
            res[i] = self[i] * rhs[i];
        }
    }

    pub fn vec_sub_assign(&mut self, rhs: &Self) {
        self.data.iter_mut()
            .zip(&rhs.data)
            .for_each(|(l, r)| *l -= r);
    }
}

// =============================================================================
// Index / IndexMut
// =============================================================================

impl<const ROW: usize, const COL: usize> core::ops::Index<usize> for HeapMatrix<ROW, COL> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe {
            self.data.get_unchecked(index)
        }
    }
}

impl<const ROW: usize, const COL: usize> core::ops::IndexMut<usize> for HeapMatrix<ROW, COL> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            self.data.get_unchecked_mut(index)
        }
    }
}

impl<const ROW: usize, const COL: usize> core::ops::Index<Range<usize>> for HeapMatrix<ROW, COL> {
    type Output = [f64];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        unsafe {
            self.data.get_unchecked(index)
        }
    }
}

impl<const ROW: usize, const COL: usize> core::ops::IndexMut<Range<usize>> for HeapMatrix<ROW, COL> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        unsafe {
            self.data.get_unchecked_mut(index)
        }
    }
}

// =============================================================================
// M += M
// =============================================================================

impl<const ROW: usize, const COL: usize> core::ops::AddAssign<&Self> for HeapMatrix<ROW, COL> {
    fn add_assign(&mut self, rhs: &Self) {
        self.mat_add_assign(rhs);
    }
}

// =============================================================================
// M -= M
// =============================================================================

impl<const ROW: usize, const COL: usize> core::ops::SubAssign<&Self> for HeapMatrix<ROW, COL> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.mat_sub_assign(rhs);
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
