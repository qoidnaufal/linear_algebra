#[cfg(target_feature = "neon")]
use core::arch::aarch64::*;
use crate::matrix::Matrix;
use crate::vector::VecF;
use crate::traits::Container;

// =============================================================================
// LU decomposition
// =============================================================================

pub struct LUMatrix<const LEN: usize, const ROW: usize, const COL: usize = ROW> {
    /// a row major ROW x COL matrix stored in flat array of LEN (ROW * COL) elements
    pub matrix: Matrix<LEN, ROW, COL>,
    perm: Perm<ROW>,
}

#[repr(align(32))]
struct Perm<const N: usize>([usize; N]);

impl<const N: usize> Perm<N> {
    const INIT: Self = {
        let mut data = [0; N];
        let mut i = 1;
        while i < N {
            data[i] = i;
            i += 1;
        }
        Self(data)
    };

    fn reset(&mut self) {
        *self = Self::INIT
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.0.swap(a, b);
    }
}

impl<const N: usize> core::ops::Index<usize> for Perm<N> {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe {
            self.0.get_unchecked(index)
        }
    }
}

impl<const N: usize> core::ops::IndexMut<usize> for Perm<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            self.0.get_unchecked_mut(index)
        }
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Index<usize>
for LUMatrix<LEN, ROW, COL> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        self.matrix.index(index)
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::IndexMut<usize>
for LUMatrix<LEN, ROW, COL> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.matrix.index_mut(index)
    }
}

// =============================================================================
// Generic shapes: util
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> LUMatrix<LEN, ROW, COL> {
    pub const INIT: Self = Self {
        matrix: Matrix::ZERO,
        perm: Perm::INIT,
    };

    #[inline(always)]
    pub fn update_start(&mut self, other: &Self, end: usize) {
        let row_end = end * COL;
        unsafe {
            core::ptr::copy_nonoverlapping(
                other.matrix.data.get_unchecked(..row_end).as_ptr(),
                self.matrix.data.get_unchecked_mut(..row_end).as_mut_ptr(),
                row_end
            );
            core::ptr::copy_nonoverlapping(
                other.perm.0.get_unchecked(..end).as_ptr(),
                self.perm.0.get_unchecked_mut(..end).as_mut_ptr(),
                end
            );
        }
    }

    #[inline(always)]
    pub fn update_end(&mut self, other: &Self, start: usize) {
        let row_start = start * COL;
        unsafe {
            core::ptr::copy_nonoverlapping(
                other.matrix.data.get_unchecked(row_start..LEN).as_ptr(),
                self.matrix.data.get_unchecked_mut(row_start..LEN).as_mut_ptr(),
                LEN - row_start
            );
            core::ptr::copy_nonoverlapping(
                other.perm.0.get_unchecked(start..COL).as_ptr(),
                self.perm.0.get_unchecked_mut(start..COL).as_mut_ptr(),
                COL - start
            );
        }
    }

    pub fn reset(&mut self) {
        self.matrix = crate::mat!();
    }
}

// =============================================================================
// Solve
// =============================================================================

impl<const LEN: usize, const N: usize> LUMatrix<LEN, N> {
    #[inline]
    pub fn solve_vec(&self, b: &VecF<N>) -> VecF<N> {
        let mut x = VecF::ZERO;
        self.solve_vec_into(b, &mut x);
        x
    }

    #[inline]
    pub fn solve_mat<const LEN2: usize, const COL2: usize, B>(
        &self,
        b: &B,
    ) -> Matrix<LEN2, N, COL2>
    where
        B: Container<N, COL2>,
    {
        let mut res = Matrix::ZERO;
        self.solve_mat_into::<COL2, _, _>(b, &mut res);
        res
    }
}

// =============================================================================
// Solve Neon
// =============================================================================

#[cfg(target_feature = "neon")]
impl<const LEN: usize, const N: usize> LUMatrix<LEN, N> {
    #[inline(always)]
    pub fn factorize(&mut self) -> bool {
        self.perm.reset();

        for i in 0..N {
            let i_offset = i * N;

            let mut max_val = self.matrix[i_offset + i].abs();
            let mut max_row = i;

            for j in i + 1..N {
                let v = self.matrix[j * N + i].abs();
                if v > max_val {
                    max_val = v;
                    max_row = j;
                }
            }

            if max_val < 1e-12 { return false; }

            if max_row != i {
                let row_max_start = max_row * N;
                let row_i = self.matrix.ptr_mut(i_offset);
                let row_m = self.matrix.ptr_mut(row_max_start);
                // let row_i = self.matrix[i_offset..i_offset + N].as_mut_ptr();
                // let row_m = self.matrix[row_max_start..row_max_start + N].as_mut_ptr();
                unsafe { core::ptr::swap_nonoverlapping(row_i, row_m, N) }
                self.perm.swap(max_row, i);
            }

            let pivot = self.matrix[i_offset + i].recip();

            let i1 = i + 1;
            let i1_offset = i_offset + i1;

            for j in i + 1..N {
                let (upper, lower) = self.matrix.split_at_mut(j * N);
        
                let factor = lower[i] * pivot;
                lower[i] = factor;

                let factor_v = unsafe { vdupq_n_f64(-factor) };

                let len = N - (i + 1);
                let mut k = 0;

                while k + 4 <= len {
                    unsafe {
                        let l_v = vld1q_f64_x2(lower.as_ptr().add(i1 + k));
                        let u_v = vld1q_f64_x2(upper.as_ptr().add(i1_offset + k));
                        vst1q_f64_x2(lower.as_mut_ptr().add(i1 + k), float64x2x2_t(
                            vfmaq_f64(l_v.0, factor_v, u_v.0),
                            vfmaq_f64(l_v.1, factor_v, u_v.1)
                        ));
                    }

                    k += 4;
                }

                while k + 2 <= len {
                    unsafe {
                        let l_v = vld1q_f64(lower.as_ptr().add(i1 + k));
                        let u_v = vld1q_f64(upper.as_ptr().add(i1_offset + k));
                        vst1q_f64(
                            lower.as_mut_ptr().add(i1 + k),
                            vfmaq_f64(l_v, factor_v, u_v)
                        );
                    }
                    k += 2;
                }

                if k < len {
                    lower[i1 + k] -= factor * upper[i1_offset + k];
                }
            }
        }
        true
    }

    #[inline(always)]
    pub fn factorize_partial(&mut self, stop: usize) -> bool {
        self.perm.reset();

        for i in 0..stop {
            let i_offset = i * N;
            let i1 = i + 1;
            let i1_offset = i_offset + i1;
            let len = N - i1;

            let mut max_val = self.matrix[i_offset + i].abs();
            let mut max_row = i;

            for j in i1..stop {
                let v = self.matrix[j * N + i].abs();
                if v > max_val {
                    max_val = v;
                    max_row = j;
                }
            }

            if max_val < 1e-12 { return false }

            if max_row != i {
                let row_max_start = max_row * N;
                let row_i = self.matrix.ptr_mut(i_offset);
                let row_m = self.matrix.ptr_mut(row_max_start);
                unsafe { core::ptr::swap_nonoverlapping(row_i, row_m, N) }
                self.perm.swap(max_row, i);
            }

            let pivot = self.matrix[i_offset + i].recip();

            for j in i1..N {
                let (upper, lower) = self.matrix.split_at_mut(j * N);
                let factor = lower[i] * pivot;
                lower[i] = factor;

                let factor_v = unsafe { vdupq_n_f64(-factor) };
                let mut k = 0;

                while k + 4 <= len {
                    unsafe {
                        let l_v = vld1q_f64_x2(lower.as_ptr().add(i1 + k));
                        let u_v = vld1q_f64_x2(upper.as_ptr().add(i1_offset + k));
                        vst1q_f64_x2(lower.as_mut_ptr().add(i1 + k), float64x2x2_t(
                            vfmaq_f64(l_v.0, factor_v, u_v.0),
                            vfmaq_f64(l_v.1, factor_v, u_v.1)
                        ));
                    }

                    k += 4;
                }

                if k + 2 <= len {
                    unsafe {
                        let l_v = vld1q_f64(lower.as_ptr().add(i1 + k));
                        let u_v = vld1q_f64(upper.as_ptr().add(i1_offset + k));
                        vst1q_f64(
                            lower.as_mut_ptr().add(i1 + k),
                            vfmaq_f64(l_v, factor_v, u_v)
                        );
                    }
                    k += 2;
                }

                if k < len {
                    lower[i1 + k] -= factor * upper[i1_offset + k];
                }
            }
        }

        true
    }

    #[inline(always)]
    pub fn factorize_remaining(&mut self, start: usize) -> bool {
        for i in start..N {
            let i_offset = i * N;
            let i1 = i + 1;
            let i1_offset = i_offset + i1;
            let len = N - i1;

            let mut max_val = self.matrix[i_offset + i].abs();
            let mut max_row = i;

            for j in i1..N {
                let v = self.matrix[j * N + i].abs();
                if v > max_val {
                    max_val = v;
                    max_row = j;
                }
            }

            if max_val < 1e-12 { return false; }

            if max_row != i {
                let row_max_start = max_row * N;
                let row_i = self.matrix.ptr_mut(i_offset);
                let row_m = self.matrix.ptr_mut(row_max_start);
                unsafe { core::ptr::swap_nonoverlapping(row_i, row_m, N) }
                self.perm.swap(max_row, i);
            }

            let pivot = self.matrix[i_offset + i].recip();

            for j in i1..N {
                let (upper, lower) = self.matrix.split_at_mut(j * N);
                let factor = lower[i] * pivot;
                lower[i] = factor;

                let factor_v = unsafe { vdupq_n_f64(-factor) };
                let lower = lower.as_mut_ptr();
                let upper = upper.as_ptr();
                let mut k = 0;

                while k + 4 <= len {
                    unsafe {
                        let l_v = vld1q_f64_x2(lower.add(i1 + k));
                        let u_v = vld1q_f64_x2(upper.add(i1_offset + k));
                        vst1q_f64_x2(lower.add(i1 + k), float64x2x2_t(
                            vfmaq_f64(l_v.0, factor_v, u_v.0),
                            vfmaq_f64(l_v.1, factor_v, u_v.1)
                        ));
                    }

                    k += 4;
                }

                if k + 2 <= len {
                    unsafe {
                        let l_v = vld1q_f64(lower.add(i1 + k));
                        let u_v = vld1q_f64(upper.add(i1_offset + k));
                        vst1q_f64(lower.add(i1 + k), vfmaq_f64(l_v, factor_v, u_v));
                    }
                    k += 2;
                }

                if k < len {
                    unsafe { *lower.add(i1 + k) -= factor * *upper.add(i1_offset + k) }
                }
            }
        }

        true
    }

    #[inline(always)]
    pub fn solve_vec_into(&self, b: &VecF<N>, x: &mut VecF<N>) {
        for i in 0..N {
            let offset = i * N;
            let mut acc0 = unsafe { vdupq_n_f64(0.0) };
            let mut acc1 = unsafe { vdupq_n_f64(0.0) };
            let mut j = 0;

            while j + 4 <= i {
                unsafe {
                    let lu_v = vld1q_f64_x2(self.matrix.ptr(offset + j));
                    let x_v = vld1q_f64_x2(x.ptr(j));
                    acc0 = vfmsq_f64(acc0, lu_v.0, x_v.0);
                    acc1 = vfmsq_f64(acc1, lu_v.1, x_v.1);
                }
                j += 4;
            }

            let mut acc_v = unsafe { vaddq_f64(acc0, acc1) };

            if j + 2 <= i {
                unsafe {
                    let lu_v = vld1q_f64(self.matrix.ptr(offset + j));
                    let x_v = vld1q_f64(x.ptr(j));
                    acc_v = vfmsq_f64(acc_v, lu_v, x_v);
                }
                j += 2;
            }

            let mut acc = b[self.perm[i]] + unsafe { vaddvq_f64(acc_v) };
            if j < i { acc -= self.matrix[offset + j] * x[j] }
            x[i] = acc;
        }

        for i in (0..N).rev() {
            let offset = i * N;
            let mut acc0 = unsafe { vdupq_n_f64(0.0) };
            let mut acc1 = unsafe { vdupq_n_f64(0.0) };
            let mut j = i + 1;

            while j + 4 <= N {
                unsafe {
                    let lu_v = vld1q_f64_x2(self.matrix.ptr(offset + j));
                    let x_v = vld1q_f64_x2(x.ptr(j));
                    acc0 = vfmsq_f64(acc0, lu_v.0, x_v.0);
                    acc1 = vfmsq_f64(acc1, lu_v.1, x_v.1);
                }
                j += 4;
            }

            let mut acc_v = unsafe { vaddq_f64(acc0, acc1) };

            if j + 2 <= N {
                unsafe {
                    let lu_v = vld1q_f64(self.matrix.ptr(offset + j));
                    let x_v = vld1q_f64(x.ptr(j));
                    acc_v = vfmsq_f64(acc_v, lu_v, x_v);
                }
                j += 2;
            }

            let mut acc = x[i] + unsafe { vaddvq_f64(acc_v) };
            if j < N { acc -= self.matrix[offset + j] * x[j] }
            x[i] = acc * self.matrix[offset + i].recip();
        }
    }

    #[inline(always)]
    pub fn solve_mat_into<const COL2: usize, B, X>(
        &self,
        b: &B,
        x: &mut X,
    )
    where
        B: Container<N, COL2>,
        X: Container<N, COL2>,
    {
        for i in 0..N {
            let i_offset = i * N;
            let x_i_off = i * COL2;
            let b_perm_off = self.perm[i] * COL2;

            for c in 0..COL2 {
                x[x_i_off + c] = b[b_perm_off + c];
            }

            for j in 0..i {
                let l_ij = self.matrix[i_offset + j];
                let x_j_off = j * COL2;
                
                let mut c = 0;
                let l_v = unsafe { vdupq_n_f64(-l_ij) };
                while c + 4 <= COL2 {
                    unsafe {
                        let xi_v = vld1q_f64_x2(x.ptr(x_i_off + c));
                        let xj_v = vld1q_f64_x2(x.ptr(x_j_off + c));
                        vst1q_f64_x2(x.ptr_mut(x_i_off + c), float64x2x2_t(
                            vfmaq_f64(xi_v.0, l_v, xj_v.0),
                            vfmaq_f64(xi_v.1, l_v, xj_v.1),
                        ));
                    }
                    c += 4;
                }
                if c + 2 <= COL2 {
                    unsafe {
                        let xi_v = vld1q_f64(x.ptr(x_i_off + c));
                        let xj_v = vld1q_f64(x.ptr(x_j_off + c));
                        let res = vfmaq_f64(xi_v, l_v, xj_v);
                        vst1q_f64(x.ptr_mut(x_i_off + c), res);
                    }
                    c += 2;
                }
                if c < COL2 {
                    x[x_i_off + c] -= l_ij * x[x_j_off + c];
                }
            }
        }

        for i in (0..N).rev() {
            let i_offset = i * N;
            let x_i_off = i * COL2;

            for j in i + 1..N {
                let u_ij = self.matrix[i_offset + j];
                let x_j_off = j * COL2;

                let mut c = 0;
                let u_v = unsafe { vdupq_n_f64(-u_ij) };
                while c + 4 <= COL2 {
                    unsafe {
                        let xi_v = vld1q_f64_x2(x.ptr(x_i_off + c));
                        let xj_v = vld1q_f64_x2(x.ptr(x_j_off + c));
                        vst1q_f64_x2(x.ptr_mut(x_i_off + c), float64x2x2_t(
                            vfmaq_f64(xi_v.0, u_v, xj_v.0),
                            vfmaq_f64(xi_v.1, u_v, xj_v.1),
                        ));
                    }
                    c += 4;
                }
                if c + 2 <= COL2 {
                    unsafe {
                        let xi_v = vld1q_f64(x.ptr(x_i_off + c));
                        let xj_v = vld1q_f64(x.ptr(x_j_off + c));
                        let res = vfmaq_f64(xi_v, u_v, xj_v);
                        vst1q_f64(x.ptr_mut(x_i_off + c), res);
                    }
                    c += 2;
                }
                if c < COL2 {
                    x[x_i_off + c] -= u_ij * x[x_j_off + c];
                }
            }

            let u_ii_inv = self.matrix[i_offset + i].recip();
            let mut c = 0;
            let inv_v = unsafe { vdupq_n_f64(u_ii_inv) };
            while c + 4 <= COL2 {
                unsafe {
                    let xi_v = vld1q_f64_x2(x.ptr(x_i_off + c));
                    vst1q_f64_x2(x.ptr_mut(x_i_off + c), float64x2x2_t(
                        vmulq_f64(xi_v.0, inv_v),
                        vmulq_f64(xi_v.1, inv_v),
                    ));
                }
                c += 4;
            }
            if c + 2 <= COL2 {
                unsafe {
                    let xi_v = vld1q_f64(x.ptr(x_i_off + c));
                    vst1q_f64(x.ptr_mut(x_i_off + c), vmulq_f64(xi_v, inv_v));
                }
                c += 2;
            }
            if c < COL2 {
                x[x_i_off + c] *= u_ii_inv;
            }
        }
    }
}

// =============================================================================
// Solve Non-Neon
// =============================================================================

#[cfg(not(target_feature = "neon"))]
impl<const LEN: usize, const N: usize> LUMatrix<LEN, N> {
    #[inline(always)]
    pub fn factorize(&mut self) -> bool {
        self.perm.reset();

        for i in 0..N {
            let i_offset = i * N;

            let mut max_val = self.matrix[i_offset + i].abs();
            let mut max_row = i;

            for j in i + 1..N {
                let v = self.matrix[j * N + i].abs();
                if v > max_val {
                    max_val = v;
                    max_row = j;
                }
            }

            if max_val < 1e-12 { return false; }

            if max_row != i {
                let row_max_start = max_row * N;
                let row_i = self.matrix[i_offset..i_offset + N].as_mut_ptr();
                let row_m = self.matrix[row_max_start..row_max_start + N].as_mut_ptr();
                unsafe { core::ptr::swap_nonoverlapping(row_i, row_m, N) }
                self.perm.swap(max_row, i);
            }

            let pivot = self.matrix[i_offset + i].recip();

            for j in i + 1..N {
                let (left, right) = self.matrix.split_at_mut(j * N);
                let lower = &mut right[..N];
                let upper = &left[i_offset..i_offset + N];
            
                let factor = lower[i] * pivot;
                lower[i] = factor;

                for k in i + 1..N {
                    lower[k] -= factor * upper[k]
                }
            }
        }
        true
    }

    #[inline(always)]
    pub fn factorize_partial(&mut self, stop: usize) -> bool {
        self.perm.reset();

        for i in 0..stop {
            let i_offset = i * N;

            let mut max_val = self.matrix[i_offset + i].abs();
            let mut max_row = i;

            for j in i + 1..stop {
                let v = self.matrix[j * N + i].abs();
                if v > max_val {
                    max_val = v;
                    max_row = j;
                }
            }

            if max_val < 1e-12 { return false }

            if max_row != i {
                let row_max_start = max_row * N;
                let row_i = self.matrix[i_offset..i_offset + N].as_mut_ptr();
                let row_m = self.matrix[row_max_start..row_max_start + N].as_mut_ptr();
                unsafe { core::ptr::swap_nonoverlapping(row_i, row_m, N) }
                self.perm.swap(max_row, i);
            }

            let pivot = self.matrix[i_offset + i].recip();

            for j in i + 1..N {
                let factor = self.matrix[j * N + i] * pivot;
                self.matrix[j * N + i] = factor;
                for k in i + 1..N {
                    self.matrix[j * N + k] -= factor * self.matrix[i * N + k];
                }
            }
        }

        true
    }

    #[inline(always)]
    pub fn factorize_remaining(&mut self, start: usize) -> bool {
        for i in start..N {
            let i_offset = i * N;

            let mut max_val = self.matrix[i_offset + i].abs();
            let mut max_row = i;

            for j in i + 1..N {
                let v = self.matrix[j * N + i].abs();
                if v > max_val {
                    max_val = v;
                    max_row = j;
                }
            }

            if max_val < 1e-12 { return false; }

            if max_row != i {
                let row_max_start = max_row * N;
                let row_i = self.matrix[i_offset..i_offset + N].as_mut_ptr();
                let row_m = self.matrix[row_max_start..row_max_start + N].as_mut_ptr();
                unsafe { core::ptr::swap_nonoverlapping(row_i, row_m, N) }
                self.perm.swap(max_row, i);
            }

            let pivot = self.matrix[i_offset + i].recip();

            for j in i + 1..N {
                let factor = self.matrix[j * N + i] * pivot;
                self.matrix[j * N + i] = factor;
                for k in i + 1..N {
                    self.matrix[j * N + k] -= factor * self.matrix[i * N + k];
                }
            }
        }

        true
    }

    #[inline(always)]
    pub fn solve_vec_into(&self, b: &VecF<N>, x: &mut VecF<N>) {
        for i in 0..N {
            let offset = i * N;
            let mut acc = b[self.perm[i]];
            for j in 0..i {
                acc -= self.matrix[offset + j] * x[j];
            }
            x[i] = acc;
        }

        for i in (0..N).rev() {
            let offset = i * N;
            let mut acc = x[i];
            for j in i + 1..N {
                acc -= self.matrix[offset + j] * x[j];
            }
            x[i] = acc * self.matrix[offset + i].recip();
        }
    }

    #[inline(always)]
    pub fn solve_mat_into<const COL2: usize, B, X>(
        &self,
        b: &B,
        x: &mut X,
    )
    where
        B: Container<N, COL2>,
        X: Container<N, COL2>,
    {
        for i in 0..N {
            let i_offset = i * N;
            let x_i_off = i * COL2;
            let b_offset = self.perm[i] * COL2;

            for c in 0..COL2 {
                x[x_i_off + c] = b[b_offset + c];
            }

            for j in 0..i {
                let l_ij = self.matrix[i_offset + j];
                let x_j_off = j * COL2;
                
                for c in 0..COL2 {
                    x[x_i_off + c] -= l_ij * x[x_j_off + c];
                }
            }
        }

        for i in (0..N).rev() {
            let i_offset = i * N;
            let x_i_off = i * COL2;

            for j in i + 1..N {
                let u_ij = self.matrix[i_offset + j];
                let x_j_off = j * COL2;

                for c in 0..COL2 {
                    x[x_i_off + c] -= u_ij * x[x_j_off + c];
                }
            }

            let u_ii_inv = self.matrix[i_offset + i].recip();
            for c in 0..COL2 {
                x[x_i_off + c] *= u_ii_inv;
            }
        }
    }
}

// =============================================================================
// fmt
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize>
    core::fmt::Debug for LUMatrix<LEN, ROW, COL> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.matrix.fmt(f)
    }
}
