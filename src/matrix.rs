#[cfg(target_feature = "neon")]
use core::arch::aarch64::*;
use crate::vector::VecF;

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

// =============================================================================
// General impl (any ROW x COL)
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> Matrix<LEN, ROW, COL> {
    pub const ZERO: Self = Self { data: [0.0; LEN] };

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

    pub fn scalar_mul(&self, scalar: f64) -> Self {
        let mut result = Self::ZERO;
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                result[offset + col] = self[offset + col] * scalar;
            }
        }
        result
    }

    pub fn scalar_add(&self, scalar: f64) -> Self {
        let mut result = Self::ZERO;
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                result[offset + col] = self[offset + col] + scalar
            }
        }
        result
    }

    pub fn mat_sub(&self, rhs: &Self) -> Self {
        let mut result = Self::ZERO;
        for row in 0..ROW {
            let offset = row * COL;
            for col in 0..COL {
                result.data[offset + col] = self.data[offset + col] - rhs.data[offset + col];
            }
        }
        result
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
        let mut out = Matrix::<LEN, COL, ROW>::ZERO;

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

// =============================================================================
// Const operations
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> Matrix<LEN, ROW, COL> {
    pub const fn const_matmul<
        const ROW_X_COL2: usize,
        const LEN2: usize,
        const COL2: usize,
    >(&self, rhs: &Matrix<LEN2, COL, COL2>) -> Matrix<ROW_X_COL2, ROW, COL2> {
        let mut result = Matrix::ZERO;
        let mut i = 0;
        while i < ROW {
            let mut k = 0;
            while k < COL {
                let a_val = self.data[i * COL + k];
                let mut j = 0;
                while j < COL2 {
                    result.data[i * COL2 + j] += a_val * rhs.data[k * COL2 + j];
                    j += 1;
                }
                k += 1;
            }
            i += 1;
        }
        result
    }

    pub const fn const_transpose(&self) -> Matrix<LEN, COL, ROW> {
        let mut out = Matrix::<LEN, COL, ROW>::ZERO;

        let mut r_tile = 0;
        while r_tile < ROW {
            let mut c_tile = 0;
            while c_tile < COL {
                let r_end = if r_tile + 8 < ROW { r_tile + 8 } else { ROW };
                let c_end = if c_tile + 8 < COL { c_tile + 8 } else { COL };

                let mut r = r_tile;
                while r < r_end {
                    let r_offset = r * COL;
                    let mut c = c_tile;
                    while c < c_end {
                        out.data[c * ROW + r] = self.data[r_offset + c];
                        c += 1;
                    }
                    r += 1;
                }

                c_tile += 8;
            }

            r_tile += 8;
        }
        out
    }
}

// =============================================================================
// Arithmetic Non-Neon
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> Matrix<LEN, ROW, COL> {
    pub fn matmul<
        const ROW_X_COL2: usize,
        const LEN2: usize,
        const COL2: usize,
    >(&self, rhs: &Matrix<LEN2, COL, COL2>) -> Matrix<ROW_X_COL2, ROW, COL2> {
        let mut result = Matrix::ZERO;
        for i in 0..ROW {
            for k in 0..COL {
                let a_val = self[i * COL + k];
                for j in 0..COL2 {
                    result[i * COL2 + j] += a_val * rhs[k * COL2 + j];
                }
            }
        }
        result
    }

    pub fn matmul_into<
        const LEN3: usize,
        const LEN2: usize,
        const COL2: usize,
    >(
        &self,
        rhs: &Matrix<LEN2, COL, COL2>,
        result: &mut Matrix<LEN3, ROW, COL2>,
    ) {
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
// Addition
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Add<Self>
    for Matrix<LEN, ROW, COL>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        mat_add(&self, &rhs)
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Add<Matrix<LEN, ROW, COL>>
    for &Matrix<LEN, ROW, COL>
{
    type Output = Matrix<LEN, ROW, COL>;

    fn add(self, rhs: Matrix<LEN, ROW, COL>) -> Self::Output {
        mat_add(self, &rhs)
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Add<Self>
    for &Matrix<LEN, ROW, COL>
{
    type Output = Matrix<LEN, ROW, COL>;

    fn add(self, rhs: Self) -> Self::Output {
        mat_add(self, rhs)
    }
}

fn mat_add<
    const LEN: usize,
    const ROW: usize,
    const COL: usize
>(
    lhs: &Matrix<LEN, ROW, COL>,
    rhs: &Matrix<LEN, ROW, COL>
) -> Matrix<LEN, ROW, COL> {
    let mut result = Matrix::ZERO;
    for row in 0..ROW {
        let offset = row * COL;
        for col in 0..COL {
            result[offset + col] = lhs[offset + col] + rhs[offset + col]
        }
    }
    result
}

// =============================================================================
// Substraction
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Sub<Self>
    for Matrix<LEN, ROW, COL>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.mat_sub(&rhs)
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Sub<Matrix<LEN, ROW, COL>>
    for &Matrix<LEN, ROW, COL>
{
    type Output = Matrix<LEN, ROW, COL>;

    fn sub(self, rhs: Matrix<LEN, ROW, COL>) -> Self::Output {
        self.mat_sub(&rhs)
    }
}

// =============================================================================
// Matmul: Matrix x Vec
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Mul<VecF<COL>>
    for Matrix<LEN, ROW, COL>
{
    type Output = VecF<ROW>;

    fn mul(self, rhs: VecF<COL>) -> VecF<ROW> {
        mat_vec_mul(&self, &rhs)
    }
}

impl<'a, const LEN: usize, const ROW: usize, const COL: usize> core::ops::Mul<VecF<COL>>
    for &'a Matrix<LEN, ROW, COL>
{
    type Output = VecF<ROW>;

    fn mul(self, rhs: VecF<COL>) -> VecF<ROW> {
        mat_vec_mul(self, &rhs)
    }
}

impl<'a, const LEN: usize, const ROW: usize, const COL: usize> core::ops::Mul<&'a VecF<COL>>
    for &'a Matrix<LEN, ROW, COL>
{
    type Output = VecF<ROW>;

    fn mul(self, rhs: &VecF<COL>) -> VecF<ROW> {
        mat_vec_mul(self, rhs)
    }
}

impl<const LEN: usize, const ROW: usize, const COL: usize> core::ops::Mul<&VecF<COL>>
    for Matrix<LEN, ROW, COL>
{
    type Output = VecF<ROW>;

    fn mul(self, rhs: &VecF<COL>) -> VecF<ROW> {
        mat_vec_mul(&self, rhs)
    }
}

#[cfg(target_feature = "neon")]
fn mat_vec_mul<const LEN: usize, const ROW: usize, const COL: usize>(
    mat: &Matrix<LEN, ROW, COL>,
    rhs: &VecF<COL>,
) -> VecF<ROW> {
    let mut res = VecF::<ROW>::ZERO;

    for i in 0..ROW {
        let offset = i * COL;
        let mut acc0 = unsafe { vdupq_n_f64(0.0) };
        let mut acc1 = unsafe { vdupq_n_f64(0.0) };

        let mut j = 0;

        while j + 4 <= COL {
            unsafe {
                let m_vec = vld1q_f64_x2(mat.data.as_ptr().add(offset + j));
                let r_vec = vld1q_f64_x2(rhs.data.as_ptr().add(j));
                acc0 = vfmaq_f64(acc0, m_vec.0, r_vec.0);
                acc1 = vfmaq_f64(acc1, m_vec.1, r_vec.1);
            }
            j += 4;
        }

        let mut acc = unsafe { vaddq_f64(acc0, acc1) };

        while j + 2 <= COL {
            unsafe {
                let m_vec = vld1q_f64(mat.data.as_ptr().add(offset + j));
                let r_vec = vld1q_f64(rhs.data.as_ptr().add(j));
                acc = vfmaq_f64(acc, m_vec, r_vec);
            }
            j += 2;
        }

        let mut dot = unsafe { vaddvq_f64(acc) };

        if j < COL {
            dot += mat[offset + j] * rhs[j];
        }

        res[i] = dot;
    }

    res
}

#[cfg(not(target_feature = "neon"))]
fn mat_vec_mul<const LEN: usize, const ROW: usize, const COL: usize>(
    mat: &Matrix<LEN, ROW, COL>,
    rhs: &VecF<COL>,
) -> VecF<ROW> {
    let mut out = VecF::ZERO;

    for row in 0..ROW {
        let offset = row * COL;
        let mut acc = 0.0;
        for col in 0..COL {
            acc += mat[offset + col] * rhs[col]
        }
        out[row] = acc
    }

    out
}

// =============================================================================
// Matmul: Square Matrices
// =============================================================================

impl<const LEN: usize, const N: usize> core::ops::Mul for Matrix<LEN, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.matmul(&rhs)
    }
}

impl<const LEN: usize, const N: usize> core::ops::Mul<&Self> for Matrix<LEN, N> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.matmul(rhs)
    }
}

impl<const LEN: usize, const N: usize> core::ops::Mul<Self> for &Matrix<LEN, N> {
    type Output = Matrix<LEN, N>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.matmul(rhs)
    }
}

impl<const LEN: usize, const N: usize> core::ops::Mul<Matrix<LEN, N>> for &Matrix<LEN, N> {
    type Output = Matrix<LEN, N>;

    fn mul(self, rhs: Matrix<LEN, N>) -> Self::Output {
        self.matmul(&rhs)
    }
}

// =============================================================================
// PartialEq
// =============================================================================

impl<const LEN: usize, const ROW: usize, const COL: usize> PartialEq
    for Matrix<LEN, ROW, COL>
{
    fn eq(&self, other: &Self) -> bool {
        &self.data == &other.data
    }
}

// =============================================================================
// Debug: this is highly inefficient but whatever
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

#[cfg(test)]
mod matrix_test {
    use super::*;
    use crate::{mat, vecf};

    #[test]
    fn matmul() {
        let m = mat!(4 =>
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4,
        );
        let v = vecf!(4 => 2, 2, 2, 2);
        let mul = &m * &v;
        let exp = vecf!(4 => 8, 16, 24, 32);
        assert_eq!(mul, exp);
        println!("{m:?}");
        println!("{mul:?}");
    }

    #[test]
    fn transposed() {
        let i = mat!(4 =>
            1, -1,  0,  0,
            0,  1, -1,  0,
            0,  0,  1,  0,
            0,  0,  1, -1,
        );
        let i_t = i.transpose();
        let c_it = i.const_transpose();
        assert_eq!(i_t, c_it);
        let g = mat!(4 =>
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        );
        println!("{i:?}");
        println!("{i_t:?}");
        println!("{:?}", i_t * g * i);
    }
}
