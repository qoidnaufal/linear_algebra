use crate::lu::LUMatrix;
use crate::matrix::Matrix;
use crate::vector::VecF;

// =============================================================================
// Woodbury:
//     (A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹
// =============================================================================

#[derive(Debug)]
/// # Woodbury Identity
/// (A + UCV)⁻¹p = (A⁻¹)p - (A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹)p
/// 
/// # Breakdown
/// x0   = `A⁻¹ * p`
/// Au   = `A⁻¹ * U`
/// M_lu = (C⁻¹ + VA⁻¹U)⁻¹   = `(C⁻¹ + VAu)⁻¹`
/// w    = VA⁻¹p             = V * A⁻¹ * p     = `Vx0`
/// x1   = Au * M_lu * VA⁻¹p = Au * M_lu * Vx0 = `Au * M_lu * w`
/// res  = `x0 - x1`
pub struct WoodburyCache<
    const LEN_A: usize, const N: usize,
    const LEN_UV: usize, const K: usize,
    const LEN_C: usize,
> {
    /// base A⁻¹
    pub a0: LUMatrix<LEN_A, N>,
    /// (C⁻¹ + VA⁻¹U)⁻¹
    pub m0_lu: LUMatrix<LEN_C, K>,
    /// A⁻¹U where U = Vᵀ
    pub a_u: Matrix<LEN_UV, N, K>,
    /// update matrix
    pub c_inv: Matrix<LEN_C, K>,
    /// incident matrix
    pub v: Matrix<LEN_UV, K, N>,
    pub is_dirty: bool,
}

impl<
    const LEN_A: usize, const N: usize,
    const LEN_UV: usize, const K: usize,
    const LEN_C: usize,
> WoodburyCache<LEN_A, N, LEN_UV, K, LEN_C> {
    pub const INIT: Self = Self {
        a0: LUMatrix::INIT,
        m0_lu: LUMatrix::INIT,
        a_u: Matrix::ZERO,
        c_inv: Matrix::ZERO,
        v: Matrix::ZERO,
        is_dirty: true,
    };

    /// In most cases, this will be called once at the initialization.
    /// Make sure a0 is already set, otherwise this won't work.
    /// # Usage
    /// Use this if you plan to update only through c, otherwise use [`prepare_u`](Self::prepare_u)
    pub fn prepare_uv(&mut self, u: &Matrix<LEN_UV, N, K>, v: &Matrix<LEN_UV, K, N>) {
        self.a0.factorize();
        self.a0.solve_mat_into(u, &mut self.a_u);
        self.v = *v;
    }

    /// c_inv is diagonal matrix where `VAR <= K` and `offset + VAR <= K`
    pub fn update_c_inv<
        const LEN_VAR: usize,
        const VAR: usize
    >(
        &mut self,
        c_inv: &Matrix<LEN_VAR, VAR>,
        offset: usize
    ) {
        debug_assert!(offset + VAR <= K);
        for i in 0..VAR {
            let lhs_row = (i + offset) * K;
            let rhs_row = i * VAR;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    c_inv.data.as_ptr().add(rhs_row),
                    self.c_inv.data.as_mut_ptr().add(lhs_row + offset),
                    VAR
                );
            }
        }
        self.is_dirty = true;
    }

    pub fn refactor_if_necessary(&mut self) -> bool {
        if self.is_dirty {
            self.m0_lu.matrix = self.c_inv + self.v.matmul(&self.a_u);
            if !self.m0_lu.factorize() { return false }
            self.is_dirty = false;
        }
        true
    }

    pub fn solve_vec(&self, rhs: &VecF<N>) -> VecF<N> {
        let x0 = self.a0.solve_vec(rhs);
        let va_rhs = self.v * x0;
        let x1 = self.a_u * self.m0_lu.solve_vec(&va_rhs);
        x0 - x1
    }

    pub fn solve_mat<
        const LEN_K_COL2: usize,
        const LEN_2: usize,
        const COL_2: usize,
    >(
        &self,
        rhs: &Matrix<LEN_2, N, COL_2>,
    ) -> Matrix<LEN_2, N, COL_2> {
        let mut x0 = Matrix::ZERO;
        self.a0.solve_mat_into(rhs, &mut x0);
        let va_rhs = self.v.matmul::<LEN_K_COL2, _, _>(&x0);
        let mut arg = Matrix::ZERO;
        self.m0_lu.solve_mat_into(&va_rhs, &mut arg);
        let x1 = self.a_u.matmul(&arg);
        x0 - x1
    }
}

impl<
    const LEN_A: usize, const N: usize,
    const LEN_UV: usize, const K: usize,
    const LEN_C: usize,
> core::ops::Index<usize> for WoodburyCache<LEN_A, N, LEN_UV, K, LEN_C> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe {
            self.a0.matrix.data.get_unchecked(index)
        }
    }
}

impl<
    const LEN_A: usize, const N: usize,
    const LEN_UV: usize, const K: usize,
    const LEN_C: usize,
> core::ops::IndexMut<usize> for WoodburyCache<LEN_A, N, LEN_UV, K, LEN_C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            self.a0.matrix.data.get_unchecked_mut(index)
        }
    }
}

// =============================================================================
// Partitioned Woodbury
// =============================================================================

#[derive(Debug)]
pub struct PartitionedWoodbury<
    const LEN_A: usize, const N: usize,
    const LEN_UV1: usize, const LEN_C1: usize, const K1: usize,
    const LEN_UV2: usize, const LEN_C2: usize, const K2: usize,
    const LEN_C1_C2: usize,
> {
    pub a0: LUMatrix<LEN_A, N>,
    
    pub a_u1: Matrix<LEN_UV1, N, K1>,
    pub a_u2: Matrix<LEN_UV2, N, K2>,
    pub v1: Matrix<LEN_UV1, K1, N>,
    pub v2: Matrix<LEN_UV2, K2, N>,
    pub c1_inv: Matrix<LEN_C1, K1>,
    pub c2_inv: Matrix<LEN_C2, K2>,

    /// V₁A⁻¹U₂
    pub v1_a_u2: Matrix<LEN_C1_C2, K1, K2>,
    /// V₂A⁻¹U₁
    pub v2_a_u1: Matrix<LEN_C1_C2, K2, K1>,
    /// V₂A⁻¹U₂
    pub v2_a_u2: Matrix<LEN_C2, K2>,

    /// (C₁⁻¹ + V₁A⁻¹U₁)⁻¹
    pub m1_lu: LUMatrix<LEN_C1, K1>,
    /// (C₁⁻¹ + V₁A⁻¹U₁)⁻¹V₁A⁻¹U₂
    pub m1_v1_a_u2: Matrix<LEN_C1_C2, K1, K2>,

    /// V₂A⁻¹U₂ - V₂A⁻¹U₁(C₁⁻¹ + V₁A⁻¹U₁)⁻¹V₁A⁻¹U₂
    pub s_base: Matrix<LEN_C2, K2>,
    /// (C₂⁻¹ + (V₂A⁻¹U₂ - V₂A⁻¹U₁(C₁⁻¹ + V₁A⁻¹U₁)⁻¹V₁A⁻¹U₂))⁻¹
    pub schur: LUMatrix<LEN_C2, K2>,

    pub is_dirty: bool,
}

impl<
    const LEN_A: usize, const N: usize,
    const LEN_UV1: usize, const LEN_C1: usize, const K1: usize,
    const LEN_UV2: usize, const LEN_C2: usize, const K2: usize,
    const LEN_C1_C2: usize,
> PartitionedWoodbury<LEN_A, N, LEN_UV1, LEN_C1, K1, LEN_UV2, LEN_C2, K2, LEN_C1_C2> {
    pub const INIT: Self = Self {
        a0: LUMatrix::INIT,
        a_u1: Matrix::ZERO,
        a_u2: Matrix::ZERO,
        v1: Matrix::ZERO,
        v2: Matrix::ZERO,
        v1_a_u2: Matrix::ZERO,
        v2_a_u1: Matrix::ZERO,
        v2_a_u2: Matrix::ZERO,
        c1_inv: Matrix::ZERO,
        c2_inv: Matrix::ZERO,
        m1_lu: LUMatrix::INIT,
        m1_v1_a_u2: Matrix::ZERO,
        s_base: Matrix::ZERO,
        schur: LUMatrix::INIT,
        is_dirty: true,
    };

    pub fn prepare_uv(
        &mut self,
        u1: &Matrix<LEN_UV1, N, K1>, v1: &Matrix<LEN_UV1, K1, N>,
        u2: &Matrix<LEN_UV2, N, K2>, v2: &Matrix<LEN_UV2, K2, N>,
    ) {
        self.a0.factorize();
        self.a0.solve_mat_into(u1, &mut self.a_u1);
        self.a0.solve_mat_into(u2, &mut self.a_u2);
        self.v1_a_u2 = v1.matmul::<LEN_C1_C2, _, _>(&self.a_u2);
        self.v2_a_u1 = v2.matmul::<LEN_C1_C2, _, _>(&self.a_u1);
        self.v2_a_u2 = v2.matmul::<LEN_C2, _, _>(&self.a_u2);
        self.v1 = *v1;
        self.v2 = *v2;
    }

    pub fn update_c1_inv<
        const LEN_VAR1: usize,
        const VAR1: usize,
    >(
        &mut self,
        c1_inv: &Matrix<LEN_VAR1, VAR1>,
        offset: usize,
    ) {
        for i in 0..VAR1 {
            let lhs_row = (i + offset) * K1;
            let rhs_row = i * VAR1;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    c1_inv.data.as_ptr().add(rhs_row),
                    self.c1_inv.data.as_mut_ptr().add(lhs_row + offset),
                    VAR1
                );
            }
        }
        self.is_dirty = true;
    }

    pub fn update_c2_inv<
        const LEN_VAR2: usize,
        const VAR2: usize,
    >(
        &mut self,
        c2_inv: &Matrix<LEN_VAR2, VAR2>,
        offset: usize,
    ) {
        for i in 0..VAR2 {
            let lhs_row = (i + offset) * K2;
            let rhs_row = i * VAR2;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    c2_inv.data.as_ptr().add(rhs_row),
                    self.c2_inv.data.as_mut_ptr().add(lhs_row + offset),
                    VAR2
                );
            }
        }
    }

    pub fn factorize1(&mut self) -> bool {
        self.m1_lu.matrix = self.c1_inv + self.v1.matmul::<LEN_C1, _, _>(&self.a_u1);
        if !self.m1_lu.factorize() { return false; }

        self.m1_lu.solve_mat_into(&self.v1_a_u2, &mut self.m1_v1_a_u2);
        self.s_base = self.v2_a_u2 - self.v2_a_u1.matmul::<LEN_C2, _, _>(&self.m1_v1_a_u2);
        true
    }

    pub fn factorize2(&mut self) -> bool {
        self.schur.matrix = self.c2_inv + self.s_base;
        self.schur.factorize()
    }

    pub fn solve_vec(&self, rhs: &VecF<N>) -> VecF<N> {
        let x0 = self.a0.solve_vec(rhs);

        let b1 = self.v1 * x0;
        let b2 = self.v2 * x0;

        let w = self.m1_lu.solve_vec(&b1);
        let y2 = b2 - self.v2_a_u1 * w;
        let x2 = self.schur.solve_vec(&y2);
        let x1 = w - self.m1_v1_a_u2 * x2;

        x0 - self.a_u1 * x1 - self.a_u2 * x2
    }

    pub fn solve_mat<
        const LEN_K1_COL2: usize,
        const LEN_K2_COL2: usize,
        const LEN_2: usize,
        const COL_2: usize,
    >(
        &self,
        rhs: &Matrix<LEN_2, N, COL_2>,
    ) -> Matrix<LEN_2, N, COL_2> {
        let mut x0 = Matrix::ZERO;
        self.a0.solve_mat_into(rhs, &mut x0);
        let b1 = self.v1.matmul::<LEN_K1_COL2, _, _>(&x0);
        let b2 = self.v2.matmul::<LEN_K2_COL2, _, _>(&x0);

        let mut w = Matrix::ZERO;
        self.m1_lu.solve_mat_into(&b1, &mut w);
        let y2 = b2 - self.v2_a_u1.matmul(&w);
        let mut x2 = Matrix::ZERO;
        self.schur.solve_mat_into(&y2, &mut x2);
        let x1 = w - self.m1_v1_a_u2.matmul(&x2);

        x0 - self.a_u1.matmul(&x1) - self.a_u2.matmul(&x2)
    }
}

impl<
    const LEN_A: usize, const N: usize,
    const LEN_UV1: usize, const LEN_C1: usize, const K1: usize,
    const LEN_UV2: usize, const LEN_C2: usize, const K2: usize,
    const LEN_C1_C2: usize,
> core::ops::Index<usize> for PartitionedWoodbury<LEN_A, N, LEN_UV1, LEN_C1, K1, LEN_UV2, LEN_C2, K2, LEN_C1_C2> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe {
            self.a0.matrix.data.get_unchecked(index)
        }
    }
}

impl<
    const LEN_A: usize, const N: usize,
    const LEN_UV1: usize, const LEN_C1: usize, const K1: usize,
    const LEN_UV2: usize, const LEN_C2: usize, const K2: usize,
    const LEN_C1_C2: usize,
> core::ops::IndexMut<usize> for PartitionedWoodbury<LEN_A, N, LEN_UV1, LEN_C1, K1, LEN_UV2, LEN_C2, K2, LEN_C1_C2> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            self.a0.matrix.data.get_unchecked_mut(index)
        }
    }
}

// =============================================================================
// LU Woodbury
// =============================================================================

impl<const LEN: usize, const N: usize> LUMatrix<LEN, N> {
    /// # Woodbury Identity
    /// (A + UCV)⁻¹p = (A⁻¹)p - (A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹)p
    /// 
    /// # Breakdown
    /// x0   = `A⁻¹ * p`
    /// Au   = `A⁻¹ * U`
    /// M_lu = (C⁻¹ + VA⁻¹U)⁻¹   = `(C⁻¹ + VAu)⁻¹`
    /// w    = VA⁻¹p             = V * A⁻¹ * p     = `Vx0`
    /// x1   = Au * M_lu * VA⁻¹p = Au * M_lu * Vx0 = `Au * M_lu * w`
    /// res  = `x0 - x1`
    pub fn solve_vec_with_woodbury<
        const LEN_U: usize,
        const K: usize,
        const LEN_C: usize
    >(
        &self,
        rhs: &VecF<N>,
        u: &Matrix<LEN_U, N, K>,
        v: &Matrix<LEN_U, K, N>,
        c_inv: &Matrix<LEN_C, K>,
    ) -> VecF<N> {
        let x0 = self.solve_vec(rhs);

        let mut a_u = Matrix::ZERO;
        self.solve_mat_into(u, &mut a_u); 
        let m = c_inv + v.matmul(&a_u);
        let mut m_lu = LUMatrix::INIT;
        if !m_lu.factorize_from(&m) { return x0 }

        let w = v * x0;
        let x1 = a_u * m_lu.solve_vec(&w);

        x0 - x1
    }
}
