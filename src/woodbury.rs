//! Woodbury: (A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹

use crate::lu::LUMatrix;
use crate::matrix::Matrix;
use crate::vector::VecF;
use crate::traits::Container;
use crate::heap::heap_matrix::HeapMatrix;

// =============================================================================
// WoodburyCache1
// =============================================================================

#[derive(Debug)]
/// Ideal for less frequent updates on C⁻¹
/// Leveraging on cached I - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹V
pub struct WoodburyCache1<
    const LEN_A: usize, const NUM_A: usize,
    const LEN_C: usize, const NUM_C: usize,
> {
    /// base A⁻¹
    pub a_inv: LUMatrix<LEN_A, NUM_A>,
    /// A⁻¹U where U = Vᵀ
    a_u: HeapMatrix<NUM_A, NUM_C>,
    /// incident matrix
    v: HeapMatrix<NUM_C, NUM_A>,
    /// cached VA⁻¹U
    v_au: Matrix<LEN_C, NUM_C>,
    /// cached I - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹V
    i_au_c_vau_v: Matrix<LEN_A, NUM_A>,
}

impl<
    const LEN_A: usize, const NUM_A: usize,
    const LEN_C: usize, const NUM_C: usize,
> WoodburyCache1<LEN_A, NUM_A, LEN_C, NUM_C> {
    pub fn new() -> Self {
        Self {
            a_inv: LUMatrix::INIT,
            a_u: HeapMatrix::zero(),
            v: HeapMatrix::zero(),
            v_au: Matrix::ZERO,
            i_au_c_vau_v: Matrix::ZERO,
        }
    }

    /// In most cases, this will be called once at the initialization.
    /// Make sure [`a`](Self::a) is already set, otherwise this won't work.
    pub fn prepare<U, V>(&mut self, u: &U, v: &V) -> bool
    where
        U: Container<NUM_A, NUM_C>,
        V: Container<NUM_C, NUM_A>,
    {
        if !self.a_inv.factorize() { return false }
        self.a_inv.solve_mat_into(u, &mut self.a_u);
        self.v.copy_from_container(v);
        self.v.mat_mul_into(&self.a_u, &mut self.v_au);
        true
    }

    #[inline]
    /// I - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹V
    pub fn factorize_update<const LEN_UV: usize>(&mut self, c_inv: &Matrix<LEN_C, NUM_C>) -> bool {
        let mut c_vau = LUMatrix::<LEN_C, NUM_C, NUM_C>::INIT;
        c_inv.mat_add_into(&self.v_au, &mut c_vau.matrix);
        if !c_vau.factorize() { return false }
        let c_vau_v = c_vau.solve_mat::<LEN_UV, _, _>(&self.v);
        self.a_u.identity_sub_mat_mul_into(&c_vau_v, &mut self.i_au_c_vau_v);
        true
    }

    /// (A + UCV)⁻¹b
    /// = A⁻¹b - (A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹V)A⁻¹b
    /// = (I - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹V) * A⁻¹b
    #[inline]
    pub fn solve_vec_into(&self, b: &VecF<NUM_A>, res: &mut VecF<NUM_A>) {
        let mut tmp_x = VecF::ZERO;
        self.a_inv.solve_vec_into(b, &mut tmp_x);
        self.i_au_c_vau_v.mat_vec_mul_into(&tmp_x, res);
    }

    /// (A + UCV)⁻¹b
    /// = A⁻¹b - (A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹V)A⁻¹b
    /// = (I - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹V) * A⁻¹b
    #[inline]
    pub fn solve_mat_into<
        const LEN_2: usize,
        const COL_2: usize,
    >(
        &self,
        b: &Matrix<LEN_2, NUM_A, COL_2>,
        res: &mut Matrix<LEN_2, NUM_A, COL_2>,
    ) {
        let mut tmp_x: Matrix<LEN_2, NUM_A, COL_2> = Matrix::ZERO;
        self.a_inv.solve_mat_into(b, &mut tmp_x);
        self.i_au_c_vau_v.mat_mul_into(&tmp_x, res);
    }
}

// =============================================================================
// WoodburyCache2
// =============================================================================

#[derive(Debug)]
/// Ideal for frequent updates on C⁻¹
pub struct WoodburyCache2<
    const LEN_A: usize, const NUM_A: usize,
    const LEN_C: usize, const NUM_C: usize,
> {
    /// base A⁻¹
    pub a_inv: LUMatrix<LEN_A, NUM_A>,
    /// transposed incident matrix
    u: HeapMatrix<NUM_A, NUM_C>,
    /// incident matrix
    v: HeapMatrix<NUM_C, NUM_A>,
    /// A⁻¹U where U = Vᵀ
    a_u: HeapMatrix<NUM_A, NUM_C>,
    /// cached VA⁻¹U
    v_au: Matrix<LEN_C, NUM_C>,
    c_vau: LUMatrix<LEN_C, NUM_C>,
}

impl<
    const LEN_A: usize, const NUM_A: usize,
    const LEN_C: usize, const NUM_C: usize,
> WoodburyCache2<LEN_A, NUM_A, LEN_C, NUM_C> {
    pub fn new() -> Self {
        Self {
            a_inv: LUMatrix::INIT,
            u: HeapMatrix::zero(),
            v: HeapMatrix::zero(),
            a_u: HeapMatrix::zero(),
            v_au: Matrix::ZERO,
            c_vau: LUMatrix::INIT,
        }
    }

    /// In most cases, this will be called once at the initialization.
    /// Make sure [`a`](Self::a) is already set, otherwise this won't work.
    pub fn prepare<U, V>(&mut self, u: &U, v: &V)
    where
        U: Container<NUM_A, NUM_C>,
        V: Container<NUM_C, NUM_A>,
    {
        self.u.copy_from_container(u);
        self.v.copy_from_container(v);
    }

    #[inline]
    pub fn initialize(&mut self) -> bool {
        if !self.a_inv.factorize() { return false }
        self.a_inv.solve_mat_into(&self.u, &mut self.a_u);
        self.v.mat_mul_into(&self.a_u, &mut self.v_au);
        true
    }

    #[inline]
    pub fn update<C: Container<NUM_C, NUM_C>>(&mut self, c_inv: &C) -> bool {
        self.v_au.mat_add_into(c_inv, &mut self.c_vau.matrix);
        self.c_vau.factorize()
    }

    #[inline]
    pub fn solve_vec_into(
        &self,
        b: &VecF<NUM_A>,
        res: &mut VecF<NUM_A>
    ) {
        let x0 = self.a_inv.solve_vec(b);
        let mut v_x0 = VecF::ZERO;
        self.v.mat_vec_mul_into(&x0, &mut v_x0);
        let x1 = self.c_vau.solve_vec(&v_x0);
        x0.sub_mat_vec_mul_into(&self.a_u, &x1, res);
    }

    #[inline]
    pub fn solve_mat_into<
        const NUM_C_X_COL2: usize,
        const LEN_2: usize,
        const COL_2: usize,
    >(
        &self,
        b: &Matrix<LEN_2, NUM_A, COL_2>,
        res: &mut Matrix<LEN_2, NUM_A, COL_2>,
    ) {
        let y: Matrix<LEN_2, NUM_A, COL_2> = self.a_inv.solve_mat(b);
        let mut vy: Matrix<NUM_C_X_COL2, NUM_C, COL_2> = Matrix::ZERO;
        self.v.mat_mul_into(&y, &mut vy);
        let x = self.c_vau.solve_mat::<NUM_C_X_COL2, _, _>(&vy);
        y.sub_mat_mul_into(&self.a_u, &x, res);
    }
}

// =============================================================================
// Index & IndexMut
// =============================================================================

impl<
    const LEN_A: usize, const NUM_A: usize,
    const LEN_C: usize, const NUM_C: usize,
> core::ops::Index<usize> for WoodburyCache1<LEN_A, NUM_A, LEN_C, NUM_C> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.a_inv.matrix.data.get_unchecked(index) }
    }
}

impl<
    const LEN_A: usize, const NUM_A: usize,
    const LEN_C: usize, const NUM_C: usize,
> core::ops::IndexMut<usize> for WoodburyCache1<LEN_A, NUM_A, LEN_C, NUM_C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            self.a_inv.matrix.data.get_unchecked_mut(index)
        }
    }
}

impl<
    const LEN_A: usize, const NUM_A: usize,
    const LEN_C: usize, const NUM_C: usize,
> core::ops::Index<usize> for WoodburyCache2<LEN_A, NUM_A, LEN_C, NUM_C> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.a_inv.matrix.data.get_unchecked(index) }
    }
}

impl<
    const LEN_A: usize, const NUM_A: usize,
    const LEN_C: usize, const NUM_C: usize,
> core::ops::IndexMut<usize> for WoodburyCache2<LEN_A, NUM_A, LEN_C, NUM_C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.a_inv.matrix.data.get_unchecked_mut(index) }
    }
}
