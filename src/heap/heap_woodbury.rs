use crate::vector::VecF;
use crate::traits::Container;
use super::heap_matrix::HeapMatrix;
use super::heap_lu::HeapLU;

// =============================================================================
// Woodbury:
//     (A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹
// =============================================================================

#[derive(Debug)]
pub struct HeapWoodbury<const NUM_A: usize, const NUM_C: usize> {
    pub a_inv: HeapLU<NUM_A>,
    pub u: HeapMatrix<NUM_A, NUM_C>,
    pub v: HeapMatrix<NUM_C, NUM_A>,
    pub c_inv: HeapMatrix<NUM_C>,

    a_u: HeapMatrix<NUM_A, NUM_C>,
    v_au: HeapMatrix<NUM_C>,
    c_vau: HeapLU<NUM_C>,
}

impl<const NUM_A: usize, const NUM_C: usize> HeapWoodbury<NUM_A, NUM_C> {
    pub fn new() -> Self {
        Self {
            a_inv: HeapLU::new(),
            u: HeapMatrix::zero(),
            v: HeapMatrix::zero(),
            c_inv: HeapMatrix::zero(),

            a_u: HeapMatrix::zero(),
            v_au: HeapMatrix::zero(),
            c_vau: HeapLU::new(),
        }
    }

    /// In most cases, this will be called once at the initialization.
    /// Make sure [`a_inv`](Self::a_inv) is already set, otherwise this won't work.
    pub fn prepare_uv<U, V>(&mut self, u: &U, v: &V)
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

    pub fn update<C: Container<NUM_C, NUM_C>>(&mut self, c_inv: &C) {
        self.c_inv.copy_from_container(c_inv);
    }

    #[inline]
    pub fn factorize_update(&mut self) -> bool {
        self.c_inv.mat_add_into(&self.v_au, &mut self.c_vau.matrix);
        self.c_vau.factorize()
    }

    #[inline]
    pub fn solve_vec_into(
        &mut self,
        b: &VecF<NUM_A>,
        res: &mut VecF<NUM_A>,
    ) {
        let x0 = self.a_inv.solve_vec(b);
        let mut v_x0 = VecF::ZERO;
        self.v.mat_vec_mul_into(&x0, &mut v_x0);
        let x1 = self.c_vau.solve_vec(&v_x0);
        x0.sub_mat_vec_mul_into(&self.a_u, &x1, res);
    }
}

// =============================================================================
// Index & IndexMut
// =============================================================================

impl<const N: usize, const C: usize> core::ops::Index<usize> for HeapWoodbury<N, C> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.a_inv.matrix.data.get_unchecked(index) }
    }
}

impl<const N: usize, const C: usize> core::ops::IndexMut<usize> for HeapWoodbury<N, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.a_inv.matrix.data.get_unchecked_mut(index) }
    }
}
