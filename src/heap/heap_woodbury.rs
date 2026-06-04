use super::heap_matrix::HeapMatrix;
use super::heap_matrix::HeapVector;
use super::heap_lu::HeapLU;

// =============================================================================
// Woodbury:
//     (A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹
// =============================================================================

#[derive(Debug)]
pub struct HeapWoodbury<const N: usize, const C: usize> {
    pub a_inv: HeapLU<N>,
    pub u: HeapMatrix<N, C>,
    pub v: HeapMatrix<C, N>,
    pub c_inv: HeapMatrix<C>,

    a_u: HeapMatrix<N, C>,
    c_vau: HeapLU<C>,
    v_au: HeapMatrix<C>,
    c_vau_v: HeapMatrix<C, N>,
    au_c_vau_v: HeapMatrix<N>,

    tmp_x1: HeapVector<N>,

    pub is_dirty: bool,
}

impl<const N: usize, const C: usize> HeapWoodbury<N, C> {
    pub fn new() -> Self {
        Self {
            a_inv: HeapLU::new(),
            u: HeapMatrix::zero(),
            v: HeapMatrix::zero(),
            c_inv: HeapMatrix::zero(),

            a_u: HeapMatrix::zero(),
            c_vau: HeapLU::new(),
            v_au: HeapMatrix::zero(),
            c_vau_v: HeapMatrix::zero(),
            au_c_vau_v: HeapMatrix::zero(),

            tmp_x1: HeapVector::zero(),

            is_dirty: true,
        }
    }

    pub fn initialize(&mut self) -> bool {
        if !self.a_inv.factorize() { return false }
        self.a_inv.solve_mat(&self.u, &mut self.a_u);
        self.v.mat_mul_into(&self.a_u, &mut self.v_au);
        true
    }

    pub fn update_c_inv(&mut self, val: f64, offset: usize) {
        self.c_inv[offset * C + offset] = val;
        self.is_dirty = true;
    }

    pub fn factorize_update(&mut self) -> bool {
        self.c_inv.mat_add_into(&self.v_au, &mut self.c_vau.matrix);
        if !self.c_vau.factorize() { return false }

        self.c_vau.solve_mat(&self.v, &mut self.c_vau_v);
        self.a_u.mat_mul_into(&self.c_vau_v, &mut self.au_c_vau_v);

        self.is_dirty = false;
        true
    }

    pub fn solve_vec_into(&mut self, b: &HeapVector<N>, x: &mut HeapVector<N>) {
        // (A + UCV)⁻¹b = A⁻¹b - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹b
        // 
        // x0 = a * b
        // x1 = au * c_vau * v * x0
        // x0 - x1

        self.a_inv.solve_vec(b, x);
        self.au_c_vau_v.mat_vec_mul_into(x, &mut self.tmp_x1);
        x.vec_sub_assign(&self.tmp_x1);
    }

    pub fn solve_mat_into<const COL2: usize>(
        &self,
        b: &HeapMatrix<N, COL2>,
        x: &mut HeapMatrix<N, COL2>,
        tmp_x1: &mut HeapMatrix<N, COL2>,
    ) {
        // (A + UCV)⁻¹b = A⁻¹b - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹b
        // 
        // x0 = a * b
        // x1 = au * c_vau * v * x0
        // x0 - x1

        self.a_inv.solve_mat(b, x);
        self.au_c_vau_v.mat_mul_into(x, tmp_x1);
        x.mat_sub_assign(tmp_x1);
    }
}

// =============================================================================
// Partitioned Woodbury
// =============================================================================

#[derive(Debug)]
pub struct HeapWoodburyPartitioned<
    const N: usize,
    const C1: usize,
    const C2: usize,
> {
    pub a0: HeapLU<N>,
    pub u1: HeapMatrix<N, C1>,
    pub u2: HeapMatrix<N, C2>,
    pub v1: HeapMatrix<C1, N>,
    pub v2: HeapMatrix<C2, N>,
    pub c1_inv: HeapMatrix<C1>,
    pub c2_inv: HeapMatrix<C2>,

    a_u1: HeapMatrix<N, C1>,
    a_u2: HeapMatrix<N, C2>,

    pub v1_a_u2: HeapMatrix<C1, C2>,
    pub v2_a_u1: HeapMatrix<C2, C1>,
    pub v2_a_u2: HeapMatrix<C2>,

    pub m1_lu: HeapLU<C1>,
    pub m1_v1_a_u2: HeapMatrix<C1, C2>,

    pub s_base: HeapMatrix<C2>,
    pub schur: HeapLU<C2>,

    tmp_x0_1: HeapVector<N>,
    tmp_x0_2: HeapVector<N>,
    tmp_x1: HeapVector<C1>,
    tmp_x2: HeapVector<C2>,
    tmp_b1: HeapVector<C1>,
    tmp_b2: HeapVector<C2>,
    tmp_w: HeapVector<C1>,

    pub is_dirty: bool,
}

impl<const N: usize, const C1: usize, const C2: usize> HeapWoodburyPartitioned<N, C1, C2> {
    pub fn new() -> Self {
        Self {
            a0: HeapLU::new(),
            u1: HeapMatrix::zero(),
            u2: HeapMatrix::zero(),
            v1: HeapMatrix::zero(),
            v2: HeapMatrix::zero(),
            c1_inv: HeapMatrix::zero(),
            c2_inv: HeapMatrix::zero(),

            a_u1: HeapMatrix::zero(),
            a_u2: HeapMatrix::zero(),
            v1_a_u2: HeapMatrix::zero(),
            v2_a_u1: HeapMatrix::zero(),
            v2_a_u2: HeapMatrix::zero(),
            m1_lu: HeapLU::new(),
            m1_v1_a_u2: HeapMatrix::zero(),
            s_base: HeapMatrix::zero(),
            schur: HeapLU::new(),

            tmp_x0_1: HeapVector::zero(),
            tmp_x0_2: HeapVector::zero(),
            tmp_x1: HeapVector::zero(),
            tmp_x2: HeapVector::zero(),
            tmp_b1: HeapVector::zero(),
            tmp_b2: HeapVector::zero(),
            tmp_w: HeapVector::zero(),
            is_dirty: true,
        }
    }

    pub fn finish_preparation(&mut self) -> bool {
        if !self.a0.factorize() { return false }
        self.a0.solve_mat(&self.u1, &mut self.a_u1);
        self.a0.solve_mat(&self.u2, &mut self.a_u2);
        self.v1.mat_mul_into(&self.a_u2, &mut self.v1_a_u2);
        self.v2.mat_mul_into(&self.a_u1, &mut self.v2_a_u1);
        self.v2.mat_mul_into(&self.a_u2, &mut self.v2_a_u2);
        true
    }

    pub fn update_c1_inv(&mut self, val: f64, offset: usize) {
        self.c1_inv[offset * C1 + offset] = val;
        self.is_dirty = true;
    }

    pub fn update_c2_inv(&mut self, val: f64, offset: usize) {
        self.c2_inv[offset * C2 + offset] = val;
    }

    pub fn factorize1(&mut self) -> bool {
        self.v1.mat_mul_add_into(&self.a_u1, &self.c1_inv, &mut self.m1_lu.matrix);
        if !self.m1_lu.factorize() { return false }
        self.m1_lu.solve_mat(&self.v1_a_u2, &mut self.m1_v1_a_u2);
        self.v2_a_u1.sub_mat_mul_into(&self.m1_v1_a_u2, &self.v2_a_u2, &mut self.s_base);
        true
    }

    pub fn factorize2(&mut self) -> bool {
        self.c2_inv.mat_add_into(&self.s_base, &mut self.schur.matrix);
        self.schur.factorize()
    }

    pub fn solve_vec_into(&mut self, b: &HeapVector<N>, x: &mut HeapVector<N>) {
        self.a0.solve_vec(b, &mut self.tmp_x0_1);
        self.v1.mat_vec_mul_into(&self.tmp_x0_1, &mut self.tmp_b1);
        self.v2.mat_vec_mul_into(&self.tmp_x0_1, &mut self.tmp_b2);
        self.m1_lu.solve_vec(&self.tmp_b1, &mut self.tmp_w);
        self.v2_a_u1.sub_mat_vec_mul_into(&self.tmp_w, &self.tmp_b2, &mut self.tmp_x2);
        self.schur.solve_vec(&self.tmp_x2, &mut self.tmp_b2);
        self.m1_v1_a_u2.sub_mat_vec_mul_into(&self.tmp_b2, &self.tmp_w, &mut self.tmp_x1);
        self.a_u1.sub_mat_vec_mul_into(&self.tmp_x1, &self.tmp_x0_1, &mut self.tmp_x0_2);
        self.a_u2.sub_mat_vec_mul_into(&self.tmp_b2, &self.tmp_x0_2, x);
    }

    pub fn solve_mat_into<const COL2: usize>(
        &mut self,
        b: &HeapMatrix<N, COL2>,
        tmp_x0_1: &mut HeapMatrix<N, COL2>,
        tmp_x0_2: &mut HeapMatrix<N, COL2>,
        tmp_x1: &mut HeapMatrix<C1, COL2>,
        tmp_x2: &mut HeapMatrix<C2, COL2>,
        tmp_b1: &mut HeapMatrix<C1, COL2>,
        tmp_b2: &mut HeapMatrix<C2, COL2>,
        tmp_w: &mut HeapMatrix<C1, COL2>,
        x: &mut HeapMatrix<N, COL2>,
    ) {
        self.a0.solve_mat(b, tmp_x0_1);
        self.v1.mat_mul_into(tmp_x0_1, tmp_b1);
        self.v2.mat_mul_into(tmp_x0_1, tmp_b2);
        self.m1_lu.solve_mat(tmp_b1, tmp_w);
        self.v2_a_u1.sub_mat_mul_into(tmp_w, tmp_b2, tmp_x2);
        self.schur.solve_mat(tmp_x2, tmp_b2);
        self.m1_v1_a_u2.sub_mat_mul_into(tmp_b2, tmp_w, tmp_x1);
        self.a_u1.sub_mat_mul_into(tmp_x1, tmp_x0_1, tmp_x0_2);
        self.a_u2.sub_mat_mul_into(tmp_b2, tmp_x0_2, x);
    }
}

// =============================================================================
// Index & IndexMut
// =============================================================================

impl<const N: usize, const C: usize> core::ops::Index<usize> for HeapWoodbury<N, C> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe {
            self.a_inv.matrix.data.get_unchecked(index)
        }
    }
}

impl<const N: usize, const C: usize> core::ops::IndexMut<usize> for HeapWoodbury<N, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            self.a_inv.matrix.data.get_unchecked_mut(index)
        }
    }
}

impl<
    const N: usize,
    const C1: usize,
    const C2: usize,
> core::ops::Index<usize> for HeapWoodburyPartitioned<N, C1, C2> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe {
            self.a0.matrix.data.get_unchecked(index)
        }
    }
}

impl<
    const N: usize,
    const C1: usize,
    const C2: usize,
> core::ops::IndexMut<usize> for HeapWoodburyPartitioned<N, C1, C2> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            self.a0.matrix.data.get_unchecked_mut(index)
        }
    }
}
