//! # Schur Complement
//! 
//! │ A B │.│x│ = │p│ 
//! │ C D │ │y│   │q│
//!
//! Ax + By = p  =>  x = A⁻¹(p - By)
//! Cx + Dy = q
//! 
//! ### step1: solve y
//! 
//! C(A⁻¹(p - By)) + Dy = q
//! CA⁻¹p - CA⁻¹By + Dy = q
//! 
//! (D - CA⁻¹B)y = q - CA⁻¹p
//! q' = q - CA⁻¹p
//! 
//! Sy = q'  =>  y = (D - CA⁻¹B)⁻¹(q - CA⁻¹p)
//! 
//! ### step2: solve x
//! 
//! x = A⁻¹(p - By)

use crate::vector::VecF;
use super::heap_matrix::HeapMatrix;
use super::heap_lu::HeapLU;

pub struct HeapSchur<const M: usize, const N: usize> {
    pub a: HeapLU<M>,
    pub b: HeapMatrix<M, N>,
    pub c: HeapMatrix<N, M>,
    pub d: HeapMatrix<N>,

    pub s: HeapLU<N>,
    a_inv_b: HeapMatrix<M, N>,

    pub x: VecF<M>,
    pub y: VecF<N>,

    pub p: VecF<M>,
    pub q: VecF<N>,
}

impl<const M: usize, const N: usize> HeapSchur<M, N> {
    #[inline]
    pub fn compute_schur(&mut self) -> bool {
        self.a.solve_mat_into(&self.b, &mut self.a_inv_b);
        self.d.sub_mat_mul_into(&self.c, &self.a_inv_b, &mut self.s.matrix);
        self.s.factorize()
    }

    #[inline]
    pub fn solve(&mut self) {
        let mut a_inv_p = VecF::zero();
        let mut q_ca_inv_p = VecF::zero();
        self.a.solve_vec_into(&self.p, &mut a_inv_p);
        self.q.sub_mat_vec_mul_into(&self.c, &a_inv_p, &mut q_ca_inv_p);
        self.s.solve_vec_into(&q_ca_inv_p, &mut self.y);
        a_inv_p.sub_mat_vec_mul_into(&self.a_inv_b, &self.y, &mut self.x);
    }
}
