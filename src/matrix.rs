//! Blocked LU decomposition with partial pivoting.
//!
//! # Quick start
//!
//! ```rust
//! mat!(Mat11, 11);   // 11×11, 4×4 blocks → 3×3 = 9 blocks
//! mat!(Mat20, 20);   // 20×20, 4×4 blocks → 5×5 = 25 blocks
//!
//! let src  = Mat11::from_flat(&my_floats);
//! let lu   = lu_decompose(&src);
//! // invariant: P * src ≈ lu.l * lu.u
//! ```
//!
//! # Layout
//!
//! Each matrix is stored as a `num_tiles(N) × num_tiles(N)` grid of 4×4 `Block`s.
//! Every `Block` is exactly one 64-byte cache line.
//! Padding blocks (for N that isn't a multiple of 4) stay zero and are
//! never written to during factorisation.

use crate::vector::VecF64;

const B: usize = 4;
const BS: usize = B * B;

#[repr(align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    pub data: [f64; 16],
}

impl Block {
    pub const ZERO: Self = Self { data: [0f64; 16] };
    pub const IDENTITY: Self = {
        let mut d = [0f64; 16];
        d[0] = 1.0;
        d[5] = 1.0;
        d[10] = 1.0;
        d[15] = 1.0;
        Self { data: d }
    };

    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        // 4 comes from 4x4 block
        unsafe { *self.data.get_unchecked(row * 4 + col) }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        // 4 comes from 4x4 block
        unsafe { self.data.get_unchecked_mut(row * 4 + col) }
    }

    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        // 4 comes from 4x4 block
        unsafe { *self.data.get_unchecked_mut(row * 4 + col) = val }
    }

    #[inline(always)]
    pub fn mulsub_assign(&mut self, a: &Self, b: &Self) {
        // 4 comes from 4x4 block
        for k in 0..4 {
            let b_row = [b.get(k, 0), b.get(k, 1), b.get(k, 2), b.get(k, 3)];
            for i in 0..4 {
                let aik = a.get(i, k);
                self.data[i * 4] -= aik * b_row[0];
                self.data[i * 4 + 1] -= aik * b_row[1];
                self.data[i * 4 + 2] -= aik * b_row[2];
                self.data[i * 4 + 3] -= aik * b_row[3];
            }
        }
    }

    #[inline(always)]
    pub fn muladd_assign(&mut self, a: &Self, b: &Self) {
        for k in 0..4 {
            let b_row = [b.get(k, 0), b.get(k, 1), b.get(k, 2), b.get(k, 3)];
            for i in 0..4 {
                let aik = a.get(i, k);
                self.data[i * 4] += aik * b_row[0];
                self.data[i * 4 + 1] += aik * b_row[1];
                self.data[i * 4 + 2] += aik * b_row[2];
                self.data[i * 4 + 3] += aik * b_row[3];
            }
        }
    }

    #[inline(always)]
    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 {
            return;
        }
        for c in 0..4 {
            self.data.swap(r1 * 4 + c, r2 * 4 + c);
        }
    }

    #[inline(always)]
    pub fn swap_rows_with(&mut self, r1: usize, other: &mut Self, r2: usize) {
        for c in 0..4 {
            let tmp = self.data[r1 * 4 + c];
            self.data[r1 * 4 + c] = other.data[r2 * 4 + c];
            other.data[r2 * 4 + c] = tmp;
        }
    }
}

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct Matrix<const N: usize, const GRID: usize> {
    pub blocks: [Block; GRID],
}

impl<const N: usize, const GRID: usize> Matrix<N, GRID> {
    pub const TILES: usize = crate::num_tiles(N);

    pub const ZERO: Self = Self { blocks: [Block::ZERO; GRID] };

    pub const IDENTITY: Self = {
        let mut m = Self::ZERO;
        let mut i = 0;
        while i < Self::TILES {
            m.blocks[i * Self::TILES + i] = Block::IDENTITY;
            i += 1
        };
        m
    };

    /// Construct the identity matrix at runtime (const IDENTITY was buggy; use this).
    pub fn identity() -> Self {
        let mut m = Self::ZERO;
        for t in 0..Self::TILES {
            *m.block_mut(t, t) = Block::IDENTITY;
        }
        m
    }

    pub fn from_flat(data: &[f64]) -> Self {
        debug_assert_eq!(data.len(), N * N, "from_flat: expected N*N elements");
        let mut m = Self::ZERO;
        for i in 0..N {
            for j in 0..N {
                m.set(i, j, data[i * N + j]);
            }
        }
        m
    }

    #[inline(always)]
    pub fn block(&self, br: usize, bc: usize) -> &Block {
        unsafe { self.blocks.get_unchecked(br * Self::TILES + bc) }
    }

    #[inline(always)]
    pub fn block_mut(&mut self, br: usize, bc: usize) -> &mut Block {
        unsafe { self.blocks.get_unchecked_mut(br * Self::TILES + bc) }
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.block(i / 4, j / 4).get(i % 4, j % 4)
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        *self.block_mut(i / 4, j / 4).get_mut(i % 4, j % 4) = val;
    }

    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 {
            return;
        }
        let (br1, lr1) = (r1 / 4, r1 % 4);
        let (br2, lr2) = (r2 / 4, r2 % 4);
        let tiles = Self::TILES;
        if br1 == br2 {
            for bc in 0..tiles {
                self.block_mut(br1, bc).swap_rows(lr1, lr2);
            }
        } else {
            for bc in 0..tiles {
                let i1 = br1 * tiles + bc;
                let i2 = br2 * tiles + bc;
                unsafe {
                    let p1 = self.blocks.get_unchecked_mut(i1) as *mut Block;
                    let p2 = self.blocks.get_unchecked_mut(i2) as *mut Block;
                    (*p1).swap_rows_with(lr1, &mut *p2, lr2);
                }
            }
        }
    }

    pub fn lu_decompose(&self) -> LUDecomp<N, GRID> {
        LUDecomp::new(self)
    }
}

pub struct LUDecomp<const N: usize, const GRID: usize> {
    pub l: Matrix<N, GRID>,
    pub u: Matrix<N, GRID>,
    /// perm[i] = original row index that ended up at row i after pivoting.
    pub perm: [usize; N],
}

/// Solve L * X = B in-place for a single 4x4 lower-unit-triangular block.
/// Forward substitution, column-major inner loop.
#[inline]
fn trsm_left_lower(l_blk: &Block, target: &mut Block) {
    for j in 0..4 {
        for k in 0..4 {
            let tkj = target.get(k, j);
            for i in k + 1..4 {
                *target.get_mut(i, j) -= l_blk.get(i, k) * tkj;
            }
        }
    }
}

/// Solve X * U = B in-place for a single 4x4 upper-triangular block.
/// Back substitution: process columns right-to-left.
#[inline]
fn trsm_right_upper(u_blk: &Block, target: &mut Block) {
    for i in 0..4 {
        for k in (0..4).rev() {
            let inv_ukk = 1.0 / u_blk.get(k, k);
            let tik = target.get(i, k) * inv_ukk;
            *target.get_mut(i, k) = tik;
            // Subtract known contributions from columns to the left of k
            for j in 0..k {
                *target.get_mut(i, j) -= tik * u_blk.get(k, j);
            }
        }
    }
}

impl<const N: usize, const GRID: usize> LUDecomp<N, GRID> {
    pub fn new(src: &Matrix<N, GRID>) -> Self {
        let tiles = Matrix::<N, GRID>::TILES;
        let mut a = *src;
        let mut perm: [usize; N] = core::array::from_fn(|i| i);

        for k in 0..tiles {
            let row_base = k * 4;
            let tile_rows = (N - row_base).min(4);

            for step in 0..tile_rows {
                let global_step = row_base + step;

                let mut max_abs = a.get(global_step, global_step).abs();
                let mut pivot_global = global_step;

                // Search all rows below global_step in this column
                for i in global_step + 1..N {
                    let v = a.get(i, global_step).abs();
                    if v > max_abs {
                        max_abs = v;
                        pivot_global = i;
                    }
                }

                if pivot_global != global_step {
                    a.swap_rows(global_step, pivot_global);
                    perm.swap(global_step, pivot_global);
                }

                // Eliminate within this column, rows global_step+1 .. row_base+tile_rows
                let pivot = a.get(global_step, global_step);
                if pivot == 0.0 {
                    continue;
                }
                let inv_pivot = 1.0 / pivot;

                for i in global_step + 1..row_base + tile_rows {
                    let m = a.get(i, global_step) * inv_pivot;
                    a.set(i, global_step, m);
                    for j in global_step + 1..row_base + 4 {
                        let u = a.get(global_step, j);
                        a.set(i, j, a.get(i, j) - m * u);
                    }
                }
            }

            // trsm: update blocks to the right of the diagonal (solve L * U_kj = A_kj)
            for j in k + 1..tiles {
                let diag_idx = k * tiles + k;
                let tgt_idx = k * tiles + j;
                debug_assert!(diag_idx < tgt_idx);
                let (left, right) = a.blocks.split_at_mut(tgt_idx);
                trsm_left_lower(&left[diag_idx], &mut right[0]);
            }

            // trsm: update blocks below the diagonal (solve L_ik * U = A_ik)
            for i in k + 1..tiles {
                let diag_idx = k * tiles + k;
                let tgt_idx = i * tiles + k;
                debug_assert!(diag_idx < tgt_idx);
                let (left, right) = a.blocks.split_at_mut(tgt_idx);
                trsm_right_upper(&left[diag_idx], &mut right[0]);
            }

            // Schur complement update for trailing submatrix
            for i in k + 1..tiles {
                let ik_idx = i * tiles + k;
                let aik = a.blocks[ik_idx];
                for j in k + 1..tiles {
                    let kj_idx = k * tiles + j;
                    let ij_idx = i * tiles + j;
                    debug_assert!(kj_idx != ij_idx);
                    let (left, right) = a.blocks.split_at_mut(ij_idx);
                    right[0].mulsub_assign(&aik, &left[kj_idx]);
                }
            }
        }

        // --- Extract L and U from combined storage ---
        let mut l = Matrix::<N, GRID>::IDENTITY;
        let mut u = Matrix::<N, GRID>::ZERO;

        for bi in 0..tiles {
            for bj in 0..tiles {
                let blk = a.block(bi, bj);
                if bi < bj {
                    // Entirely above diagonal: belongs to U
                    *u.block_mut(bi, bj) = *blk;
                } else if bi > bj {
                    // Entirely below diagonal: belongs to L
                    *l.block_mut(bi, bj) = *blk;
                } else {
                    // Diagonal block: split by r <= c (U) vs r > c (L)
                    let ub = u.block_mut(bi, bj);
                    let lb = l.block_mut(bi, bj);
                    for r in 0..4 {
                        for c in 0..4 {
                            let v = blk.get(r, c);
                            if r <= c {
                                ub.set(r, c, v);
                            } else {
                                lb.set(r, c, v);
                                // L diagonal stays 1.0 from identity() above
                            }
                        }
                    }
                }
            }
        }

        LUDecomp { l, u, perm }
    }

    /// Solve A*x = b using the decomposition P*A = L*U.
    /// Apply permutation to b first, then forward/back substitute.
    pub fn solve<const PAD: usize>(&self, b: &VecF64<N, PAD>) -> VecF64<N, PAD> {
        // Apply row permutation
        let mut data: [f64; PAD] = [0.0; PAD];
        for i in 0..N {
            data[i] = b[self.perm[i]];
        }

        // Forward substitution: solve L*y = Pb
        for i in 0..N {
            for j in 0..i {
                data[i] -= self.l.get(i, j) * data[j];
            }
            // L diagonal is 1, no division needed
        }

        // Back substitution: solve U*x = y
        for i in (0..N).rev() {
            for j in i + 1..N {
                data[i] -= self.u.get(i, j) * data[j];
            }
            data[i] /= self.u.get(i, i);
        }

        VecF64 { data }
    }
}

impl<const N: usize, const GRID: usize> core::fmt::Debug for Matrix<N, GRID> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Matrix<{N}x{N}> => {GRID} blocks")?;
        for i in 0..N {
            write!(f, "  │")?;
            for j in 0..N { write!(f, "{:6.2}", self.get(i, j))? }
            writeln!(f, "  │")?;
        }
        Ok(())
    }
}

impl<const N: usize, const GRID: usize> core::fmt::Debug for LUDecomp<N, GRID> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L: {:?}", self.l)?;
        write!(f, "U: {:?}", self.u)
    }
}

impl<const N: usize, const GRID: usize> PartialEq for Matrix<N, GRID> {
    fn eq(&self, other: &Self) -> bool {
        self.blocks.iter()
            .zip(&other.blocks)
            .all(|(a, b)| a == b)
    }
}

impl<const N: usize, const GRID: usize> Eq for Matrix<N, GRID> {}

impl<const N: usize, const GRID: usize> core::ops::Mul<Self> for Matrix<N, GRID> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut c = Matrix::<N, GRID>::ZERO;
        for i in 0..N {
            for k in 0..N {
                let aik = self.get(i, k);
                if aik == 0.0 {
                    continue;
                }
                for j in 0..N {
                    c.set(i, j, c.get(i, j) + aik * rhs.get(k, j));
                }
            }
        }
        c
    }
}

impl<'a, const N: usize, const GRID: usize> core::ops::Mul<Self> for &'a Matrix<N, GRID> {
    type Output = Matrix::<N, GRID>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut c = Matrix::<N, GRID>::ZERO;
        for i in 0..N {
            for k in 0..N {
                let aik = self.get(i, k);
                if aik == 0.0 {
                    continue;
                }
                for j in 0..N {
                    c.set(i, j, c.get(i, j) + aik * rhs.get(k, j));
                }
            }
        }
        c
    }
}

impl<const N: usize, const GRID: usize, const PAD: usize> core::ops::Mul<VecF64<N, PAD>> for Matrix<N, GRID> {
    type Output = VecF64<N, PAD>;

    fn mul(self, rhs: VecF64<N, PAD>) -> VecF64<N, PAD> {
        let tiles = Self::TILES;
        let x = &rhs.data;
        let m = &self.blocks;
        let mut out = VecF64::ZERO;

        for br in 0..tiles {
            // Keep B scalar accumulators — but initialize as an array
            // so the compiler sees them as a unit and packs into one vector reg
            let mut acc = [0.0f64; B];

            for bc in 0..tiles {
                let block = &m[br * tiles + bc];
                let xb = &x[bc * B..];

                // Unroll k (the column index within the tile) explicitly
                // Each row of the block is a contiguous [f64; B]
                // Writing it this way makes the fmla structure obvious:
                // acc[row] += dot(block_row[row], xb)
                for row in 0..B {
                    let r = &block.data[row * B..row * B + B];
                    // This is a 4-wide dot product — should emit fmul+fmla×3
                    acc[row] += r[0]*xb[0] + r[1]*xb[1] + r[2]*xb[2] + r[3]*xb[3];
                }
            }

            let base = br * B;
            for row in 0..B {
                out[base + row] = acc[row];
            }
        }

        out
    }
}

#[cfg(test)]
mod matrix_test {
    use super::*;
    use crate::{mat, vecf64};

    #[test]
    fn mat_mul() {
        let mat = mat!((11) =>
            3, 0, 1, 4, 9, 8, 0, 6, 9, 9, 2,
            1, 1, 1, 5, 0, 7, 3, 1, 9, 3, 7,
            5, 5, 0, 8, 0, 8, 5, 7, 0, 0, 2,
            6, 0, 9, 3, 5, 1, 5, 1, 3, 5, 9,
            6, 5, 6, 4, 6, 9, 7, 0, 9, 6, 9,
            8, 8, 7, 1, 7, 7, 4, 8, 1, 3, 9,
            5, 4, 4, 5, 2, 1, 6, 0, 6, 6, 4,
            6, 0, 4, 3, 9, 3, 5, 3, 3, 9, 4,
            1, 7, 6, 4, 3, 1, 0, 3, 2, 9, 9,
            3, 5, 5, 0, 0, 2, 8, 6, 9, 5, 9,
            0, 5, 9, 3, 0, 8, 1, 3, 1, 8, 2,
        );
        let vec = vecf64!((11, 4) => 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let mul = mat * vec;
        println!("{mul:?}");
    }
}
