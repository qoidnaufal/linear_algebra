use crate::heap::heap_matrix::HeapMatrix;

#[derive(Debug)]
pub struct CSRMatrix<const ROW: usize, const COL: usize = ROW> {
    offsets: Box<[usize]>,
    cols: Box<[usize]>,
    data: Box<[f64]>,
}

// TODO: handle empty row (necessary?),
// currently this assumes each rows have at least 1 non-zero value
impl<const ROW: usize, const COL: usize> CSRMatrix<ROW, COL> {
    pub fn new(mut triplets: Vec<(usize, usize, f64)>) -> Result<Self, SparseMatrixError> {
        if triplets.len() > ROW * COL { return Err(SparseMatrixError::InvalidTriplets) }
        triplets.sort_by(|t1, t2| t1.0.cmp(&t2.0));
        let mut current_row = 0;
        let mut offsets = vec![0];
        let mut cols = vec![];
        let mut data = vec![];
        for (row, col, val) in &triplets {
            if *row != current_row {
                offsets.push(cols.len());
                current_row = *row;
            }
            cols.push(*col);
            data.push(*val);
        }
        Ok(Self {
            offsets: offsets.into_boxed_slice(),
            cols: cols.into_boxed_slice(),
            data: data.into_boxed_slice(),
        })
    }

    pub fn to_dense(&self) -> HeapMatrix<ROW, COL> {
        let mut dense = HeapMatrix::zero();
        for row in 0..ROW {
            let start = self.offsets[row];
            let end = *self.offsets
                .get(row + 1)
                .unwrap_or(&self.cols.len());
            self.cols[start..end]
                .iter()
                .zip(&self.data[start..end])
                .for_each(|(&col, &val)| dense[row * COL + col] = val);
        }
        dense
    }
}

#[derive(Debug)]
pub enum SparseMatrixError {
    InvalidTriplets
}

impl core::fmt::Display for SparseMatrixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::error::Error for SparseMatrixError {}

#[cfg(test)]
mod sparse_test {
    use super::*;

    #[test]
    fn debug_matrix() {
        let sparse = CSRMatrix::<3, 3>::new(vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (0, 2, 3.0),
            (1, 1, 5.0),
            (2, 0, 7.0),
            (2, 2, 9.0)
        ]);

        assert!(sparse.is_ok());
        println!("{sparse:?}");

        let dense = sparse.unwrap().to_dense();

        println!("{dense:?}");
    }
}
