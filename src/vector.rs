#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct VecF32<const N: usize, const PAD: usize> {
    pub data: [f32; PAD]
}

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct VecF64<const N: usize, const PAD: usize> {
    pub data: [f64; PAD]
}

macro_rules! impl_vec {
    ($name:ident => $ty:ty) => {
        impl<const N: usize, const PAD: usize> Default for $name<N, PAD> {
            fn default() -> Self {
                Self::ZERO
            }
        }

        impl<const N: usize, const PAD: usize> $name<N, PAD> {
            pub const ZERO: Self = Self { data: [0 as $ty; PAD] };

            pub fn map<R>(self, f: impl FnMut($ty) -> R) -> [R; PAD] {
                self.data.map(f)
            }
        }

        impl<const N: usize, const PAD: usize> PartialEq for $name<N, PAD> {
            fn eq(&self, other: &Self) -> bool {
                self.data[..N].iter()
                    .zip(&other.data[..N])
                    .all(|(a, b)| a == b)
            }
        }

        impl<const N: usize, const PAD: usize> std::ops::Index<usize> for $name<N, PAD> {
            type Output = $ty;

            fn index(&self, index: usize) -> &Self::Output {
                &self.data[index]
            }
        }

        impl<const N: usize, const PAD: usize> std::ops::IndexMut<usize> for $name<N, PAD> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.data[index]
            }
        }

        impl<const N: usize, const PAD: usize> std::ops::Mul<Self> for $name<N, PAD> {
            type Output = $ty;

            fn mul(self, rhs: Self) -> Self::Output {
                self.data[..N].into_iter()
                    .zip(&rhs.data[..N])
                    .map(|(a, b)| a * *b)
                    .sum()
            }
        }

        impl<const N: usize, const PAD: usize> std::ops::Mul<[$ty; N]> for $name<N, PAD> {
            type Output = $ty;

            fn mul(self, rhs: [$ty; N]) -> Self::Output {
                self.data[..N].into_iter()
                    .zip(rhs)
                    .map(|(a, b)| a * b)
                    .sum()
            }
        }

        impl<const N: usize, const PAD: usize> std::ops::Add<Self> for $name<N, PAD> {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                let mut output = Self::ZERO;
                for i in (0..PAD).step_by(4) {
                    output[i]     = self[i]     + rhs[i];
                    output[i + 1] = self[i + 1] + rhs[i + 1];
                    output[i + 2] = self[i + 2] + rhs[i + 2];
                    output[i + 3] = self[i + 3] + rhs[i + 3];
                }
                output
            }
        }

        impl<const N: usize, const PAD: usize> std::ops::AddAssign<Self> for $name<N, PAD> {
            fn add_assign(&mut self, rhs: Self) {
                self.data[..N].iter_mut()
                    .zip(&rhs.data[..N])
                    .for_each(|(a, b)| *a += b);
            }
        }

        impl<const N: usize, const PAD: usize> std::ops::Add<&Self> for $name<N, PAD> {
            type Output = Self;

            fn add(self, rhs: &Self) -> Self::Output {
                let mut output = Self::ZERO;
                for i in (0..PAD).step_by(4) {
                    output[i]     = self[i]     + rhs[i];
                    output[i + 1] = self[i + 1] + rhs[i + 1];
                    output[i + 2] = self[i + 2] + rhs[i + 2];
                    output[i + 3] = self[i + 3] + rhs[i + 3];
                }
                output
            }
        }

        impl<const N: usize, const PAD: usize> std::ops::AddAssign<&Self> for $name<N, PAD> {
            fn add_assign(&mut self, rhs: &Self) {
                self.data[..N].iter_mut()
                    .zip(&rhs.data[..N])
                    .for_each(|(a, b)| *a += *b);
            }
        }

        impl<const N: usize, const PAD: usize> std::fmt::Debug for $name<N, PAD> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_list()
                    .entries(&self.data[..N])
                    .finish()
            }
        }
    };
}

impl_vec!(VecF32 => f32);
impl_vec!(VecF64 => f64);
