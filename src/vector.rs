#[derive(Clone, Copy)]
pub struct VecF32<const N: usize> {
    pub data: [f32; N]
}

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct AlignedVecF32<const N: usize> {
    pub data: [f32; N]
}

#[derive(Clone, Copy)]
pub struct VecF64<const N: usize> {
    pub data: [f64; N]
}

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct AlignedVecF64<const N: usize> {
    pub data: [f64; N]
}

#[derive(Clone, Copy)]
pub struct VecU32<const N: usize> {
    pub data: [u32; N]
}

macro_rules! impl_vec {
    ($name:ident => $ty:ty) => {
        impl<const N: usize> Default for $name<N> {
            fn default() -> Self {
                Self::ZERO
            }
        }

        impl<const N: usize> $name<N> {
            pub const ZERO: Self = Self { data: [0 as $ty; N] };
            pub const ONES: Self = Self { data: [1 as $ty; N] };

            pub fn map<R>(self, f: impl FnMut($ty) -> R) -> [R; N] {
                self.data.map(f)
            }
        }

        impl<const N: usize> PartialEq for $name<N> {
            fn eq(&self, other: &Self) -> bool {
                self.data.iter()
                    .zip(&other.data)
                    .all(|(a, b)| a == b)
            }
        }

        impl<const N: usize> std::ops::Index<usize> for $name<N> {
            type Output = $ty;

            fn index(&self, index: usize) -> &Self::Output {
                &self.data[index]
            }
        }

        impl<const N: usize> std::ops::IndexMut<usize> for $name<N> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.data[index]
            }
        }

        impl<const N: usize> std::ops::Mul<Self> for $name<N> {
            type Output = $ty;

            fn mul(self, rhs: Self) -> Self::Output {
                self.data.into_iter()
                    .zip(rhs.data)
                    .map(|(a, b)| a * b)
                    .sum()
            }
        }

        impl<const N: usize> std::ops::Mul<[$ty; N]> for $name<N> {
            type Output = $ty;

            fn mul(self, rhs: [$ty; N]) -> Self::Output {
                self.data.into_iter()
                    .zip(rhs)
                    .map(|(a, b)| a * b)
                    .sum()
            }
        }

        impl<const N: usize> std::ops::Add<Self> for $name<N> {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                let mut output = Self::ZERO;
                for i in 0..N {
                    output[i] = self[i] + rhs[i]
                }
                output
            }
        }

        impl<const N: usize> std::ops::AddAssign<Self> for $name<N> {
            fn add_assign(&mut self, rhs: Self) {
                self.data.iter_mut()
                    .zip(rhs.data)
                    .for_each(|(a, b)| *a += b);
            }
        }

        impl<const N: usize> std::ops::Add<&Self> for $name<N> {
            type Output = Self;

            fn add(self, rhs: &Self) -> Self::Output {
                let mut output = Self::ZERO;
                for i in 0..N {
                    output[i] = self[i] + rhs[i]
                }
                output
            }
        }

        impl<const N: usize> std::ops::AddAssign<&Self> for $name<N> {
            fn add_assign(&mut self, rhs: &Self) {
                self.data.iter_mut()
                    .zip(rhs.data)
                    .for_each(|(a, b)| *a += b);
            }
        }
    };
}

macro_rules! impl_debug {
    ($name:ident => f) => {
        impl<const N: usize> std::fmt::Debug for $name<N> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_list()
                    .entries(&self.data)
                    .finish()
            }
        }
    };
    ($name:ident => u) => {
        impl<const N: usize> std::fmt::Debug for $name<N> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_list()
                    .entries(&self.data)
                    .finish()
            }
        }
    };
}

impl_vec!(VecF32 => f32);
impl_vec!(VecF64 => f64);
impl_vec!(AlignedVecF32 => f32);
impl_vec!(AlignedVecF64 => f64);
impl_vec!(VecU32 => u32);
impl_debug!(VecF32 => f);
impl_debug!(VecF64 => f);
impl_debug!(VecU32 => u);
