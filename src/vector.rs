macro_rules! impl_vec {
    ($name:ident => $ty:ty $(=> $align:literal)?) => {
        $(#[repr(align($align))])?
        #[derive(Clone, Copy)]
        pub struct $name<const N: usize> {
            pub data: [$ty; N]
        }

        impl<const N: usize> Default for $name<N> {
            fn default() -> Self {
                Self::ZERO
            }
        }

        impl<const N: usize> $name<N> {
            pub const ZERO: Self = Self { data: unsafe { core::mem::zeroed::<[$ty; N]>() } };

            pub const SEQUENTIAL: Self = {
                let mut data = unsafe { core::mem::zeroed::<[$ty; N]>() };
                let mut i = 1;
                while i < N {
                    data[i] = i as $ty;
                    i += 1;
                }
                Self { data }
            };

            pub fn sequential() -> Self {
                Self {
                    data: core::array::from_fn::<$ty, N, _>(|i| i as $ty)
                }
            }

            pub const fn swap(&mut self, a: usize, b: usize) {
                self.data.swap(a, b)
            }

            pub fn map<R>(&self, f: impl FnMut($ty) -> R) -> [R; N] {
                self.data.map(f)
            }

            pub fn iter(&self) -> Iter<'_, $ty, N> {
                Iter {
                    data: self.data.as_ptr(),
                    count: 0,
                    marker: core::marker::PhantomData,
                }
            }

            pub fn iter_mut(&mut self) -> IterMut<'_, $ty, N> {
                IterMut {
                    data: self.data.as_mut_ptr(),
                    count: 0,
                    marker: core::marker::PhantomData,
                }
            }

            pub fn chunks_exact(&self, size: usize) -> core::slice::ChunksExact<'_, $ty> {
                self.data.chunks_exact(size)
            }

            pub fn chunks_exact_mut(&mut self, size: usize) -> core::slice::ChunksExactMut<'_, $ty> {
                self.data.chunks_exact_mut(size)
            }

            pub fn scalar_mul(&self, rhs: $ty) -> Self {
                let mut out = Self::ZERO;
                out.iter_mut()
                    .zip(&self.data)
                    .for_each(|(val, src)| *val = src * rhs);
                out
            }
        }

        impl<const N: usize> PartialEq for $name<N> {
            fn eq(&self, other: &Self) -> bool {
                self.iter().zip(other.iter())
                    .all(|(a, b)| a == b)
            }
        }

        // =============================================================================
        // Arithmetic
        // =============================================================================

        impl<const N: usize> core::ops::Mul<Self> for $name<N> {
            type Output = $ty;

            fn mul(self, rhs: Self) -> Self::Output {
                self.iter().zip(&rhs).map(|(a, b)| a * b).sum()
            }
        }

        impl<'a, const N: usize> core::ops::Mul<$name<N>> for &'a $name<N> {
            type Output = $ty;

            fn mul(self, rhs: $name<N>) -> Self::Output {
                self.iter().zip(&rhs).map(|(a, b)| a * b).sum()
            }
        }

        impl<'a, const N: usize> core::ops::Mul<Self> for &'a $name<N> {
            type Output = $ty;

            fn mul(self, rhs: Self) -> Self::Output {
                self.iter().zip(rhs).map(|(a, b)| a * b).sum()
            }
        }

        impl<const N: usize> core::ops::Mul<[$ty; N]> for $name<N> {
            type Output = $ty;

            fn mul(self, rhs: [$ty; N]) -> Self::Output {
                self.iter().zip(&rhs).map(|(a, b)| a * b).sum()
            }
        }

        impl<const N: usize> core::ops::Add<Self> for $name<N> {
            type Output = Self;

            fn add(mut self, rhs: Self) -> Self::Output {
                self.iter_mut().zip(&rhs)
                    .for_each(|(a, b)| *a += b);
                self
            }
        }

        impl<const N: usize> core::ops::AddAssign<Self> for $name<N> {
            fn add_assign(&mut self, rhs: Self) {
                self.iter_mut().zip(&rhs)
                    .for_each(|(a, b)| *a += b);
            }
        }

        impl<const N: usize> core::ops::Add<&Self> for $name<N> {
            type Output = Self;

            fn add(mut self, rhs: &Self) -> Self::Output {
                self.iter_mut().zip(rhs)
                    .for_each(|(a, b)| *a += b);
                self
            }
        }

        impl<const N: usize> core::ops::AddAssign<&Self> for $name<N> {
            fn add_assign(&mut self, rhs: &Self) {
                self.iter_mut().zip(rhs)
                    .for_each(|(a, b)| *a += b);
            }
        }

        impl<const N: usize> core::ops::Sub<Self> for $name<N> {
            type Output = Self;

            fn sub(mut self, rhs: Self) -> Self::Output {
                self.iter_mut().zip(&rhs)
                    .for_each(|(a, b)| *a -= b);
                self
            }
        }

        impl<const N: usize> core::ops::Sub<$name<N>> for &$name<N> {
            type Output = $name<N>;

            fn sub(self, rhs: $name<N>) -> Self::Output {
                let mut result = $name::ZERO;
                for i in 0..N {
                    result[i] = self[i] - rhs[i]
                }
                result
            }
        }

        // =============================================================================
        // IntoIterator
        // =============================================================================

        impl<'a, const N: usize> IntoIterator for &'a $name<N> {
            type Item = &'a $ty;
            type IntoIter = Iter<'a, $ty, N>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl<'a, const N: usize> IntoIterator for &'a mut $name<N> {
            type Item = &'a mut $ty;
            type IntoIter = IterMut<'a, $ty, N>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter_mut()
            }
        }

        impl<const N: usize> FromIterator<$ty> for $name<N> {
            fn from_iter<T: IntoIterator<Item = $ty>>(iter: T) -> Self {
                let mut this = Self::ZERO;
                this.iter_mut().zip(iter)
                    .for_each(|(a, b)| *a = b);
                this
            }
        }

        impl<const N: usize> core::iter::Sum for $name<N> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |acc, v2| acc + v2)
            }
        }

        // =============================================================================
        // Indexing
        // =============================================================================

        impl<const N: usize> core::ops::Index<usize> for $name<N> {
            type Output = $ty;

            fn index(&self, index: usize) -> &Self::Output {
                unsafe { self.data.get_unchecked(index) }
            }
        }

        impl<const N: usize> core::ops::IndexMut<usize> for $name<N> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                unsafe { self.data.get_unchecked_mut(index) }
            }
        }

        impl<const N: usize> core::ops::Index<core::ops::Range<usize>> for $name<N> {
            type Output = [$ty];

            fn index(&self, index: core::ops::Range<usize>) -> &Self::Output {
                unsafe { self.data.get_unchecked(index) }
            }
        }

        impl<const N: usize> core::ops::IndexMut<core::ops::Range<usize>> for $name<N> {
            fn index_mut(&mut self, index: core::ops::Range<usize>) -> &mut Self::Output {
                unsafe { self.data.get_unchecked_mut(index) }
            }
        }

        // =============================================================================
        // Debug
        // =============================================================================

        impl<const N: usize> core::fmt::Debug for $name<N> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "[")?;
                self.data.iter().try_for_each(|v| write!(f, " {:.2e} ", v))?;
                write!(f, "]")
            }
        }
    };
}

impl_vec!(VecF => f64 => 32);
impl_vec!(VecU => usize => 32);

// =============================================================================
// Iter
// =============================================================================

pub struct Iter<'a, T, const N: usize> {
    data: *const T,
    count: usize,
    marker: core::marker::PhantomData<&'a T>,
}

impl<'a, T, const N: usize> Iterator for Iter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == N { return None }
        unsafe {
            let out = self.data.add(self.count);
            self.count += 1;
            Some(&*out)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (N - self.count, Some(N - self.count))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for Iter<'a, T, N> {
    fn len(&self) -> usize {
        N - self.count
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for Iter<'a, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.count == N { return None }
        unsafe {
            let idx = N - self.count;
            self.count += 1;
            Some(&*self.data.add(idx))
        }
    }
}

// =============================================================================
// IterMut
// =============================================================================

pub struct IterMut<'a, T, const N: usize> {
    data: *mut T,
    count: usize,
    marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T, const N: usize> Iterator for IterMut<'a, T, N> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == N { return None }
        unsafe {
            let out = self.data.add(self.count);
            self.count += 1;
            Some(&mut *out)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (N - self.count, Some(N - self.count))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for IterMut<'a, T, N> {
    fn len(&self) -> usize {
        N - self.count
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for IterMut<'a, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.count == N { return None }
        unsafe {
            let idx = N - self.count;
            self.count += 1;
            Some(&mut *self.data.add(idx))
        }
    }
}

pub struct VecBool<const N: usize> {
    pub data: [bool; N]
}

impl<const N: usize> core::ops::Index<usize> for VecBool<N> {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe {
            self.data.get_unchecked(index)
        }
    }
}

impl<const N: usize> core::ops::IndexMut<usize> for VecBool<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            self.data.get_unchecked_mut(index)
        }
    }
}
