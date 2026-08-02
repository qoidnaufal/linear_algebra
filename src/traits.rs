pub trait Container<const ROW: usize, const COL: usize>
where Self:
    core::ops::Index<usize, Output = f64>
    + core::ops::IndexMut<usize, Output = f64>
{
    fn ptr(&self, offset: usize) -> *const f64;
    fn ptr_mut(&mut self, offset: usize) -> *mut f64;
    #[inline]
    fn copy_from_container<T: Container<ROW, COL>>(&mut self, src: &T) {
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.ptr(0),
                self.ptr_mut(0),
                ROW * COL,
            );
        }
    }
}
