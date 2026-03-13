pub mod matrix;
pub mod vector;
pub mod macros;

pub const fn padding(n: usize, align: usize) -> usize {
    (n + (align - 1)) & !(align- 1)
}

pub const fn num_tiles(n: usize) -> usize {
    (n + 3) / 4
}

pub const fn num_blocks(n: usize) -> usize {
    let t = num_tiles(n);
    t * t
}
