#[macro_export]
macro_rules! mat {
    (($n:literal)) => {{
        const __GRID__: usize = (($n + 3) / 4) * (($n + 3) / 4);
        Matrix::<$n, __GRID__>::ZERO
    }};
    (($n:literal) => $($val:literal),* $(,)?) => {{
        const __GRID__: usize = (($n + 3) / 4) * (($n + 3) / 4);
        Matrix::<$n, __GRID__>::from_flat(&[$($val as f64),*])
    }};
}

#[macro_export]
macro_rules! vecf32 {
    () => {
        VecF32 { data: [0f32; _] }
    };
    ($($val:literal),* $(,)?) => {
        VecF32 { data: [$($val as f32),*] }
    };
}

#[macro_export]
macro_rules! vecf64 {
    (($n:literal, $align:literal)) => {
        VecF64 { data: [0f64; _] }
    };
    (($n:literal, $align:literal) => $($val:literal),* $(,)?) => {{
        const __PAD__: usize = ($n + ($align - 1)) & !($align - 1);
        let arr = &[$($val as f64),*];
        let len = arr.len();
        let mut data = [0f64; __PAD__];
        for i in 0..len {
            data[i] = arr[i];
        }
        VecF64::<$n, __PAD__> { data }
    }};
}
