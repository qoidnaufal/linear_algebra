#[macro_export]
macro_rules! mat {
    (($n:literal)) => {
        GridMatrix::<$n, { (($n + 3) / 4) * (($n + 3) / 4) }>::ZERO
    };
    (($n:literal) => $($val:literal),* $(,)?) => {
        GridMatrix::<$n, { (($n + 3) / 4) * (($n + 3) / 4) }>::from_flat(&[$($val as f32),*])
    };
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
    () => {
        VecF32 { data: [0f64; _] }
    };
    ($($val:literal),* $(,)?) => {
        VecF32 { data: [$($val as f64),*] }
    };
}

#[macro_export]
macro_rules! vecu32 {
    () => {
        VecU32 { data: [0; _] }
    };
    ($($val:literal),* $(,)?) => {
        VecU32 { data: [$($val as u32),*] }
    };
}
