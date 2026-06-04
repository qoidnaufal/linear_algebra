#[macro_export]
macro_rules! mat {
    () => {
        Matrix::ZERO
    };
    ($n:expr) => {
        Matrix::<{ $n * $n }, $n, $n>::ZERO
    };
    ($row:expr, $col:expr) => {
        Matrix::<{ $row * $col }, $row, $col>::ZERO
    };
    ($n:expr => $($val:literal),* $(,)?) => {
        Matrix::<{ $n * $n }, $n, $n> { data: [$($val as f64),*] }
    };
    (($row:expr, $col:expr) => $($val:expr),* $(,)?) => {
        Matrix::<{ $row * $col }, $row, $col> { data: [$($val),*] }
    };
    ($n:expr => $arr:expr) => {
        Matrix::<{ $n * $n }, $n, $n> { data: $arr }
    };
}

#[macro_export]
macro_rules! incidence {
    (($rows:expr, $cols:expr) => $(($($c1:expr)?, $($c2:expr)?)),* $(,)?) => {{
        $($(const _: () = debug_assert!($c1 < $cols);)?)*
        $($(const _: () = debug_assert!($c2 < $cols);)?)*

        let mut m: Matrix<{ $rows * $cols }, $rows, $cols> = Matrix::ZERO;
        let mut i = 0;
        $(
            $(m.data[i * $cols + $c1] = 1.0;)?
            $(m.data[i * $cols + $c2] = -1.0;)?
            i += 1;
        )*
        m
    }};
}

#[macro_export]
macro_rules! t_incidence {
    (($rows:expr, $cols:expr) => $(($($c1:expr)?, $($c2:expr)?)),* $(,)?) => {{
        $($(const _: () = debug_assert!($c1 < $rows);)?)*
        $($(const _: () = debug_assert!($c2 < $rows);)?)*

        let mut m: Matrix<{ $rows * $cols }, $rows, $cols> = Matrix::ZERO;
        let mut i = 0;
        $(
            $(m.data[$c1 * $cols + i] = 1.0;)?
            $(m.data[$c2 * $cols + i] = -1.0;)?
            i += 1;
        )*
        m
    }};
}

#[macro_export]
macro_rules! diagonal {
    (($rows:expr, $cols:expr) => $($val:expr),* $(,)?) => {{
        let mut m: Matrix<{ $rows * $cols }, $rows, $cols> = Matrix::ZERO;
        let mut i = 0usize;
        $(
            m.data[i * $cols + i] = $val;
            i += 1;
        )*
        m
    }};
}

#[macro_export]
macro_rules! create_mat {
    ($name:ident => $row:ident, $col:ident) => {
        type $name = Matrix<{ $row * $col }, $row, $col>;
    };
    ($name:ident => $row:expr, $col:expr) => {
        type $name = Matrix<{ $row * $col }, $row, $col>;
    };
}

#[macro_export]
macro_rules! vecf {
    () => {
        VecF { data: [0.0; _] }
    };
    ([$val:literal; $n:literal]) => {
        VecF { data: [$val as f64; $n] }
    };
    ($n:literal => $($val:literal),* $(,)?) => {
        VecF::<$n> { data: [$($val as f64),*] }
    };
    ($n:literal => $($val:ident),* $(,)?) => {
        VecF::<$n> { data: [$($val),*] }
    };
    ($arr:expr) => {
        VecF { data: $arr }
    };
}

#[macro_export]
macro_rules! vecu {
    () => {
        VecU { data: [0; _] }
    };
    (($n:literal)) => {
        VecU::<$n> { data: [0; $n] }
    };
    ($n:literal => $($val:literal),* $(,)?) => {
        VecU::<$n> { data: [$($val as usize),*] }
    };
}

#[macro_export]
macro_rules! vecb {
    () => {
        VecBool { data: [false; _] }
    };
    ([$val:literal; $n:literal]) => {
        VecBool { data: [$val; $n] }
    };
    ($n:literal => $($val:literal),* $(,)?) => {
        VecBool::<$n> { data: [$($val),*] }
    };
    ($arr:expr) => {
        VecBool { data: $arr }
    };
}

#[macro_export]
macro_rules! create_vec {
    ($name:ident => [f64; $n:literal]) => {
        type $name = VecF<$n>;
    };
    ($name:ident => [usize; $n:literal]) => {
        type $name = VecU<$n>;
    };
}
