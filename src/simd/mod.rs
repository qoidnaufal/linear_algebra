mod neon;

#[cfg(target_feature = "neon")]
pub use neon::*;
