//! constants to work with a ring Z_q[X]/(X^n + 1)

/// kyber module (simple number)
/// q was selected for historical reasons
/// q = 1 + k * N
/// k historically = 13
/// N - polynome size
pub const Q: i32 = 3329;

/// polynome size
pub const N: usize = 256;

/// the primitive root of the 512th degree of 1 modulo Q
pub const ROOT: i32 = 17;
