//! math primitives for postquantum cryptography

pub mod modular;
pub mod ntt;
pub mod params;

pub use ntt::{BasicNTT, NTT};
pub use params::{F, N, Q, ZETAS};
