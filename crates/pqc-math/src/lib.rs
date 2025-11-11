pub mod modular;
pub mod ntt;
pub mod params;

pub use ntt::{BasicNTT, NTT};
pub use params::{N, Q, ROOT};
