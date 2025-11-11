//! number theoretic transform (ntt) realization

mod trait_def;
mod basic;

pub use trait_def::NTT;
pub use basic::BasicNTT;

// TODO: add polyhedral realization
// pub use polyhedral::PolyhedralNTT;