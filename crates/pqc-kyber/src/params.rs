//! params for different security levels

/// kyber-512
pub mod kyber512 {
  pub const K: usize = 2;
  pub const ETA1: usize = 3;
  pub const ETA2: usize = 2;
  pub const DU: usize = 10;
  pub const DV: usize = 4;
}

/// kyber-768
pub mod kyber768 {
  pub const K: usize = 3;
  pub const ETA1: usize = 2;
  pub const ETA2: usize = 2;
  pub const DU: usize = 10;
  pub const DV: usize = 4;
}

/// kyber-1024
pub mod kyber1024 {
  pub const K: usize = 4;
  pub const ETA1: usize = 2;
  pub const ETA2: usize = 2;
  pub const DU: usize = 11;
  pub const DV: usize = 5;
}
