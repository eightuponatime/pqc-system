//! trait for different realizations of NTT

/// trait for ntt transformations
pub trait NTT {
  /// ntt forward transformation
  fn forward(&self, a: &mut [i32; 256]);

  /// ntt inverse transformation
  fn inverse(&self, a: &mut [i32; 256]);

  /// realization name
  fn name(&self) -> &'static str;
} 