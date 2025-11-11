//! CRYSTALS-Kyber: post-quantum KEM
//!
//! kyber algo realizations with a support
//! of different NTT optimizations 

use pqc_math::NTT;

pub mod params;

/// kyber protection level
#[derive(Debug, Clone, Copy)]
pub enum SecurityLevel {
  Kyber512,
  Kyber768,
  Kyber1024
}

pub struct Kyber<N: NTT> {
  ntt_engine: N,
  security_level: SecurityLevel
}

impl<N: NTT> Kyber<N> {
  pub fn new(ntt_engine: N, security_level: SecurityLevel) -> Self {
    Self {
      ntt_engine,
      security_level
    }
  }

  pub fn keygen(&self) -> (PublicKey, SecretKey) {
    // TODO: need to realize
    todo!("keygen not implemented yet");
  }
}

pub struct PublicKey {
  // TODO: add fields
}

pub struct SecretKey {
  // TODO: add fields
}



