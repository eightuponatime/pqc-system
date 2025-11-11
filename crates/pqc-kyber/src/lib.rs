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
    Kyber1024,
}

pub struct Kyber<N: NTT> {
    ntt_engine: N,
    security_level: SecurityLevel,
}

impl<N: NTT> Kyber<N> {
    /// creating new Kyber instance
    pub fn new(ntt_engine: N, security_level: SecurityLevel) -> Self {
        Self {
            ntt_engine,
            security_level,
        }
    }

    /// generate a key pair
    pub fn keygen(&self) -> (PublicKey, SecretKey) {
        // TODO: need to realize
        todo!("keygen not implemented yet");
    }

    /// encapsulation - (creating the general secret)
    pub fn encapsulate(&self, _pk: &PublicKey) -> (Vec<u8>, Vec<u8>) {
        todo!("Encapsulation not yet implemented")
    }

    /// revert the general secret
    pub fn decapsulate(&self, _ct: &u8, _sk: &SecretKey) -> Vec<u8> {
        todo!("Decapsulation not yet implemented")
    }
}

#[derive(Debug, Clone)]
pub struct PublicKey {
    // TODO: add fields (polyvec, seed)
}

#[derive(Debug, Clone)]
pub struct SecretKey {
    // TODO: add fields (polyvec)
}
