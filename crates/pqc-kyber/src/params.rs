//! params for different security levels

/// kyber-512
pub mod kyber512 {
    /// module size (matrix k x k)
    pub const K: usize = 2;
    /// distribution parameter for the secret key
    pub const ETA1: usize = 3;
    /// distribution parameter for the error
    pub const ETA2: usize = 2;
    /// bits for U compression
    pub const DU: usize = 10;
    /// bits for V compression
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
