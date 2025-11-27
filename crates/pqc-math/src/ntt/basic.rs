//! Basic implementation of NTT (Kyber-compatible)
//! a simple realization of NTT to compare with other methods.
//! rust's compiler (LLVM) automatically applies basic optimizations

use super::trait_def::NTT;
use crate::modular::{barrett_reduce, mod_add, mod_mul, mod_sub};
use crate::params::{N, Q, ZETAS};

pub struct BasicNTT;

impl BasicNTT {
    pub fn new() -> Self {
        Self
    }
}

impl NTT for BasicNTT {
    fn forward(&self, a: &mut [i32; 256]) {
        let mut k = 1;
        let mut len = 128;

        while len >= 2 {
            for start in (0..N).step_by(2 * len) {
                let zeta = ZETAS[k] as i32;
                k += 1;

                for j in start..start + len {
                    let t = mod_mul(a[j + len], zeta);
                    a[j + len] = mod_sub(a[j], t);
                    a[j] = mod_add(a[j], t);
                }
            }
            len >>= 1;
        }
    }

    fn inverse(&self, a: &mut [i32; 256]) {
        let mut k = 127;
        let mut len = 2;

        while len <= 128 {
            let mut start = 0;
            while start < N {
                let zeta = ZETAS[k] as i32;
                k = k.saturating_sub(1);

                let mut j = start;
                while j < start + len {
                    let t = a[j];
                    a[j] = barrett_reduce(t + a[j + len]);
                    a[j + len] = mod_sub(a[j + len], t);
                    a[j + len] = mod_mul(a[j + len], zeta);
                    j += 1;
                }

                start = j + len;
            }
            len <<= 1;
        }

        let f = 512;
        for x in a.iter_mut() {
            *x = mod_mul(*x, f);
            // normalize by canonical range [0, Q)
            if *x < 0 {
                *x += Q;
            }
        }
    }

    fn name(&self) -> &'static str {
        "BasicNTT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let ntt = BasicNTT::new();
        let mut poly = [0i32; 256];

        for i in 0..8 {
            poly[i] = (i + 1) as i32;
        }

        println!("Before NTT: {:?}", &poly[..8]);

        let mut original = [0i32; 256];
        original.copy_from_slice(&poly);

        ntt.forward(&mut poly);
        println!("After forward NTT: {:?}", &poly[..8]);

        ntt.inverse(&mut poly);
        println!("After inverse NTT: {:?}", &poly[..8]);
        println!("Original: {:?}", &original[..8]);

        assert_eq!(poly, original, "NTT roundtrip failed");
    }

    #[test]
    fn test_roundtrip_full() {
        let ntt = BasicNTT::new();
        let mut poly = [0i32; 256];

        for i in 0..256 {
            poly[i] = (i as i32 * 13) % 3329;
        }

        let original = poly.clone();

        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(poly, original, "NTT roundtrip failed for full array");
    }
}
