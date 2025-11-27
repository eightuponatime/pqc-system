//! Manual loop unrolling implementation of NTT
//!
//! optimizations:
//! - manual loop unrolling (4x)
//! - inline butterfly operations
//! - reduced function call overhead
//!
//! BasicNTT ~5.57 micro sec.
//! UnrolledNTT ~5.72 micro sec. -> regression ~2.7%
//!
//! current optimizations slowed the program

use super::trait_def::NTT;
use crate::modular::{barrett_reduce, mod_add, mod_mul, mod_sub};
use crate::params::{N, Q, ZETAS};

pub struct UnrolledNTT {
    twiddles: Vec<i32>,
}

impl UnrolledNTT {
    pub fn new() -> Self {
        let mut twiddles = Vec::with_capacity(128);
        for i in 0..128 {
            twiddles.push(ZETAS[i] as i32);
        }
        Self { twiddles }
    }

    /// Optimization: Inlined butterfly with manual unrolling
    fn forward_polyhedral(&self, a: &mut [i32; 256]) {
        let mut k = 1;
        let mut len = 128;

        while len >= 2 {
            for start in (0..N).step_by(2 * len) {
                let zeta = self.twiddles[k];
                k += 1;

                let mut j = start;
                let end = start + len;

                // Manual unrolling - 4 elements per iteration
                while j + 4 <= end {
                    // Butterfly 1 (inline)
                    let t0 = mod_mul(a[j + len], zeta);
                    a[j + len] = mod_sub(a[j], t0);
                    a[j] = mod_add(a[j], t0);

                    // Butterfly 2 (inline)
                    let t1 = mod_mul(a[j + 1 + len], zeta);
                    a[j + 1 + len] = mod_sub(a[j + 1], t1);
                    a[j + 1] = mod_add(a[j + 1], t1);

                    // Butterfly 3 (inline)
                    let t2 = mod_mul(a[j + 2 + len], zeta);
                    a[j + 2 + len] = mod_sub(a[j + 2], t2);
                    a[j + 2] = mod_add(a[j + 2], t2);

                    // Butterfly 4 (inline)
                    let t3 = mod_mul(a[j + 3 + len], zeta);
                    a[j + 3 + len] = mod_sub(a[j + 3], t3);
                    a[j + 3] = mod_add(a[j + 3], t3);

                    j += 4;
                }

                // Остаточные элементы
                while j < end {
                    let t = mod_mul(a[j + len], zeta);
                    a[j + len] = mod_sub(a[j], t);
                    a[j] = mod_add(a[j], t);
                    j += 1;
                }
            }
            len >>= 1;
        }
    }

    /// Polyhedral-optimized inverse NTT
    fn inverse_polyhedral(&self, a: &mut [i32; 256]) {
        let mut k = 127;
        let mut len = 2;

        while len <= 128 {
            let mut start = 0;

            while start < N {
                let zeta = self.twiddles[k];
                k = k.saturating_sub(1);

                let mut j = start;
                let end = start + len;

                // Manual unrolling for inverse
                while j + 4 <= end {
                    // Operation 1
                    let t0 = a[j];
                    a[j] = barrett_reduce(t0 + a[j + len]);
                    a[j + len] = mod_mul(mod_sub(a[j + len], t0), zeta);

                    // Operation 2
                    let t1 = a[j + 1];
                    a[j + 1] = barrett_reduce(t1 + a[j + 1 + len]);
                    a[j + 1 + len] = mod_mul(mod_sub(a[j + 1 + len], t1), zeta);

                    // Operation 3
                    let t2 = a[j + 2];
                    a[j + 2] = barrett_reduce(t2 + a[j + 2 + len]);
                    a[j + 2 + len] = mod_mul(mod_sub(a[j + 2 + len], t2), zeta);

                    // Operation 4
                    let t3 = a[j + 3];
                    a[j + 3] = barrett_reduce(t3 + a[j + 3 + len]);
                    a[j + 3 + len] = mod_mul(mod_sub(a[j + 3 + len], t3), zeta);

                    j += 4;
                }

                // residual elements
                while j < end {
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

        // Final normalization
        let f = 512;
        for x in a.iter_mut() {
            *x = mod_mul(*x, f);
            if *x < 0 {
                *x += Q;
            }
        }
    }
}

impl NTT for UnrolledNTT {
    fn forward(&self, a: &mut [i32; 256]) {
        self.forward_polyhedral(a);
    }

    fn inverse(&self, a: &mut [i32; 256]) {
        self.inverse_polyhedral(a);
    }

    fn name(&self) -> &'static str {
        "UnrolledNTT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation() {
        let _ = UnrolledNTT::new();
    }

    #[test]
    fn test_polyhedral_roundtrip() {
        let ntt = UnrolledNTT::new();
        let mut poly = [0i32; 256];

        for i in 0..8 {
            poly[i] = (i + 1) as i32;
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(poly, original, "Unrolled NTT roundtrip failed");
    }

    #[test]
    fn test_polyhedral_full() {
        let ntt = UnrolledNTT::new();
        let mut poly = [0i32; 256];

        for i in 0..256 {
            poly[i] = (i as i32 * 13) % 3329;
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(poly, original, "Unrolled NTT full roundtrip failed");
    }

    #[test]
    fn compare_with_basic() {
        use crate::ntt::BasicNTT;

        let basic_ntt = BasicNTT::new();
        let polyhedral_ntt = UnrolledNTT::new();

        let mut poly1 = [0i32; 256];
        let mut poly2 = [0i32; 256];

        for i in 0..256 {
            let val = (i as i32 * 17) % 3329;
            poly1[i] = val;
            poly2[i] = val;
        }

        basic_ntt.forward(&mut poly1);
        polyhedral_ntt.forward(&mut poly2);

        assert_eq!(poly1, poly2, "forward transforms don't match");

        basic_ntt.inverse(&mut poly1);
        polyhedral_ntt.inverse(&mut poly2);

        assert_eq!(poly1, poly2, "inverse transforms don't match");
    }
}
