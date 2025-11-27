//! NTT algo realization using SIMD instructions

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::trait_def::NTT;
use crate::modular::{barrett_reduce, mod_add, mod_mul, mod_sub};
use crate::params::{N, Q, ZETAS};

pub struct TrueSimdNTT {
    twiddles: Vec<i32>,
}

impl TrueSimdNTT {
    pub fn new() -> Self {
        let mut twiddles = Vec::with_capacity(128);
        for i in 0..128 {
            twiddles.push(ZETAS[i] as i32);
        }
        Self { twiddles }
    }

    /// SIMD arithmetic operations
    #[target_feature(enable = "avx2")]
    pub unsafe fn simd_mod_add_8(&self, a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
        let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
        let vsum = _mm256_add_epi32(va, vb);

        let vq = _mm256_set1_epi32(Q);
        let mask = _mm256_cmpgt_epi32(vsum, _mm256_sub_epi32(vq, _mm256_set1_epi32(1)));
        let vq_masked = _mm256_and_si256(mask, vq);
        let result = _mm256_sub_epi32(vsum, vq_masked);

        let mut output = [0i32; 8];
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        output
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn simd_mod_sub_8(&self, a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
        let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
        let vdiff = _mm256_sub_epi32(va, vb);

        let vq = _mm256_set1_epi32(Q);
        let vzero = _mm256_setzero_si256();
        let mask = _mm256_cmpgt_epi32(vzero, vdiff);
        let vq_masked = _mm256_and_si256(mask, vq);
        let result = _mm256_add_epi32(vdiff, vq_masked);

        let mut output = [0i32; 8];
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        output
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn simd_mod_mul_8(&self, a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
        let mut output = [0i32; 8];

        let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);

        // === processing of even indexes [0, 2, 4, 6] ===
        let prod64_even = _mm256_mul_epi32(va, vb);
        let vqinv = _mm256_set1_epi64x(-3327);
        let t_even = _mm256_mul_epi32(prod64_even, vqinv);

        let mask_16bit = _mm256_set1_epi64x(0xFFFF);
        let t_even_masked = _mm256_and_si256(t_even, mask_16bit);

        let sign_bit = _mm256_and_si256(t_even_masked, _mm256_set1_epi64x(0x8000));
        let sign_extend = _mm256_cmpeq_epi64(sign_bit, _mm256_set1_epi64x(0x8000));
        let sign_fill = _mm256_and_si256(
            sign_extend,
            _mm256_set1_epi64x(0xFFFFFFFFFFFF0000u64 as i64),
        );
        let t_even_signed = _mm256_or_si256(t_even_masked, sign_fill);

        let vq64 = _mm256_set1_epi64x(Q as i64);
        let mult_result = _mm256_mul_epi32(t_even_signed, vq64);
        let sub_result = _mm256_sub_epi64(prod64_even, mult_result);
        let result_even_64 = _mm256_srli_epi64(sub_result, 16);

        // === processing of odd indexes [1, 3, 5, 7] ===
        let va_odd = _mm256_srli_si256(va, 4);
        let vb_odd = _mm256_srli_si256(vb, 4);
        let prod64_odd = _mm256_mul_epi32(va_odd, vb_odd);

        let t_odd = _mm256_mul_epi32(prod64_odd, vqinv);
        let t_odd_masked = _mm256_and_si256(t_odd, mask_16bit);

        let sign_bit_odd = _mm256_and_si256(t_odd_masked, _mm256_set1_epi64x(0x8000));
        let sign_extend_odd = _mm256_cmpeq_epi64(sign_bit_odd, _mm256_set1_epi64x(0x8000));
        let sign_fill_odd = _mm256_and_si256(
            sign_extend_odd,
            _mm256_set1_epi64x(0xFFFFFFFFFFFF0000u64 as i64),
        );
        let t_odd_signed = _mm256_or_si256(t_odd_masked, sign_fill_odd);

        let mult_result_odd = _mm256_mul_epi32(t_odd_signed, vq64);
        let sub_result_odd = _mm256_sub_epi64(prod64_odd, mult_result_odd);
        let result_odd_64 = _mm256_srli_epi64(sub_result_odd, 16);

        // === packaging of results ===
        let mut temp_even = [0i64; 4];
        let mut temp_odd = [0i64; 4];
        _mm256_storeu_si256(temp_even.as_mut_ptr() as *mut __m256i, result_even_64);
        _mm256_storeu_si256(temp_odd.as_mut_ptr() as *mut __m256i, result_odd_64);

        output[0] = temp_even[0] as i32;
        output[1] = temp_odd[0] as i32;
        output[2] = temp_even[1] as i32;
        output[3] = temp_odd[1] as i32;
        output[4] = temp_even[2] as i32;
        output[5] = temp_odd[2] as i32;
        output[6] = temp_even[3] as i32;
        output[7] = temp_odd[3] as i32;

        output
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn simd_barrett_reduce_8(&self, a: &[i32; 8]) -> [i32; 8] {
        let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);

        let v: i64 = ((1i64 << 26) + (Q as i64 / 2)) / Q as i64;
        let vv = _mm256_set1_epi64x(v);

        // === indexes [0, 2] ===
        let va_lower = _mm256_castsi256_si128(va);
        let va_02_64 = _mm256_cvtepi32_epi64(va_lower);
        let prod_02 = _mm256_mul_epi32(vv, va_02_64);
        let t_02_64 = _mm256_srli_epi64(
            _mm256_add_epi64(prod_02, _mm256_set1_epi64x(1i64 << 25)),
            26,
        );

        // === indexes [1, 3] ===
        let va_shifted = _mm256_srli_si256(va, 4);
        let va_shifted_lower = _mm256_castsi256_si128(va_shifted);
        let va_13_64 = _mm256_cvtepi32_epi64(va_shifted_lower);
        let prod_13 = _mm256_mul_epi32(vv, va_13_64);
        let t_13_64 = _mm256_srli_epi64(
            _mm256_add_epi64(prod_13, _mm256_set1_epi64x(1i64 << 25)),
            26,
        );

        // === indexes [4, 6] ===
        let va_upper = _mm256_extracti128_si256(va, 1);
        let va_46_64 = _mm256_cvtepi32_epi64(va_upper);
        let prod_46 = _mm256_mul_epi32(vv, va_46_64);
        let t_46_64 = _mm256_srli_epi64(
            _mm256_add_epi64(prod_46, _mm256_set1_epi64x(1i64 << 25)),
            26,
        );

        // === indexes [5, 7] ===
        let va_upper_shifted = _mm_srli_si128(va_upper, 4);
        let va_57_64 = _mm256_cvtepi32_epi64(va_upper_shifted);
        let prod_57 = _mm256_mul_epi32(vv, va_57_64);
        let t_57_64 = _mm256_srli_epi64(
            _mm256_add_epi64(prod_57, _mm256_set1_epi64x(1i64 << 25)),
            26,
        );

        // collecting the results
        let mut t_02 = [0i64; 4];
        let mut t_13 = [0i64; 4];
        let mut t_46 = [0i64; 4];
        let mut t_57 = [0i64; 4];
        _mm256_storeu_si256(t_02.as_mut_ptr() as *mut __m256i, t_02_64);
        _mm256_storeu_si256(t_13.as_mut_ptr() as *mut __m256i, t_13_64);
        _mm256_storeu_si256(t_46.as_mut_ptr() as *mut __m256i, t_46_64);
        _mm256_storeu_si256(t_57.as_mut_ptr() as *mut __m256i, t_57_64);

        let mut t_values = [0i32; 8];
        t_values[0] = t_02[0] as i32;
        t_values[1] = t_13[0] as i32;
        t_values[2] = t_02[2] as i32;
        t_values[3] = t_13[2] as i32;
        t_values[4] = t_46[0] as i32;
        t_values[5] = t_57[0] as i32;
        t_values[6] = t_46[2] as i32;
        t_values[7] = t_57[2] as i32;

        let mut output = [0i32; 8];
        for i in 0..8 {
            output[i] = a[i] - t_values[i] * Q;
        }
        output
    }

    // ==============================================================
    // SIMD-OPTIMIZED NTT
    // ==============================================================

    /// SIMD-optimized forward NTT (checking AVX2 arch OUTSIDE)
    #[target_feature(enable = "avx2")]
    unsafe fn forward_simd(&self, a: &mut [i32; 256]) {
        let mut k = 1;
        let mut len = 128;

        while len >= 2 {
            for start in (0..N).step_by(2 * len) {
                let zeta = self.twiddles[k];
                k += 1;

                let mut j = start;
                let end = start + len;

                // SIMD: processing 8 elements per time
                while j + 8 <= end {
                    let a_lower = [
                        a[j],
                        a[j + 1],
                        a[j + 2],
                        a[j + 3],
                        a[j + 4],
                        a[j + 5],
                        a[j + 6],
                        a[j + 7],
                    ];
                    let a_upper = [
                        a[j + len],
                        a[j + len + 1],
                        a[j + len + 2],
                        a[j + len + 3],
                        a[j + len + 4],
                        a[j + len + 5],
                        a[j + len + 6],
                        a[j + len + 7],
                    ];
                    let zeta_vec = [zeta; 8];

                    let t = self.simd_mod_mul_8(&a_upper, &zeta_vec);
                    let a_new = self.simd_mod_add_8(&a_lower, &t);
                    let b_new = self.simd_mod_sub_8(&a_lower, &t);

                    for i in 0..8 {
                        a[j + i] = a_new[i];
                        a[j + len + i] = b_new[i];
                    }

                    j += 8;
                }

                // Остаток
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

    /// SIMD-optimized inverse NTT (also checking AVX2 arch outside)
    #[target_feature(enable = "avx2")]
    unsafe fn inverse_simd(&self, a: &mut [i32; 256]) {
        let mut k = 127;
        let mut len = 2;

        while len <= 128 {
            let mut start = 0;

            while start < N {
                let zeta = self.twiddles[k];
                k = k.saturating_sub(1);

                let mut j = start;
                let end = start + len;

                // SIMD: processing 8 elements per time
                while j + 8 <= end {
                    let a_lower = [
                        a[j],
                        a[j + 1],
                        a[j + 2],
                        a[j + 3],
                        a[j + 4],
                        a[j + 5],
                        a[j + 6],
                        a[j + 7],
                    ];
                    let a_upper = [
                        a[j + len],
                        a[j + len + 1],
                        a[j + len + 2],
                        a[j + len + 3],
                        a[j + len + 4],
                        a[j + len + 5],
                        a[j + len + 6],
                        a[j + len + 7],
                    ];
                    let zeta_vec = [zeta; 8];

                    let sum = self.simd_mod_add_8(&a_lower, &a_upper);
                    let a_new = self.simd_barrett_reduce_8(&sum);

                    let diff = self.simd_mod_sub_8(&a_upper, &a_lower);
                    let b_new = self.simd_mod_mul_8(&diff, &zeta_vec);

                    for i in 0..8 {
                        a[j + i] = a_new[i];
                        a[j + len + i] = b_new[i];
                    }

                    j += 8;
                }

                // residual
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
        let f_vec = [f; 8];
        let mut i = 0;

        while i + 8 <= 256 {
            let chunk = [
                a[i],
                a[i + 1],
                a[i + 2],
                a[i + 3],
                a[i + 4],
                a[i + 5],
                a[i + 6],
                a[i + 7],
            ];
            let result = self.simd_mod_mul_8(&chunk, &f_vec);

            for j in 0..8 {
                a[i + j] = result[j];
                if a[i + j] < 0 {
                    a[i + j] += Q;
                }
            }
            i += 8;
        }

        // Residual
        while i < 256 {
            a[i] = mod_mul(a[i], f);
            if a[i] < 0 {
                a[i] += Q;
            }
            i += 1;
        }
    }

    /// Fallback: scalar version of forward
    fn forward_scalar(&self, a: &mut [i32; 256]) {
        let mut k = 1;
        let mut len = 128;

        while len >= 2 {
            for start in (0..N).step_by(2 * len) {
                let zeta = self.twiddles[k];
                k += 1;

                for j in start..(start + len) {
                    let t = mod_mul(a[j + len], zeta);
                    a[j + len] = mod_sub(a[j], t);
                    a[j] = mod_add(a[j], t);
                }
            }
            len >>= 1;
        }
    }

    /// Fallback: scalar version of inverse
    fn inverse_scalar(&self, a: &mut [i32; 256]) {
        let mut k = 127;
        let mut len = 2;

        while len <= 128 {
            let mut start = 0;
            while start < N {
                let zeta = self.twiddles[k];
                k = k.saturating_sub(1);

                for j in start..(start + len) {
                    let t = a[j];
                    a[j] = barrett_reduce(t + a[j + len]);
                    a[j + len] = mod_sub(a[j + len], t);
                    a[j + len] = mod_mul(a[j + len], zeta);
                }

                start += 2 * len;
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

impl NTT for TrueSimdNTT {
    fn forward(&self, a: &mut [i32; 256]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.forward_simd(a);
                }
                return;
            }
        }

        self.forward_scalar(a);
    }

    fn inverse(&self, a: &mut [i32; 256]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.inverse_simd(a);
                }
                return;
            }
        }

        self.inverse_scalar(a);
    }

    fn name(&self) -> &'static str {
        "TrueSimdNTT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation() {
        let _ = TrueSimdNTT::new();
    }

    #[test]
    fn test_polyhedral_roundtrip() {
        let ntt = TrueSimdNTT::new();
        let mut poly = [0i32; 256];

        for i in 0..8 {
            poly[i] = (i + 1) as i32;
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(poly, original);
    }

    #[test]
    fn test_polyhedral_full() {
        let ntt = TrueSimdNTT::new();
        let mut poly = [0i32; 256];

        for i in 0..256 {
            poly[i] = (i as i32 * 13) % 3329;
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(poly, original);
    }

    #[test]
    fn compare_with_basic() {
        use crate::ntt::BasicNTT;

        let basic_ntt = BasicNTT::new();
        let simd_ntt = TrueSimdNTT::new();

        let mut poly1 = [0i32; 256];
        let mut poly2 = [0i32; 256];

        for i in 0..256 {
            let val = (i as i32 * 17) % 3329;
            poly1[i] = val;
            poly2[i] = val;
        }

        basic_ntt.forward(&mut poly1);
        simd_ntt.forward(&mut poly2);
        assert_eq!(poly1, poly2);

        basic_ntt.inverse(&mut poly1);
        simd_ntt.inverse(&mut poly2);
        assert_eq!(poly1, poly2);
    }
}
