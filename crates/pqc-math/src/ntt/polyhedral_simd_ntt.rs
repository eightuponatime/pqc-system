//! Polyhedral-optimized NTT with SIMD vectorization

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::trait_def::NTT;
use crate::modular::{barrett_reduce, mod_add, mod_mul, mod_sub};
use crate::params::{N, Q, ZETAS};

pub struct PolyhedralSimdNTT {
    twiddles: Vec<i32>,
}

impl PolyhedralSimdNTT {
    pub fn new() -> Self {
        let mut twiddles = Vec::with_capacity(128);
        for i in 0..128 {
            twiddles.push(ZETAS[i] as i32);
        }
        Self { twiddles }
    }

    // ======== SIMD OPERATIONS ========
    #[target_feature(enable = "avx2")]
    unsafe fn simd_mod_add_8(&self, a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
        unsafe {
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
    }

    #[target_feature(enable = "avx2")]
    unsafe fn simd_mod_sub_8(&self, a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
        unsafe {
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
    }

    #[target_feature(enable = "avx2")]
    unsafe fn simd_mod_mul_8(&self, a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
        unsafe {
            let mut output = [0i32; 8];
            let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);

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
    }

    #[target_feature(enable = "avx2")]
    unsafe fn simd_barrett_reduce_8(&self, a: &[i32; 8]) -> [i32; 8] {
        unsafe {
            let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
            let v: i64 = ((1i64 << 26) + (Q as i64 / 2)) / Q as i64;
            let vv = _mm256_set1_epi64x(v);

            let va_lower = _mm256_castsi256_si128(va);
            let va_02_64 = _mm256_cvtepi32_epi64(va_lower);
            let prod_02 = _mm256_mul_epi32(vv, va_02_64);
            let t_02_64 = _mm256_srli_epi64(
                _mm256_add_epi64(prod_02, _mm256_set1_epi64x(1i64 << 25)),
                26,
            );

            let va_shifted = _mm256_srli_si256(va, 4);
            let va_shifted_lower = _mm256_castsi256_si128(va_shifted);
            let va_13_64 = _mm256_cvtepi32_epi64(va_shifted_lower);
            let prod_13 = _mm256_mul_epi32(vv, va_13_64);
            let t_13_64 = _mm256_srli_epi64(
                _mm256_add_epi64(prod_13, _mm256_set1_epi64x(1i64 << 25)),
                26,
            );

            let va_upper = _mm256_extracti128_si256(va, 1);
            let va_46_64 = _mm256_cvtepi32_epi64(va_upper);
            let prod_46 = _mm256_mul_epi32(vv, va_46_64);
            let t_46_64 = _mm256_srli_epi64(
                _mm256_add_epi64(prod_46, _mm256_set1_epi64x(1i64 << 25)),
                26,
            );

            let va_upper_shifted = _mm_srli_si128(va_upper, 4);
            let va_57_64 = _mm256_cvtepi32_epi64(va_upper_shifted);
            let prod_57 = _mm256_mul_epi32(vv, va_57_64);
            let t_57_64 = _mm256_srli_epi64(
                _mm256_add_epi64(prod_57, _mm256_set1_epi64x(1i64 << 25)),
                26,
            );

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
    }

    // ======================================================================
    // POLYHEDRAL OPTIMIZATIONS: Loop Tiling + SIMD
    // ======================================================================

    /// Polyhedral-optimized forward NTT with cache blocking
    #[target_feature(enable = "avx2")]
    unsafe fn forward_polyhedral_simd(&self, a: &mut [i32; 256]) {
        unsafe {
            const TILE_SIZE: usize = 64;

            let mut k = 1;
            let mut len = 128;

            while len >= 2 {
                // for big len (>= TILE_SIZE) using simple logic without tiles
                if len >= TILE_SIZE {
                    for start in (0..N).step_by(2 * len) {
                        let zeta = self.twiddles[k];
                        let zeta_vec = [zeta; 8];
                        k += 1;

                        let mut j = start;
                        let end = start + len;

                        // SIMD
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

                            let t = self.simd_mod_mul_8(&a_upper, &zeta_vec);
                            let a_new = self.simd_mod_add_8(&a_lower, &t);
                            let b_new = self.simd_mod_sub_8(&a_lower, &t);

                            for i in 0..8 {
                                a[j + i] = a_new[i];
                                a[j + len + i] = b_new[i];
                            }

                            j += 8;
                        }

                        // residual
                        while j < end {
                            let t = mod_mul(a[j + len], zeta);
                            a[j + len] = mod_sub(a[j], t);
                            a[j] = mod_add(a[j], t);
                            j += 1;
                        }
                    }
                } else {
                    // for little len using polyhedral blocking
                    let mut zeta_idx = k;

                    for tile_base in (0..N).step_by(TILE_SIZE) {
                        let tile_end = (tile_base + TILE_SIZE).min(N);

                        for start in (tile_base..tile_end).step_by(2 * len) {
                            if start + len > N {
                                break;
                            }

                            let zeta = self.twiddles[zeta_idx];
                            let zeta_vec = [zeta; 8];
                            zeta_idx += 1;

                            let mut j = start;
                            let end = (start + len).min(N);

                            // SIMD
                            while j + 8 <= end && j + len + 8 <= N {
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

                                let t = self.simd_mod_mul_8(&a_upper, &zeta_vec);
                                let a_new = self.simd_mod_add_8(&a_lower, &t);
                                let b_new = self.simd_mod_sub_8(&a_lower, &t);

                                for i in 0..8 {
                                    a[j + i] = a_new[i];
                                    a[j + len + i] = b_new[i];
                                }

                                j += 8;
                            }

                            // residual
                            while j < end && j + len < N {
                                let t = mod_mul(a[j + len], zeta);
                                a[j + len] = mod_sub(a[j], t);
                                a[j] = mod_add(a[j], t);
                                j += 1;
                            }
                        }
                    }

                    k = zeta_idx;
                }

                len >>= 1;
            }
        }
    }

    /// Polyhedral-optimized inverse NTT with cache blocking
    #[target_feature(enable = "avx2")]
    unsafe fn inverse_polyhedral_simd(&self, a: &mut [i32; 256]) {
        unsafe {
            const TILE_SIZE: usize = 64;

            let mut k = 127;
            let mut len = 2;

            while len <= 128 {
                // for big len (>= TILE_SIZE) using simple logic without tiles
                if len >= TILE_SIZE {
                    let mut start = 0;
                    while start < N {
                        let zeta = self.twiddles[k];
                        let zeta_vec = [zeta; 8];
                        k = k.saturating_sub(1);

                        let mut j = start;
                        let end = start + len;

                        // SIMD
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

                        start += 2 * len;
                    }
                } else {
                    // for little len using polyhedral blocking
                    let mut zeta_idx = k;

                    for tile_base in (0..N).step_by(TILE_SIZE) {
                        let tile_end = (tile_base + TILE_SIZE).min(N);

                        let mut start = tile_base;
                        while start < tile_end {
                            if start + len > N {
                                break;
                            }

                            let zeta = self.twiddles[zeta_idx];
                            let zeta_vec = [zeta; 8];
                            zeta_idx = zeta_idx.saturating_sub(1);

                            let mut j = start;
                            let end = (start + len).min(N);

                            // SIMD
                            while j + 8 <= end && j + len + 8 <= N {
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
                            while j < end && j + len < N {
                                let t = a[j];
                                a[j] = barrett_reduce(t + a[j + len]);
                                a[j + len] = mod_sub(a[j + len], t);
                                a[j + len] = mod_mul(a[j + len], zeta);
                                j += 1;
                            }

                            start = j + len;
                        }
                    }

                    k = zeta_idx;
                }

                len <<= 1;
            }

            // final normalization (with SIMD)
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

            while i < 256 {
                a[i] = mod_mul(a[i], f);
                if a[i] < 0 {
                    a[i] += Q;
                }
                i += 1;
            }
        }
    }

    // fallback version (scalar)
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

        let f = 512;
        for x in a.iter_mut() {
            *x = mod_mul(*x, f);
            if *x < 0 {
                *x += Q;
            }
        }
    }
}

impl NTT for PolyhedralSimdNTT {
    fn forward(&self, a: &mut [i32; 256]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.forward_polyhedral_simd(a);
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
                    self.inverse_polyhedral_simd(a);
                }
                return;
            }
        }

        self.inverse_scalar(a);
    }

    fn name(&self) -> &'static str {
        "PolyhedralSimdNTT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyhedral_simd_compilation() {
        let _ = PolyhedralSimdNTT::new();
    }

    #[test]
    fn test_polyhedral_simd_roundtrip_small() {
        let ntt = PolyhedralSimdNTT::new();
        let mut poly = [0i32; 256];

        // Небольшой набор данных
        for i in 0..8 {
            poly[i] = (i + 1) as i32;
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(
            poly, original,
            "Polyhedral SIMD NTT roundtrip failed (small)"
        );
    }

    #[test]
    fn test_polyhedral_simd_roundtrip_full() {
        let ntt = PolyhedralSimdNTT::new();
        let mut poly = [0i32; 256];

        // Полный массив
        for i in 0..256 {
            poly[i] = (i as i32 * 13) % 3329;
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(
            poly, original,
            "Polyhedral SIMD NTT roundtrip failed (full)"
        );
    }

    #[test]
    fn test_polyhedral_simd_vs_basic() {
        use crate::ntt::BasicNTT;

        let basic_ntt = BasicNTT::new();
        let polyhedral_ntt = PolyhedralSimdNTT::new();

        let mut poly1 = [0i32; 256];
        let mut poly2 = [0i32; 256];

        // Тестовые данные
        for i in 0..256 {
            let val = (i as i32 * 17) % 3329;
            poly1[i] = val;
            poly2[i] = val;
        }

        // Forward
        basic_ntt.forward(&mut poly1);
        polyhedral_ntt.forward(&mut poly2);
        assert_eq!(poly1, poly2, "Polyhedral forward doesn't match BasicNTT");

        // Inverse
        basic_ntt.inverse(&mut poly1);
        polyhedral_ntt.inverse(&mut poly2);
        assert_eq!(poly1, poly2, "Polyhedral inverse doesn't match BasicNTT");
    }

    #[test]
    fn test_polyhedral_simd_vs_true_simd() {
        use crate::ntt::TrueSimdNTT;

        let true_simd = TrueSimdNTT::new();
        let polyhedral = PolyhedralSimdNTT::new();

        let mut poly1 = [0i32; 256];
        let mut poly2 = [0i32; 256];

        for i in 0..256 {
            let val = (i as i32 * 23) % 3329;
            poly1[i] = val;
            poly2[i] = val;
        }

        // Forward
        true_simd.forward(&mut poly1);
        polyhedral.forward(&mut poly2);
        assert_eq!(poly1, poly2, "Polyhedral forward doesn't match TrueSimdNTT");

        // Inverse
        true_simd.inverse(&mut poly1);
        polyhedral.inverse(&mut poly2);
        assert_eq!(poly1, poly2, "Polyhedral inverse doesn't match TrueSimdNTT");
    }

    #[test]
    fn test_polyhedral_large_values() {
        let ntt = PolyhedralSimdNTT::new();
        let mut poly = [0i32; 256];

        // Большие значения близкие к Q
        for i in 0..256 {
            poly[i] = 3320 + (i as i32 % 9);
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(poly, original, "Polyhedral failed with large values");
    }

    #[test]
    fn test_polyhedral_negative_values() {
        let ntt = PolyhedralSimdNTT::new();
        let mut poly = [0i32; 256];

        // Смешанные положительные и отрицательные
        for i in 0..256 {
            poly[i] = if i % 2 == 0 {
                (i as i32 * 11) % 3329
            } else {
                -((i as i32 * 11) % 3329)
            };
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        // Сравниваем по модулю Q
        for i in 0..256 {
            let left = ((poly[i] % Q) + Q) % Q;
            let right = ((original[i] % Q) + Q) % Q;
            assert_eq!(
                left, right,
                "Mismatch at index {}: {} vs {}",
                i, poly[i], original[i]
            );
        }
    }

    #[test]
    fn test_polyhedral_edge_cases() {
        let ntt = PolyhedralSimdNTT::new();

        // Тест 1: Все нули
        let mut poly_zeros = [0i32; 256];
        let original_zeros = poly_zeros.clone();
        ntt.forward(&mut poly_zeros);
        ntt.inverse(&mut poly_zeros);
        assert_eq!(poly_zeros, original_zeros, "Failed with all zeros");

        // Тест 2: Все единицы
        let mut poly_ones = [1i32; 256];
        let original_ones = poly_ones.clone();
        ntt.forward(&mut poly_ones);
        ntt.inverse(&mut poly_ones);
        assert_eq!(poly_ones, original_ones, "Failed with all ones");

        // Тест 3: Один элемент ненулевой
        let mut poly_single = [0i32; 256];
        poly_single[0] = 1234;
        let original_single = poly_single.clone();
        ntt.forward(&mut poly_single);
        ntt.inverse(&mut poly_single);
        assert_eq!(poly_single, original_single, "Failed with single non-zero");
    }

    #[test]
    fn test_polyhedral_deterministic() {
        let ntt = PolyhedralSimdNTT::new();

        let mut poly1 = [0i32; 256];
        let mut poly2 = [0i32; 256];

        for i in 0..256 {
            let val = (i as i32 * 19) % 3329;
            poly1[i] = val;
            poly2[i] = val;
        }

        // Выполняем дважды - результаты должны быть идентичны
        ntt.forward(&mut poly1);
        ntt.forward(&mut poly2);
        assert_eq!(poly1, poly2, "Forward not deterministic");

        ntt.inverse(&mut poly1);
        ntt.inverse(&mut poly2);
        assert_eq!(poly1, poly2, "Inverse not deterministic");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_polyhedral_simd_operations() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping SIMD tests");
            return;
        }

        let ntt = PolyhedralSimdNTT::new();

        // Тест SIMD операций
        let a = [10, 20, 30, 40, 50, 60, 70, 80];
        let b = [5, 10, 15, 20, 25, 30, 35, 40];

        unsafe {
            // Add
            let result_add = ntt.simd_mod_add_8(&a, &b);
            for i in 0..8 {
                let expected = mod_add(a[i], b[i]);
                assert_eq!(result_add[i], expected, "SIMD add failed at {}", i);
            }

            // Sub
            let result_sub = ntt.simd_mod_sub_8(&a, &b);
            for i in 0..8 {
                let expected = mod_sub(a[i], b[i]);
                assert_eq!(result_sub[i], expected, "SIMD sub failed at {}", i);
            }

            // Mul
            let result_mul = ntt.simd_mod_mul_8(&a, &b);
            for i in 0..8 {
                let expected = mod_mul(a[i], b[i]);
                assert_eq!(result_mul[i], expected, "SIMD mul failed at {}", i);
            }

            // Barrett
            let large = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];
            let result_barrett = ntt.simd_barrett_reduce_8(&large);
            for i in 0..8 {
                let expected = barrett_reduce(large[i]);
                assert_eq!(result_barrett[i], expected, "SIMD barrett failed at {}", i);
            }
        }

        println!("✅ All SIMD operations passed");
    }

    #[test]
    fn test_polyhedral_stress() {
        let ntt = PolyhedralSimdNTT::new();

        // Стресс-тест: много итераций с разными данными
        for iteration in 0..100 {
            let mut poly = [0i32; 256];

            // Генерируем псевдослучайные данные
            for i in 0..256 {
                poly[i] = ((iteration * 7 + i * 13) as i32 * 271) % 3329;
            }

            let original = poly.clone();
            ntt.forward(&mut poly);
            ntt.inverse(&mut poly);

            assert_eq!(
                poly, original,
                "Stress test failed at iteration {}",
                iteration
            );
        }

        println!("✅ Stress test with 100 iterations passed");
    }
}
