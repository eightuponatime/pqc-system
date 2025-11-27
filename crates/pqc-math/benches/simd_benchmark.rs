use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pqc_math::ntt::{BasicNTT, TrueSimdNTT, PolyhedralSimdNTT, NTT};
use pqc_math::modular::{mod_add, mod_sub, mod_mul, barrett_reduce};

#[cfg(target_arch = "x86_64")]
fn bench_simd_operations(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        println!("AVX2 not supported, skipping SIMD benchmarks");
        return;
    }
    
    let ntt = TrueSimdNTT::new();
    
    let mut group = c.benchmark_group("simd_vs_scalar");
    
    let a = [1234, 2345, 3001, 987, 654, 2111, 2999, 1500];
    let b = [111, 222, 333, 444, 555, 666, 777, 888];
    
    // === Модульное сложение ===
    group.bench_function("scalar_mod_add_8x", |bench| {
        bench.iter(|| {
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = mod_add(black_box(a[i]), black_box(b[i]));
            }
            black_box(result)
        });
    });
    
    group.bench_function("simd_mod_add_8x", |bench| {
        bench.iter(|| unsafe {
            black_box(ntt.simd_mod_add_8(black_box(&a), black_box(&b)))
        });
    });
    
    // === Модульное вычитание ===
    group.bench_function("scalar_mod_sub_8x", |bench| {
        bench.iter(|| {
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = mod_sub(black_box(a[i]), black_box(b[i]));
            }
            black_box(result)
        });
    });
    
    group.bench_function("simd_mod_sub_8x", |bench| {
        bench.iter(|| unsafe {
            black_box(ntt.simd_mod_sub_8(black_box(&a), black_box(&b)))
        });
    });
    
    // === Модульное умножение ===
    group.bench_function("scalar_mod_mul_8x", |bench| {
        bench.iter(|| {
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = mod_mul(black_box(a[i]), black_box(b[i]));
            }
            black_box(result)
        });
    });
    
    group.bench_function("simd_mod_mul_8x", |bench| {
        bench.iter(|| unsafe {
            black_box(ntt.simd_mod_mul_8(black_box(&a), black_box(&b)))
        });
    });
    
    // === Barrett редукция ===
    let large_values = [6700, 7000, 8000, 9000, 10000, 5000, 12000, 15000];
    
    group.bench_function("scalar_barrett_8x", |bench| {
        bench.iter(|| {
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = barrett_reduce(black_box(large_values[i]));
            }
            black_box(result)
        });
    });
    
    group.bench_function("simd_barrett_8x", |bench| {
        bench.iter(|| unsafe {
            black_box(ntt.simd_barrett_reduce_8(black_box(&large_values)))
        });
    });
    
    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn bench_ntt_comparison(c: &mut Criterion) {
    let basic_ntt = BasicNTT::new();
    let simd_ntt = TrueSimdNTT::new();
    let polyhedral_ntt = PolyhedralSimdNTT::new();
    
    let mut group = c.benchmark_group("ntt_comparison");
    
    let mut poly = [0i32; 256];
    for i in 0..256 {
        poly[i] = (i as i32 * 17) % 3329;
    }
    
    // === Forward NTT ===
    group.bench_function("BasicNTT_forward", |bench| {
        bench.iter(|| {
            let mut p = poly.clone();
            basic_ntt.forward(black_box(&mut p));
            black_box(p)
        });
    });
    
    group.bench_function("TrueSimdNTT_forward", |bench| {
        bench.iter(|| {
            let mut p = poly.clone();
            simd_ntt.forward(black_box(&mut p));
            black_box(p)
        });
    });
    
    group.bench_function("PolyhedralSimdNTT_forward", |bench| {
        bench.iter(|| {
            let mut p = poly.clone();
            polyhedral_ntt.forward(black_box(&mut p));
            black_box(p)
        });
    });
    
    // === Inverse NTT ===
    let mut poly_transformed = poly.clone();
    basic_ntt.forward(&mut poly_transformed);
    
    group.bench_function("BasicNTT_inverse", |bench| {
        bench.iter(|| {
            let mut p = poly_transformed.clone();
            basic_ntt.inverse(black_box(&mut p));
            black_box(p)
        });
    });
    
    group.bench_function("TrueSimdNTT_inverse", |bench| {
        bench.iter(|| {
            let mut p = poly_transformed.clone();
            simd_ntt.inverse(black_box(&mut p));
            black_box(p)
        });
    });
    
    group.bench_function("PolyhedralSimdNTT_inverse", |bench| {
        bench.iter(|| {
            let mut p = poly_transformed.clone();
            polyhedral_ntt.inverse(black_box(&mut p));
            black_box(p)
        });
    });
    
    // === Full roundtrip ===
    group.bench_function("BasicNTT_roundtrip", |bench| {
        bench.iter(|| {
            let mut p = poly.clone();
            basic_ntt.forward(black_box(&mut p));
            basic_ntt.inverse(black_box(&mut p));
            black_box(p)
        });
    });
    
    group.bench_function("TrueSimdNTT_roundtrip", |bench| {
        bench.iter(|| {
            let mut p = poly.clone();
            simd_ntt.forward(black_box(&mut p));
            simd_ntt.inverse(black_box(&mut p));
            black_box(p)
        });
    });
    
    group.bench_function("PolyhedralSimdNTT_roundtrip", |bench| {
        bench.iter(|| {
            let mut p = poly.clone();
            polyhedral_ntt.forward(black_box(&mut p));
            polyhedral_ntt.inverse(black_box(&mut p));
            black_box(p)
        });
    });
    
    group.finish();
}

#[cfg(target_arch = "x86_64")]
criterion_group!(benches, bench_simd_operations, bench_ntt_comparison);

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches);

criterion_main!(benches);
