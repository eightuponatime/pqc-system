//! Benchmarks для сравнения всех NTT реализаций
//!
//! Запуск: cargo bench
//! Результаты сохраняются в target/criterion/

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pqc_math::{BasicNTT, SimdNTT, UnrolledNTT, NTT};

/// Генерация тестовых данных
fn generate_test_poly(seed: i32) -> [i32; 256] {
    let mut poly = [0i32; 256];
    for i in 0..256 {
        poly[i] = ((i as i32 * seed) % 3329 + 3329) % 3329;
    }
    poly
}

/// Бенчмарк forward NTT для всех реализаций
fn bench_forward_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT Forward");

    // Тестируем на разных данных
    for seed in [13, 17, 23].iter() {
        let test_data = generate_test_poly(*seed);

        // BasicNTT forward
        group.bench_with_input(BenchmarkId::new("BasicNTT", seed), seed, |b, _| {
            let ntt = BasicNTT::new();
            b.iter(|| {
                let mut poly = test_data.clone();
                ntt.forward(black_box(&mut poly));
            });
        });

        // UnrolledNTT forward
        group.bench_with_input(BenchmarkId::new("UnrolledNTT", seed), seed, |b, _| {
            let ntt = UnrolledNTT::new();
            b.iter(|| {
                let mut poly = test_data.clone();
                ntt.forward(black_box(&mut poly));
            });
        });

        // SimdNTT forward
        group.bench_with_input(BenchmarkId::new("SimdNTT", seed), seed, |b, _| {
            let ntt = SimdNTT::new();
            b.iter(|| {
                let mut poly = test_data.clone();
                ntt.forward(black_box(&mut poly));
            });
        });
    }

    group.finish();
}

/// Бенчмарк inverse NTT для всех реализаций
fn bench_inverse_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT Inverse");

    for seed in [13, 17, 23].iter() {
        let mut test_data = generate_test_poly(*seed);

        // Применяем forward для подготовки данных
        let basic_ntt = BasicNTT::new();
        basic_ntt.forward(&mut test_data);

        // BasicNTT inverse
        group.bench_with_input(BenchmarkId::new("BasicNTT", seed), seed, |b, _| {
            let ntt = BasicNTT::new();
            b.iter(|| {
                let mut poly = test_data.clone();
                ntt.inverse(black_box(&mut poly));
            });
        });

        // UnrolledNTT inverse
        group.bench_with_input(BenchmarkId::new("UnrolledNTT", seed), seed, |b, _| {
            let ntt = UnrolledNTT::new();
            b.iter(|| {
                let mut poly = test_data.clone();
                ntt.inverse(black_box(&mut poly));
            });
        });

        // SimdNTT inverse
        group.bench_with_input(BenchmarkId::new("SimdNTT", seed), seed, |b, _| {
            let ntt = SimdNTT::new();
            b.iter(|| {
                let mut poly = test_data.clone();
                ntt.inverse(black_box(&mut poly));
            });
        });
    }

    group.finish();
}

/// Бенчмарк полного roundtrip (forward + inverse)
fn bench_roundtrip_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT Roundtrip");

    let test_data = generate_test_poly(13);

    // BasicNTT roundtrip
    group.bench_function("BasicNTT", |b| {
        let ntt = BasicNTT::new();
        b.iter(|| {
            let mut poly = test_data.clone();
            ntt.forward(black_box(&mut poly));
            ntt.inverse(black_box(&mut poly));
        });
    });

    // UnrolledNTT roundtrip
    group.bench_function("UnrolledNTT", |b| {
        let ntt = UnrolledNTT::new();
        b.iter(|| {
            let mut poly = test_data.clone();
            ntt.forward(black_box(&mut poly));
            ntt.inverse(black_box(&mut poly));
        });
    });

    // SimdNTT roundtrip
    group.bench_function("SimdNTT", |b| {
        let ntt = SimdNTT::new();
        b.iter(|| {
            let mut poly = test_data.clone();
            ntt.forward(black_box(&mut poly));
            ntt.inverse(black_box(&mut poly));
        });
    });

    group.finish();
}

/// Дополнительный бенчмарк: сравнение только SIMD версий
fn bench_simd_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Comparison");

    let test_data = generate_test_poly(42);

    // SimdNTT (псевдо-SIMD с prefetch)
    group.bench_function("SimdNTT (pseudo)", |b| {
        let ntt = SimdNTT::new();
        b.iter(|| {
            let mut poly = test_data.clone();
            ntt.forward(black_box(&mut poly));
            ntt.inverse(black_box(&mut poly));
        });
    });

    group.finish();
}

/// Бенчмарк для анализа отдельных операций
fn bench_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Individual Operations");

    let test_data = generate_test_poly(7);

    // Forward only - BasicNTT
    group.bench_function("Forward/BasicNTT", |b| {
        let ntt = BasicNTT::new();
        b.iter(|| {
            let mut poly = test_data.clone();
            ntt.forward(black_box(&mut poly));
        });
    });

    // Forward only - TrueSimdNTT
    group.bench_function("Forward/TrueSimdNTT", |b| {
        let ntt = TrueSimdNTT::new();
        b.iter(|| {
            let mut poly = test_data.clone();
            ntt.forward(black_box(&mut poly));
        });
    });

    // Inverse only - BasicNTT
    let mut transformed = test_data.clone();
    BasicNTT::new().forward(&mut transformed);

    group.bench_function("Inverse/BasicNTT", |b| {
        let ntt = BasicNTT::new();
        b.iter(|| {
            let mut poly = transformed.clone();
            ntt.inverse(black_box(&mut poly));
        });
    });

    group.finish();
}

// Регистрируем все бенчмарки
criterion_group!(
    benches,
    bench_forward_ntt,
    bench_inverse_ntt,
    bench_roundtrip_ntt,
    bench_simd_comparison,
    bench_operations
);

criterion_main!(benches);
