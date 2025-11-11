//! ntt realizations comparisons

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pqc_math::{BasicNTT, NTT};

fn benchmark_basic_ntt(c: &mut Criterion) {
  let mut group = c.benchmark_group("NTT");

  let ntt = BasicNTT::new();
  let mut data = [1i32; 256];

  group.bench_function("basic_forward", |b| {
    b.iter(|| {
      let mut d = data;
      ntt.forward(black_box(&mut d));
    })
  });

  group.finish();
}

criterion_group!(benches, benchmark_basic_ntt);
criterion_main!(benches);
