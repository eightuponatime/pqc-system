use pqc_math::{BasicNTT, NTT};

fn main() {
  println!("=== Testing Basic NTT ===\n");

  let ntt = BasicNTT::new();
  println!("created {}", ntt.name());

  let mut data = [0i32; 256];
  for i in 0..256 {
    data[i] = i as i32;
  }

  println!("before NTT: [{}, {}, {}, ...]", data[0], data[1], data[2]);

  ntt.forward(&mut data);

  println!("after NTT:  [{}, {}, {}, ...]", data[0], data[1], data[2]);

  println!("\n=== Testing with simple data ===\n");

  let mut simple = [1i32; 256];
  println!("Input: all ones");

  ntt.forward(&mut simple);
  println!("Output first 5: [{}, {}, {}, {}, {}]",
            simple[0], simple[1], simple[2], simple[3], simple[4]);

  // compare_implementations();
}

fn compare_implementations() {
    use std::time::Instant;

    let ntt = BasicNTT::new();
    let mut data = [1i32; 256];

    let start = Instant::now();
    for _ in 0..10000 {
        ntt.forward(&mut data);
    }
    let elapsed = start.elapsed();

    println!("Basic NTT: 10000 iterations in {:?}", elapsed);
    println!("Average: {:?} per iteration", elapsed / 10000);
}