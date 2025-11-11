use pqc_math::NTT;
use pqc_math::ntt::basic::BasicNTT;

fn main() {
  test_ntt_roundtrip();
}

/// Проверка прямого и обратного преобразования NTT
fn test_ntt_roundtrip() {
  let ntt = BasicNTT::new();
  let mut data = [0i32; 256];
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;

  let original = data;

  ntt.forward(&mut data);
  ntt.inverse(&mut data);

  println!("Original:      {:?}", &original[..8]);
  println!("After inverse: {:?}", &data[..8]);

  if data == original {
    println!("✅ Round-trip successful!");
  } else {
    println!("❌ Round-trip failed!");
  }
}
