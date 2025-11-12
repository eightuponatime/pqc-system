use pqc_math::{BasicNTT, NTT};

fn main() {
    let ntt = BasicNTT::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    println!("Before NTT (first 8): {:?}", &poly[..8]);
    let original = poly.clone();

    ntt.forward(&mut poly);
    println!("After forward (first 8): {:?}", &poly[..8]);

    ntt.inverse(&mut poly);
    println!("After inverse (first 8): {:?}", &poly[..8]);
    println!("Expected (first 8): {:?}", &original[..8]);

    // Проверим где отличается
    let mut diff_count = 0;
    for i in 0..256 {
        if poly[i] != original[i] {
            if diff_count < 5 {
                println!("Diff at {}: got {}, expected {}", i, poly[i], original[i]);
            }
            diff_count += 1;
        }
    }
    println!("Total differences: {}/256", diff_count);

    if poly == original {
        println!("✓ NTT correctness test passed");
    } else {
        println!("✗ NTT correctness test FAILED");
    }
}
