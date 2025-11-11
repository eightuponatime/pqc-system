use pqc_math::{BasicNTT, NTT};

fn main() {
    /// test of basic ntt
    test_basic_ntt();
}

fn test_basic_ntt() {
    let ntt = BasicNTT::new();
    let mut data = [1i32; 256];
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;

    println!("before ntt: {} {} {}", data[0], data[1], data[2]);

    ntt.forward(&mut data);

    println!("after ntt: {} {} {}", data[0], data[1], data[2]);
}
