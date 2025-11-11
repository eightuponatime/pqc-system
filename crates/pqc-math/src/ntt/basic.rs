//! Basic NTT implementation

use super::NTT;
use crate::params::{ROOT};
use crate::modular::mod_q;

pub struct BasicNTT {
  twiddles: [i32; 128],
}

impl BasicNTT {
  pub fn new() -> Self {
    Self {
      twiddles: Self::precompute_twiddles(),
    }
  }

  fn precompute_twiddles() -> [i32; 128] {
    let mut twiddles = [0; 128];
    let mut zeta = 1;
    for i in 0..128 {
      twiddles[i] = zeta;
      zeta = mod_q(zeta * ROOT);
    }
    twiddles
  }
}

impl NTT for BasicNTT {
    fn forward(&self, a: &mut [i32; 256]) {
      let mut len = 128;
      let mut k = 1;

      while len >= 1 {
        for start in (0..256).step_by(len * 2) {
          let zeta = self.twiddles[k];
          for j in 0..len {
            let t = mod_q(zeta * a[start + j + len]);
            a[start + j + len] = mod_q(a[start + j] - t);
            a[start + j] = mod_q(a[start + j] + t);
          }
          k += 1;
        }
        len /= 2;
      }
    }

    fn inverse(&self, _a: &mut [i32; 256]) {
      todo!("Inverse NTT")
    }

    fn name(&self) -> &'static str {
      "Basic NTT"
    }
}