//! modular arithmetic in Z_q

use crate::params::Q;

#[inline]
pub fn mod_q(x: i32) -> i32 {
  let mut y = x % Q;
  if y < 0 {
    y += Q;
  }
  return y;
}

#[inline]
pub fn mod_mul(a: i32, b: i32) -> i32 {
  return mod_q(a * b);
}

#[inline]
pub fn mod_add(a: i32, b: i32) -> i32 {
  return mod_q(a + b);
}

#[inline]
pub fn mod_sub(a: i32, b: i32) -> i32 {
  return mod_q(a - b);
}
