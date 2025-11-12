//! modular arithmetic in Z_q

use crate::params::Q;

const QINV: i32 = -3327; // q^(-1) mod 2^16

#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    let t = ((a as i32 as i64).wrapping_mul(-3327)) as i16 as i64;
    ((a - t * Q as i64) >> 16) as i32
}

#[inline]
pub fn mod_add(a: i32, b: i32) -> i32 {
    let res = a + b;
    if res >= Q {
        res - Q
    } else {
        res
    }
}

#[inline]
pub fn mod_sub(a: i32, b: i32) -> i32 {
    let res = a - b;
    if res < 0 {
        res + Q
    } else {
        res
    }
}

#[inline]
pub fn mod_mul(a: i32, b: i32) -> i32 {
    montgomery_reduce(a as i64 * b as i64)
}

/// Barrett reduction
#[inline]
pub fn barrett_reduce(a: i32) -> i32 {
    let v = ((1i64 << 26) + (Q as i64 / 2)) / Q as i64;
    let t = ((v * a as i64 + (1i64 << 25)) >> 26) as i32;
    a - t * Q
}
