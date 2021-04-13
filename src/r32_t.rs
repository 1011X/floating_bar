use core::fmt;
use core::cmp::Ordering;
use core::ops::*;
use core::str::FromStr;

use gcd::Gcd;

use super::ParseRatioErr;

/// The 32-bit floating bar type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, Default)]
pub struct r32(u32);

const DSIZE_SIZE: u32 = 5;
const FRACTION_SIZE: u32 = 27;

const FRACTION_FIELD: u32 = (1 << FRACTION_SIZE) - 1;

impl r32 {
	
	// PRIVATE methods
	
	#[inline]
	fn get_frac_size(n: i64, d: u64) -> u32 {
		let dsize = 64 - d.leading_zeros() - 1;
		let nsize = if n >= 0 {
			64 - n.leading_zeros() + 1
		} else {
			64 - n.leading_ones() + 1
		};
		
		nsize + dsize
	}
	
	// PUBLIC API
	
	/// Creates a rational number without checking the values.
	/// 
	/// # Safety
	/// 
	/// The values must fit in the fraction field.
	#[inline]
	pub const unsafe fn new_unchecked(numer: i32, denom: u32) -> r32 {
		let denom_size = 32 - denom.leading_zeros() - 1;
		let denom_mask = (1 << denom_size) - 1;
		let numer_mask = FRACTION_FIELD & !denom_mask;
		
		r32(
			denom_size << FRACTION_SIZE |
			((numer << denom_size) as u32) & numer_mask |
			denom & denom_mask
		)
	}
	
	/// Creates a rational number if the given values both fit in the fraction
	/// field.
	pub fn new(mut numer: i32, mut denom: u32) -> Option<r32> {
		// do this first anyways to simplify logic stuff
		let gcd = numer.unsigned_abs().gcd(denom);
		numer /= gcd as i32;
		denom /= gcd;
		
		// the `- 1` at the end is the implicit denominator bit
		let denom_size = 32 - denom.leading_zeros() - 1;
		
		// the `+ 1` at the end accounts for the sign
		let numer_size = if numer >= 0 {
			32 - numer.leading_zeros() + 1
		} else {
			32 - numer.leading_ones() + 1
		};
		
		if numer_size + denom_size > FRACTION_SIZE {
			return None;
		}
		
		// SAFETY: we just checked if the values fit.
		unsafe {
			Some(r32::new_unchecked(numer, denom))
		}
	}
	
	/// Calculates the approximate square root of the value.
	/// 
	/// **Warning**: This method can give a number that overflows easily, so
	/// use it with caution, and discard it as soon as you're done with it.
	#[inline]
	pub fn sqrt(self) -> r32 {
		// shh...
		let f: f32 = self.into();
		r32::from(f.sqrt())
		// we should test to make sure that roundtrip calculations give the
		// same value...
	}
	
	/// Calculates the approximate cube root of the value.
	/// 
	/// **Warning**: This method can give a number that overflows easily, so
	/// use it with caution, and discard it as soon as you're done with it.
	#[inline]
	pub fn cbrt(self) -> r32 {
		let f: f32 = self.into();
		r32::from(f.cbrt())
	}
	
	/// Checked addition. Computes `self + rhs`, returning `None` if overflow
	/// occurred.
	pub fn checked_add(self, rhs: r32) -> Option<r32> {
		if self.is_nan() || rhs.is_nan() {
			return Some(r32::NAN);
		}
		// self = a/b, other = c/d
		
		// num = ad + bc
		let mut num =
			self.numer() as i64 * rhs.denom() as i64
			+ self.denom() as i64 * rhs.numer() as i64;
		// den = bd
		let mut den = self.denom() as u64 * rhs.denom() as u64;
		
		let mut size = r32::get_frac_size(num, den);
		
		if size > FRACTION_SIZE {
			let gcd = num.unsigned_abs().gcd(den);
			num /= gcd as i64;
			den /= gcd;
			size = r32::get_frac_size(num, den);
		}
		
		if size <= FRACTION_SIZE {
			unsafe { Some(r32::new_unchecked(num as i32, den as u32)) }
		} else {
			None
		}
	}
	
	/// Checked multiplication. Computes `self * rhs`, returning `None`
	/// if overflow occurred.
	/// 
	/// If one argument is NaN and the other is zero, this returns zero.
	pub fn checked_mul(self, rhs: r32) -> Option<r32> {
		match (self.is_nan(), rhs.is_nan()) {
			(true, false) if rhs.numer() == 0 => return Some(r32(0)),
			(false, true) if self.numer() == 0 => return Some(r32(0)),
			(false, false) => {}
			_ => return Some(r32::NAN),
		}
		
		// a/b * c/d = ac/bd
		let mut n = self.numer() as i64 * rhs.numer() as i64;
		let mut d = self.denom() as u64 * rhs.denom() as u64;
		
		let mut size = r32::get_frac_size(n, d);
		
		if size > FRACTION_SIZE {
			let gcd = n.unsigned_abs().gcd(d);
			n /= gcd as i64;
			d /= gcd;
			size = r32::get_frac_size(n, d);
		}
		
		if size <= FRACTION_SIZE {
			unsafe { Some(r32::new_unchecked(n as i32, d as u32)) }
		} else {
			None
		}
	}
	
	/// Checked conversion to `i32`.
	///
	/// Returns the numeric value as an `i32` if its irreducible form has a 
	/// denominator of 1. Otherwise, returns `None`.
	#[inline]
	pub fn to_i32(self) -> Option<i32> {
		let norm = self.normalize();
		if norm.denom_size() == 0 {
			Some(norm.numer())
		} else {
			None
		}
	}
}

crate::impl_ratio_type! { r32 u32 i32 NonZeroU32 }

impl From<u16> for r32 {
	#[inline]
	fn from(v: u16) -> Self { r32(v as u32) }
}

impl From<i16> for r32 {
	fn from(v: i16) -> Self {
		unsafe { r32::new_unchecked(v as i32, 1) }
	}
}

impl From<f32> for r32 {
	fn from(mut f: f32) -> Self {
		// why 13? bc it's fraction_size / 2
		// div by 2 is to have enough space for both numer and denom.
		// don't count implicit bit because then we can only represent 0 - 0.5
		// in a number that could be 0 - 1.
		const N: u32 = (1 << 13) - 1; // 2^13 - 1 = 8191
		
		let is_neg = f < 0.0;
		
		if f.is_nan() || f.is_infinite() {
			return r32::NAN;
		}
		
		if is_neg { f = f.abs() }
		
		let (mut a, mut b) = (0, 1); // lower
		let (mut c, mut d) = (1, 0); // upper
		let mut is_mediant = false;
		
		// while neither denominator is too big,
		while b <= N && d <= N {
			let mediant = (a + c) as f32 / (b + d) as f32;
			
			if f == mediant {
				is_mediant = true;
				break;
			} else if f > mediant {
				a += c;
				b += d;
			} else {
				c += a;
				d += b;
			}
		}
		
		let result = if is_mediant {
			// if N can contain sum of both denoms,
			if b + d <= N { (a + c, b + d) } // use sum of numers & sum of denoms
			// else if upper bound denom is bigger than lower bound denom,
			else if d > b { (c, d) } // use upper bound
			else          { (a, b) } // else, use lower bound
		} else {
			// if lower bound denom is too big,
			if b > N { (c, d) } // use upper bound
			else     { (a, b) } // else, lower bound
		};
		
		// SAFETY: values were given a maximum in which they could fit.
		unsafe {
			if is_neg {
				r32::new_unchecked(-result.0, result.1)
			} else {
				r32::new_unchecked(result.0, result.1)
			}
		}
	}
}

// TODO add TryFrom version
impl From<r32> for f32 {
	fn from(r: r32) -> f32 {
		if r.is_nan() {
			f32::NAN
		} else {
			(r.numer() as f32) / (r.denom() as f32)
		}
	}
}

impl From<r32> for f64 {
	fn from(r: r32) -> f64 {
		if r.is_nan() {
			f64::NAN
		} else {
			(r.numer() as f64) / (r.denom() as f64)
		}
	}
}

impl PartialOrd for r32 {
	fn partial_cmp(&self, other: &r32) -> Option<Ordering> {
		// both are nan
		if self.is_nan() && other.is_nan() {
			return Some(Ordering::Equal);
		}
		
		// only one of them is nan
		if self.is_nan() || other.is_nan() {
			return None;
		}
		
		Some(
			// compare numbers
			// a/b = c/d <=> ad = bc
			(self.numer() as i64 * other.denom() as i64)
			.cmp(&(self.denom() as i64 * other.numer() as i64))
		)
	}
}

#[cfg(test)]
mod tests {
	#[cfg(feature = "bench")]
	extern crate test;

	use super::*;
	use crate::r32;

	#[test]
	fn checked_pow() {
		assert_eq!(r32(3).checked_pow(30), None);
	}
	
	#[test]
	fn trunc() {
		assert_eq!(r32::NAN.trunc(), r32::NAN);
		
		assert_eq!(r32(5).trunc(),     r32(5));
		assert_eq!(r32!( 1/2).trunc(), r32(0));
		assert_eq!(r32!(-1/2).trunc(), r32(0));
		assert_eq!(r32!( 3/2).trunc(), r32(1));
		assert_eq!(r32!(-3/2).trunc(), r32!(-1));
	}
	
	#[test]
	fn fract() {
		assert_eq!(r32::NAN.fract(), r32::NAN);
		
		assert_eq!(r32(5).fract(),     r32(0));
		assert_eq!(r32!( 3/2).fract(), r32!( 1/2));
		assert_eq!(r32!(-3/2).fract(), r32!(-1/2));
	}

	#[test]
	fn floor() {
		assert_eq!(r32::NAN.floor(), r32::NAN);
		
		assert_eq!(r32(1).floor(),     r32(1));
		assert_eq!(r32!(-1).floor(),   r32!(-1));
		assert_eq!(r32!( 3/2).floor(), r32(1));
		assert_eq!(r32!(-3/2).floor(), r32!(-2));
	}

	#[test]
	fn ceil() {
		assert_eq!(r32::NAN.ceil(), r32::NAN);
		
		assert_eq!(r32(1).ceil(),     r32(1));
		assert_eq!(r32!(-1).ceil(),   r32!(-1));
		assert_eq!(r32!( 3/2).ceil(), r32(2));
		assert_eq!(r32!(-3/2).ceil(), r32!(-1));
	}

	#[test]
	fn round() {
		assert_eq!(r32::NAN.round(), r32::NAN);
		
		assert_eq!(r32(1).round(),     r32(1));
		assert_eq!(r32!(-1).round(),   r32!(-1));
		assert_eq!(r32!( 3/2).round(), r32(2));
		assert_eq!(r32!(-3/2).round(), r32!(-2));
	}
	
	#[test]
	fn min() {
		assert_eq!(r32::NAN.min(r32::NAN), r32::NAN);
		assert_eq!(r32::NAN.min(r32(0)),   r32(0));
		assert_eq!(r32(0).min(r32::NAN),   r32(0));
		assert_eq!(r32(0).min(r32(1)),     r32(0));
	}
	
	#[test]
	fn max() {
		assert_eq!(r32::NAN.max(r32::NAN), r32::NAN);
		assert_eq!(r32::NAN.max(r32(0)),   r32(0));
		assert_eq!(r32(0).max(r32::NAN),   r32(0));
		assert_eq!(r32(0).max(r32(1)),     r32(1));
	}
	
	#[test]
	fn abs() {
		assert_eq!(r32::NAN.abs(), r32::NAN);
		assert_eq!(r32(0).abs(), r32(0));
		assert_eq!(r32(1).abs(), r32(1));
		
		assert_eq!(r32!(-1).abs(), r32(1));
	}
	
	#[test]
	fn signum() {
		assert_eq!(r32::NAN.signum(), r32::NAN);
		
		assert_eq!(r32(0).signum(), r32(0));
		assert_eq!(r32(1).signum(), r32(1));
		assert_eq!(r32(2).signum(), r32(1));
		
		assert_eq!(r32!(-1).signum(), r32!(-1));
		assert_eq!(r32!(-2).signum(), r32!(-1));
	}
	
	#[test]
	fn recip() {
		assert_eq!(r32::NAN.recip(), r32::NAN);
		
		assert_eq!(r32(5).recip(), r32!(1/5));
		assert_eq!(r32!(5/2).recip(), r32!(2/5));
		assert_eq!(r32(1).recip(), r32(1));
	}
	
	#[test]
	fn normalize() {
		assert_eq!(r32!( 4 / 2).normalize(), r32!( 2));
		assert_eq!(r32!(-4 / 2).normalize(), r32!(-2));
	}

	#[test]
	fn pow() {
		assert_eq!( r32::NAN.pow(0), r32(1) );
		
		assert_eq!( r32(0).pow(0),   r32(1) );
		assert_eq!( r32(1).pow(1),   r32(1) );
		
		assert_eq!( r32(3).pow( 2),   r32(9)    );
		assert_eq!( r32(3).pow(-2),   r32!(1/9) );
		assert_eq!( r32!(-3).pow( 2), r32(9)    );
		assert_eq!( r32!(-3).pow(-2), r32!(1/9) );
		
		assert_eq!( r32(2).pow( 3),    r32(8)    );
		assert_eq!( r32(2).pow(-3),    r32!(1/8) );
		assert_eq!( r32!(1/2).pow( 3), r32!(1/8) );
		assert_eq!( r32!(1/2).pow(-3), r32(8)    );
		
		assert_eq!( r32!(-2).pow( 3),   r32!(-8)   );
		assert_eq!( r32!(-2).pow(-3),   r32!(-1/8) );
		assert_eq!( r32!(-1/2).pow( 3), r32!(-1/8) );
		assert_eq!( r32!(-1/2).pow(-3), r32!(-8)   );
	}
	
	#[test]
	fn cmp() {
		assert!(r32(0) == r32(0));
		
		assert!(r32(0) < r32(1));
		assert!(r32(2) < r32(3));
		assert!(r32(0) > -r32(1));
		assert!(r32(2) > -r32(3));
		
		// TODO more assertions here
	}
	
	#[test]
	fn neg() {
		assert_eq!(-r32!( 0), r32!( 0));
		assert_eq!(-r32!( 1), r32!(-1));
		assert_eq!(-r32!(-1), r32!( 1));
	}
	
	#[test]
	fn checked_neg() {
		assert_eq!(r32!(-1/67108864).checked_neg(), None);
	}
	
	#[test]
	fn add() {
		assert_eq!(r32(0) + r32(0), r32(0));
		
		assert_eq!(r32(1) + r32(1), r32(2));
		assert_eq!(r32(1) + r32!(-1), r32(0));
		assert_eq!(r32!(-1) + r32(1), r32(0));
		assert_eq!(r32!(-1) + r32!(-1), r32!(-2));
		
		assert_eq!(r32(2)     + r32(2),     r32(4));
		assert_eq!(r32!(1/2)  + r32!(3/4),  r32!(5/4));
		assert_eq!(r32!(1/2)  + r32!(-3/4), r32!(-1/4));
		assert_eq!(r32!(-1/2) + r32!(3/4),  r32!(1/4));
	}
	
	#[test] #[should_panic]
	fn add_invalid() {
		let _ = r32(1 << FRACTION_SIZE - 1) + r32(1 << FRACTION_SIZE - 1);
	}
	
	#[test]
	fn mul() {
		assert_eq!(r32(0) * r32(0), r32(0));
		
		assert_eq!(r32(0) * r32(1), r32(0));
		assert_eq!(r32(1) * r32(0), r32(0));
		assert_eq!(r32(1) * r32(1), r32(1));
		
		assert_eq!(-r32(1) *  r32(1), -r32(1));
		assert_eq!( r32(1) * -r32(1), -r32(1));
		assert_eq!(-r32(1) * -r32(1),  r32(1));
		
		assert_eq!(r32(1) * r32(2), r32(2));
		assert_eq!(r32(2) * r32(2), r32(4));
		
		assert_eq!(
			r32!(1/2) * r32!(1/2), r32!(1/4)
		);
		assert_eq!(
			r32!(-1/2) * r32!(1/2), r32!(-1/4)
		);
		assert_eq!(
			r32!(2/3) * r32!(2/3), r32!(4/9)
		);
		assert_eq!(
			r32!(3/2) * r32!(2/3), r32(1)
		);
	}
	
	#[test] #[should_panic]
	fn mul_invalid() {
		let _ = r32(1 << FRACTION_SIZE - 1) * r32(1 << FRACTION_SIZE - 1);
	}
	
	#[test]
	fn div() {
		assert_eq!(r32(0) / r32(1), r32(0));
		assert_eq!(r32(0) / r32(2), r32(0));
		assert_eq!(r32(1) / r32(1), r32(1));
		
		assert_eq!(-r32(1) /  r32(1), -r32(1));
		assert_eq!( r32(1) / -r32(1), -r32(1));
		assert_eq!(-r32(1) / -r32(1),  r32(1));
		
		assert_eq!(r32(1) / r32(2), r32!(1/2));
		assert_eq!(r32(2) / r32(1), r32(2));
		assert_eq!(r32(2) / r32(2), r32(1));
	}

	#[test]
	fn rem() {
		assert_eq!(r32(5) % r32(2), r32(1));
		assert_eq!(r32(6) % r32(2), r32(0));
		assert_eq!(r32(8) % (r32(3) / r32(2)), r32(1) / r32(2));
		
		// always returns sign of divisor (2nd number)
		assert_eq!(-r32(5) %  r32(2),  r32(1));
		assert_eq!( r32(5) % -r32(2), -r32(1));
		assert_eq!(-r32(5) % -r32(2), -r32(1));
	}
	
	#[test]
	fn from_str() {
		assert_eq!("NaN".parse::<r32>().unwrap(), r32::NAN);
		assert_eq!("0".parse::<r32>().unwrap(),   r32(0));
		assert_eq!("1".parse::<r32>().unwrap(),   r32(1));
		assert_eq!("+1".parse::<r32>().unwrap(),  r32(1));
		assert_eq!("-1".parse::<r32>().unwrap(),  r32!(-1));
		assert_eq!("1/1".parse::<r32>().unwrap(), r32(1));
	}
	
	#[test] #[should_panic]
	fn from_str_invalid() {
		"1/-1".parse::<r32>().unwrap();
		"/1".parse::<r32>().unwrap();
		"1/".parse::<r32>().unwrap();
		"1/0".parse::<r32>().unwrap();
	}
	
	#[test]
	fn from_f32() {
		assert_eq!(r32::from(0.0), r32(0));
		assert_eq!(r32::from(1.0), r32(1));
		assert_eq!(r32::from(-1.0), -r32(1));
		assert_eq!(r32::from(0.2), r32(1) / r32(5));
		assert_eq!(r32::from(1.0 - 0.7), r32(3) / r32(10));
		//assert_eq!(r32::from(std::f32::consts::E), r32(15062) / r32(5541));
		//assert_eq!(r32::from(std::f32::consts::TAU), r32(710) / r32(113));
	}
}
