use core::fmt;
use core::cmp::Ordering;
use core::ops::*;
use core::str::FromStr;

use gcd::Gcd;

use super::{ParseRatioErr, r32};

/// The 64-bit floating bar type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, Default)]
pub struct r64(u64);

const DSIZE_SIZE: u32 = 6;
const FRACTION_SIZE: u64 = 58;

const FRACTION_FIELD: u64 = (1 << FRACTION_SIZE) - 1;

impl r64 {
	// PRIVATE methods
	#[inline]
	fn get_frac_size(n: i128, d: u128) -> u64 {
		let dsize = 128 - d.leading_zeros() - 1;
		let nsize = if n >= 0 {
			128 - n.leading_zeros() + 1
		} else {
			128 - n.leading_ones() + 1
		};
		
		(nsize + dsize) as u64
	}
	
	// PUBLIC API
	
	/// Creates a rational number without checking the values.
	/// 
	/// # Safety
	/// 
	/// The values must fit in the fraction field.
	#[inline]
	pub const unsafe fn new_unchecked(numer: i64, denom: u64) -> r64 {
		let denom_size = 64 - denom.leading_zeros() - 1;
		let denom_mask = (1 << denom_size as u64) - 1;
		let numer_mask = FRACTION_FIELD & !denom_mask;
		
		r64(
			(denom_size as u64) << FRACTION_SIZE |
			((numer << denom_size) as u64) & numer_mask |
			denom & denom_mask
		)
	}
	
	/// Creates a rational number if the given values both fit in the fraction
	/// field.
	pub fn new(mut numer: i64, mut denom: u64) -> Option<r64> {
		let gcd = numer.unsigned_abs().gcd(denom);
		numer /= gcd as i64;
		denom /= gcd;
		
		let denom_size = 64 - denom.leading_zeros() - 1;
		let numer_size = if numer >= 0 {
			64 - numer.leading_zeros() + 1
		} else {
			64 - numer.leading_ones() + 1
		};
		
		if numer_size + denom_size > FRACTION_SIZE as u32 {
			return None;
		}
		
		// SAFETY: we just checked if the values fit.
		unsafe {
			Some(r64::new_unchecked(numer, denom))
		}
	}
	
	/// Calculates the approximate square root of the value.
	/// 
	/// **Warning**: This method can give a number that overflows easily, so
	/// use it with caution, and discard it as soon as you're done with it.
	pub fn sqrt(self) -> r64 {
		// shh...
		let f: f64 = self.into();
		r64::from(f.sqrt())
	}
	
	/// Calculates the approximate cube root of the value.
	/// 
	/// **Warning**: This method can give a number that overflows easily, so
	/// use it with caution, and discard it as soon as you're done with it.
	pub fn cbrt(self) -> r64 {
		// shh...
		let f: f64 = self.into();
		r64::from(f.cbrt())
	}
	
	/// Checked rational addition. Computes `self + rhs`, returning `None` if
	/// overflow occurred.
	pub fn checked_add(self, rhs: r64) -> Option<r64> {
		if self.is_nan() || rhs.is_nan() { return Some(r64::NAN) }
		// self = a/b, other = c/d
		
		// num = ad + bc
		let mut num =
			self.numer() as i128 * rhs.denom() as i128
			+ self.denom() as i128 * rhs.numer() as i128;
		// den = bd
		let mut den = self.denom() as u128 * rhs.denom() as u128;
		
		let mut size = r64::get_frac_size(num, den);
		
		if size > FRACTION_SIZE {
			let gcd = num.unsigned_abs().gcd(den);
			num /= gcd as i128;
			den /= gcd;
			size = r64::get_frac_size(num, den);
		}
		
		if size <= FRACTION_SIZE {
			unsafe { Some(r64::new_unchecked(num as i64, den as u64)) }
		} else {
			None
		}
	}
	
	/// Checked rational multiplication. Computes `self * rhs`, returning `None`
	/// if overflow occurred.
	pub fn checked_mul(self, rhs: r64) -> Option<r64> {
		match (self.is_nan(), rhs.is_nan()) {
			(true, false) if rhs.numer() == 0 => return Some(r64(0)),
			(false, true) if self.numer() == 0 => return Some(r64(0)),
			(false, false) => {}
			_ => return Some(r64::NAN),
		}
		
		// a/b * c/d = ac/bd
		let mut n = self.numer() as i128 * rhs.numer() as i128;
		let mut d = self.denom() as u128 * rhs.denom() as u128;
		
		let mut size = r64::get_frac_size(n, d);
		
		if size > FRACTION_SIZE {
			let gcd = n.unsigned_abs().gcd(d);
			n /= gcd as i128;
			d /= gcd;
			size = r64::get_frac_size(n, d);
		}
		
		if size <= FRACTION_SIZE {
			unsafe { Some(r64::new_unchecked(n as i64, d as u64)) }
		} else {
			None
		}
	}
	
	/// Checked conversion to `i64`.
	///
	/// Returns `i64` value if denominator is 1. Otherwise, returns `None`.
	#[inline]
	pub fn to_i64(self) -> Option<i64> {
		let norm = self.normalize();
		if norm.denom_size() == 0 {
			Some(norm.numer())
		} else {
			None
		}
	}
}

crate::impl_ratio_type! { r64 u64 i64 NonZeroU64 }

impl From<u16> for r64 {
	#[inline]
	fn from(v: u16) -> Self { r64(v as u64) }
}

impl From<i16> for r64 {
	fn from(v: i16) -> Self {
		unsafe { r64::new_unchecked(v as i64, 1) }
	}
}

impl From<u32> for r64 {
	#[inline]
	fn from(v: u32) -> Self { r64(v as u64) }
}

impl From<i32> for r64 {
	fn from(v: i32) -> Self {
		unsafe { r64::new_unchecked(v as i64, 1) }
	}
}

impl From<f32> for r64 {
	fn from(f: f32) -> Self { r64::from(f as f64) }
}

impl From<f64> for r64 {
	/// Based on [John D. Cook's Best Rational Approximation post](https://www.johndcook.com/blog/2010/10/20/best-rational-approximation/)
	fn from(mut f: f64) -> Self {
		// why 29? bc it's fraction_size / 2 + 1
		// div by 2 is to have enough space for both numer and denom.
		// plus 1 is to count implicit bit bc numer and denom can both have 29
		// bits of precision here.
		const N: u64 = (1 << 29) - 1; // 2^29 - 1 = 536870911
		let is_neg = f < 0.0;
		
		if f.is_nan() || f.is_infinite() {
			return r64::NAN;
		}
		
		if is_neg { f = f.abs() }
		
		let (mut a, mut b) = (0, 1); // lower
		let (mut c, mut d) = (1, 0); // upper
		let mut is_mediant = false;
		
		// while neither denominator is too big,
		while b <= N && d <= N {
			let mediant = (a + c) as f64 / (b + d) as f64;
			
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
				r64::new_unchecked(-result.0, result.1)
			} else {
				r64::new_unchecked(result.0, result.1)
			}
		}
	}
}

impl From<r32> for r64 {
	fn from(v: r32) -> Self {
		unsafe { r64::new_unchecked(v.numer() as i64, v.denom() as u64) }
	}
}

impl From<r64> for f32 {
	fn from(r: r64) -> f32 {
		if r.is_nan() {
			f32::NAN
		} else {
			(r.numer() as f32) / (r.denom() as f32)
		}
	}
}

impl From<r64> for f64 {
	fn from(r: r64) -> f64 {
		if r.is_nan() {
			f64::NAN
		} else {
			(r.numer() as f64) / (r.denom() as f64)
		}
	}
}

impl PartialOrd for r64 {
	fn partial_cmp(&self, other: &r64) -> Option<Ordering> {
		// both are nan or both are zero
		if self.is_nan() && other.is_nan() {
			return Some(Ordering::Equal);
		}
		
		// only one of them is nan
		if self.is_nan() || other.is_nan() {
			return None;
		}
		
		// compare signs
		self.is_positive()
		.partial_cmp(&other.is_positive())
		.map(|c| c.then(
			// compare numbers
			// a/b = c/d <=> ad = bc
			// when a, b, c, and d are all > 0
			(self.numer() as i128 * other.denom() as i128)
			.cmp(&(self.denom() as i128 * other.numer() as i128))
		))
	}
}


#[cfg(test)]
mod tests {
	#[cfg(feature = "bench")]
	extern crate test;
	
	use super::*;
	use super::super::r64;
	
	#[test]
	fn normalize() {
		assert_eq!(r64!(4/2).normalize(), r64!(2));
		assert_eq!(r64!(-4/2).normalize(), r64!(-2));
	}
	
	#[test]
	fn neg() {
		assert_eq!(-r64(0), r64(0));
		assert_eq!(-r64(1), r64!(-1));
		assert_eq!(-r64!(-1), r64(1));
	}
	
	#[test]
	fn signum() {
		assert_eq!(r64::NAN.signum(), r64::NAN);
		assert_eq!(r64(0).signum(), r64(0));
		assert_eq!(r64(1).signum(), r64(1));
		assert_eq!(r64(2).signum(), r64(1));
		assert_eq!(r64!(-1).signum(), r64!(-1));
		assert_eq!(r64!(-2).signum(), r64!(-1));
	}

	#[test]
	fn pow() {
		assert_eq!(r64::NAN.pow(0), r64(1));
		
		assert_eq!(r64(0).pow(0),   r64(1));
		assert_eq!(r64(1).pow(1),   r64(1));
		
		assert_eq!(r64(3).pow(2),    r64(9));
		assert_eq!(r64(3).pow(-2),   r64!(1/9));
		assert_eq!(r64!(-3).pow(2),  r64(9));
		assert_eq!(r64!(-3).pow(-2), r64!(1/9));
		
		assert_eq!(r64(2).pow(3),     r64(8));
		assert_eq!(r64(2).pow(-3),    r64!(1/8));
		assert_eq!(r64!(1/2).pow(3),  r64!(1/8));
		assert_eq!(r64!(1/2).pow(-3), r64(8));
		
		assert_eq!(r64!(-2/1).pow(3),  r64!(-8/1));
		assert_eq!(r64!(-2/1).pow(-3), r64!(-1/8));
		assert_eq!(r64!(-1/2).pow(3),  r64!(-1/8));
		assert_eq!(r64!(-1/2).pow(-3), r64!(-8/1));
	}

	#[test]
	fn checked_pow() {
		assert_eq!(r64::NAN.checked_pow(0), Some(r64(1)));
		assert_eq!(r64(3).checked_pow(60), None);
	}

	#[test]
	#[cfg(feature = "roots")]
	fn checked_sqrt() {
		assert_eq!(r64(0).checked_sqrt(), Some(r64(0)));
		assert_eq!(r64(1).checked_sqrt(), Some(r64(1)));
		assert_eq!(r64(2).checked_sqrt(), None);
		assert_eq!(r64(4).checked_sqrt(), Some(r64(2)));
	}
	
	#[test]
	fn fract() {
		assert_eq!(r64(5).fract(),     r64(0));
		assert_eq!(r64!(3/2).fract(),  r64!( 1/2));
		assert_eq!(r64!(-3/2).fract(), r64!(-1/2));
	}
	
	#[test]
	fn trunc() {
		assert_eq!(r64(5).trunc(),     r64(5));
		assert_eq!(r64!( 1/2).trunc(), r64(0));
		assert_eq!(r64!(-1/2).trunc(), r64(0));
		assert_eq!(r64!( 3/2).trunc(), r64(1));
		assert_eq!(r64!(-3/2).trunc(), r64!(-1));
	}

	#[test]
	fn floor() {
		assert_eq!(r64!( 3/2).floor(), r64(1));
		assert_eq!(r64!( 2/1).floor(), r64(2));
		assert_eq!(r64!(-3/2).floor(), r64!(-2));
		assert_eq!(r64!(-2/1).floor(), r64!(-2));
	}

	#[test]
	fn ceil() {
		assert_eq!(r64!( 3/2).ceil(), r64(2));
		assert_eq!(r64!( 2/1).ceil(), r64(2));
		assert_eq!(r64!(-3/2).ceil(), r64!(-1));
		assert_eq!(r64!(-2/1).ceil(), r64!(-2));
	}

	#[test]
	fn round() {
		assert_eq!(r64(1).round(),     r64(1));
		assert_eq!(r64!(-1).round(),   r64!(-1));
		assert_eq!(r64!( 3/2).round(), r64(2));
		assert_eq!(r64!(-3/2).round(), r64!(-2));
	}
	
	#[test]
	fn min() {
		assert_eq!(r64::NAN.min(r64::NAN), r64::NAN);
		assert_eq!(r64::NAN.min(r64(0)),   r64(0));
		assert_eq!(r64(0).min(r64::NAN),   r64(0));
		assert_eq!(r64(0).min(r64(1)),     r64(0));
	}
	
	#[test]
	fn max() {
		assert_eq!(r64::NAN.max(r64::NAN), r64::NAN);
		assert_eq!(r64::NAN.max(r64(0)),   r64(0));
		assert_eq!(r64(0).max(r64::NAN),   r64(0));
		assert_eq!(r64(0).max(r64(1)),     r64(1));
	}
	
	#[test]
	fn recip() {
		assert_eq!(r64(5).recip(),    r64!(1/5));
		assert_eq!(r64!(5/2).recip(), r64!(2/5));
		assert_eq!(r64(1).recip(),    r64(1));
	}
	
	#[test]
	fn cmp() {
		assert!(r64(0) == r64(0));
		
		assert!(r64(0) < r64(1));
		assert!(r64(2) < r64(3));
		assert!(r64(0) > -r64(1));
		assert!(r64(2) > -r64(3));
		
		// TODO add more assertions here
	}
	
	#[test]
	fn mul() {
		assert_eq!(r64(0) * r64(0), r64(0));
		
		assert_eq!(r64(0) * r64(1), r64(0));
		assert_eq!(r64(1) * r64(0), r64(0));
		assert_eq!(r64(1) * r64(1), r64(1));
		
		assert_eq!(-r64(1) *  r64(1), -r64(1));
		assert_eq!( r64(1) * -r64(1), -r64(1));
		assert_eq!(-r64(1) * -r64(1),  r64(1));
		
		assert_eq!(r64(1) * r64(2), r64(2));
		assert_eq!(r64(2) * r64(2), r64(4));
		
		assert_eq!(
			r64!(1/2) * r64!(1/2), r64!(1/4)
		);
		assert_eq!(
			r64!(-1/2) * r64!(1/2), r64!(-1/4)
		);
		assert_eq!(
			r64!(2/3) * r64!(2/3), r64!(4/9)
		);
		assert_eq!(
			r64!(3/2) * r64!(2/3), r64(1)
		);
	}
	
	#[test] #[should_panic]
	fn mul_invalid() {
		let _ = r64(1 << FRACTION_SIZE - 1) * r64(1 << FRACTION_SIZE - 1);
	}
	
	#[test]
	fn div() {
		assert_eq!(r64(0) / r64(1), r64(0));
		assert_eq!(r64(0) / r64(2), r64(0));
		assert_eq!(r64(1) / r64(1), r64(1));
		
		assert_eq!(-r64(1) / r64(1), -r64(1));
		assert_eq!(r64(1) / -r64(1), -r64(1));
		assert_eq!(-r64(1) / -r64(1), r64(1));
		
		assert_eq!(r64(1) / r64(2), r64!(1/2));
		assert_eq!(r64(2) / r64(1), r64(2));
		assert_eq!(r64(2) / r64(2), r64(1));
	}

	#[test]
	fn rem() {
		assert_eq!(r64(5) % r64(2), r64(1));
		assert_eq!(r64(6) % r64(2), r64(0));
		assert_eq!(r64(8) % r64!(3/2), r64!(1/2));
		
		assert_eq!(-r64(5) %  r64(2),  r64(1));
		assert_eq!( r64(5) % -r64(2), -r64(1));
		assert_eq!(-r64(5) % -r64(2), -r64(1));
	}
	
	#[test]
	fn add() {
		assert_eq!(r64(0) + r64(0), r64(0));
		assert_eq!(-r64(0) + r64(0), r64(0));
		
		assert_eq!( r64(1) +  r64(1),  r64(2));
		assert_eq!( r64(1) + -r64(1),  r64(0));
		assert_eq!(-r64(1) +  r64(1),  r64(0));
		assert_eq!(-r64(1) + -r64(1), -r64(2));
		
		assert_eq!(r64(2) + r64(2), r64(4));
		assert_eq!(
			r64!(1/2) + r64!(3/4), r64!(5/4)
		);
		assert_eq!(
			r64!(1/2) + r64!(-3/4), r64!(-1/4)
		);
		assert_eq!(
			r64!(-1/2) + r64!(3/4), r64!(1/4)
		);
	}
	
	#[test] #[should_panic]
	fn add_invalid() {
		let _ = r64(1 << FRACTION_SIZE - 1) + r64(1 << FRACTION_SIZE - 1);
	}
	
	#[test]
	fn from_str() {
		assert_eq!("0".parse::<r64>().unwrap(),   r64(0));
		assert_eq!("1".parse::<r64>().unwrap(),   r64(1));
		assert_eq!("+1".parse::<r64>().unwrap(),  r64(1));
		assert_eq!("-1".parse::<r64>().unwrap(),  r64!(-1));
		assert_eq!("1/1".parse::<r64>().unwrap(), r64(1));
	}
	
	#[test] #[should_panic]
	fn from_str_fail() {
		"1/-1".parse::<r64>().unwrap();
		"/1".parse::<r64>().unwrap();
		"1/".parse::<r64>().unwrap();
		"1/0".parse::<r64>().unwrap();
	}
	
	#[test]
	fn from_f32() {
		//assert_eq!(r64::from(std::f32::consts::E), r64(2850325) / r64(1048576));
		//assert_eq!(r64::from(std::f32::consts::TAU), r64(13176795) / r64(2097152));
	}
	/*
	#[test]
	fn from_f64() {
		assert_eq!(r64::from(0.0), r64(0));
		assert_eq!(r64::from(1.0), r64(1));
		assert_eq!(r64::from(-1.0), -r64(1));
		assert_eq!(r64::from(0.2), r64(1) / r64(5));
		assert_eq!(r64::from(1.0 - 0.7), r64(3) / r64(10));
		//assert_eq!(r64::from(std::f64::consts::E), r64(268876667) / r64(98914198));
		//assert_eq!(r64::from(std::f64::consts::TAU), r64(411557987) / r64(65501488));
	}
	*/
}
