#![allow(dead_code)]
#![allow(unused_variables)]

use core::fmt;
use core::cmp::Ordering;
use core::ops::*;
use core::str::FromStr;

use gcd::Gcd;
use integer_sqrt::IntegerSquareRoot;

use super::{ParseRatioErr, r32};

/// The 64-bit floating bar type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, Default)]
pub struct r64(u64);

const FRACTION_SIZE: u64 = 58;

const FRACTION_FIELD: u64 = (1 << FRACTION_SIZE) - 1;
const SIZE_FIELD: u64 = !FRACTION_FIELD;

impl r64 {
	/// The highest value that can be represented by this rational type.
	pub const MAX: r64 = r64(1 << (FRACTION_SIZE - 1));
	
	/// The lowest value that can be represented by this rational type.
	pub const MIN: r64 = r64(1 << FRACTION_SIZE);
	
	/// The smallest positive normal value that can be represented by this
	/// rational type.
	pub const MIN_POSITIVE: r64 = r64(FRACTION_SIZE << FRACTION_SIZE | FRACTION_FIELD);
	
	/// Not a Number (NaN).
	pub const NAN: r64 = r64(SIZE_FIELD);
	
	/// Creates a rational number from a signed numerator and an unsigned
	/// denominator.
	#[inline]
	pub fn new(numer: i64, denom: u64) -> r64 {
		let denom_size = 64 - denom.leading_zeros() - 1;
		let numer_size = if numer >= 0 {
			64 - numer.leading_zeros() + 1
		} else {
			64 - numer.leading_ones() + 1
		};
		
		if numer_size + denom_size > FRACTION_SIZE as u32 {
			panic!("numbers are too big")
		}
		
		let denom_mask = (1 << (denom_size as u64)) - 1;
		let numer_mask = FRACTION_FIELD & !denom_mask;
		
		r64(
			(denom_size as u64) << FRACTION_SIZE |
			((numer << denom_size as i64) as u64) & numer_mask |
			denom & denom_mask
		)
	}
	
	#[inline]
	fn denom_size(self) -> u64 {
		self.0 >> FRACTION_SIZE
	}
	
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
	
	/// Returns the numerator value for this rational number.
	#[inline]
	pub fn numer(self) -> i64 {
		let denom_size = self.denom_size();
		// apparently this does sign-extension
		(self.0 as i64).wrapping_shl(6).wrapping_shr(6 + denom_size as u32)
	}
	
	/// Returns the denominator of this rational number.
	#[inline]
	pub fn denom(self) -> u64 {
		let denom_region = (1 << self.denom_size()) - 1;
		1 << self.denom_size() | self.0 & denom_region
	}
	
	// BEGIN related float stuff
	
	/// Returns the integer part of a number.
	#[inline]
	pub fn trunc(self) -> r64 {
		r64::new(self.numer() / self.denom() as i64, 1)
	}
	
	/// Returns the fractional part of a number.
	#[inline]
	pub fn fract(self) -> r64 {
		let d = self.denom();
		r64::new(self.numer() % (d as i64), d)
	}
	
	/// Returns the largest integer less than or equal to a number.
	pub fn floor(self) -> r64 {
		if self.is_negative() {
			// if self is a whole number,
			if self.numer() % (self.denom() as i64) == 0 {
				self
			} else {
				self.trunc() - r64(1)
			}
		} else {
			self.trunc()
		}
	}
	
	/// Returns the smallest integer greater than or equal to a number.
	pub fn ceil(self) -> r64 {
		if self.is_negative() {
			self.trunc()
		} else {
			// if self is a whole number,
			if self.numer() % (self.denom() as i64) == 0 {
				self
			} else {
				self.trunc() + r64(1)
			}
		}
	}
	
	/// Returns the nearest integer to a number. Round half-way cases away from
	/// zero.
	pub fn round(self) -> r64 {
		if self.is_negative() {
			self - r64(1) / r64(2)
		} else {
			self + r64(1) / r64(2)
		}
		.trunc()
	}
	
	/// Computes the absolute value of `self`. Returns NaN if the number is NaN.
	#[inline]
	pub fn abs(self) -> r64 {
		r64::new(self.numer().abs(), self.denom())
	}
	
	/// Returns a number that represents the sign of `self`.
	/// 
	/// * `1` if the number is positive
	/// * `-1` if the number is negative
	/// * `0` if the number is `+0`, `-0`, or `NaN`
	pub fn signum(self) -> r64 {
		r64::new(self.numer().signum(), 1)
	}
	
	/// Raises a number to an integer power.
	/// 
	/// # Panics
	/// 
	/// Panics on overflow.
	#[cfg(not(feature = "quiet-nan"))]
	#[inline]
	pub fn pow(self, exp: i32) -> r64 {
		self.checked_pow(exp).expect("attempt to multiply with overflow")
	}
	
	/// Raises a number to an integer power.
	/// 
	/// Returns NaN on overflow.
	#[cfg(feature = "quiet-nan")]
	#[inline]
	pub fn pow(self, exp: i32) -> r32 {
		self.checked_pow(exp).unwrap_or(r32::NAN)
	}
	
	/// Calculates the approximate square root of the value.
	/// 
	/// **Warning**: This method can give a number that overflows easily, so
	/// use it with caution, and discard it as soon as you're done with it.
	fn sqrt(self) -> r64 {
		// TODO: If `self` is positive, this should approximate its square root
		// by calculating a repeated fraction for a fixed number of steps.
		let f: f64 = self.into();
		r64::from(f.sqrt())
	}
	/*
	TODO consider whether to actually add these.
	/// Calculates the approximate cube root of the value.
	/// 
	/// If `self` is positive and its numerator and denominator are perfect
	/// cubes, returns their cube root. Otherwise, returns `None`.
	pub fn checked_cbrt(self) -> Option<r64> {
		todo!()
	}
	*/
	/// Returns `true` if this value is `NaN` and `false` otherwise.
	#[inline]
	pub fn is_nan(self) -> bool {
		self.denom_size() >= FRACTION_SIZE
	}
	
	/// Returns `true` if the number is neither zero, subnormal, or `NaN`.
	#[inline]
	fn is_normal(self) -> bool {
		self.numer() != 0
		&& self.denom_size() < FRACTION_SIZE
	}
	
	/// Returns `true` if `self` is positive and `false` if the number is zero,
	/// negative, or `NaN`.
	#[inline]
	pub fn is_positive(self) -> bool {
		!self.is_nan()
		&& self.numer().is_positive()
	}
	
	/// Returns `true` if and only if self has a negative sign, including
	/// `-0.0` (but not NaNs with negative sign bit).
	#[inline]
	pub fn is_negative(self) -> bool {
		!self.is_nan()
		&& self.numer().is_negative()
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`.
	/// 
	/// # Panics
	/// 
	/// Panics when trying to set a numerator of zero as the denominator.
	#[cfg(not(feature = "quiet-nan"))]
	#[inline]
	pub fn recip(self) -> r64 {
		self.checked_recip().expect("attempt to divide by zero")
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`.
	/// 
	/// If the numerator is zero, this will return `NaN`.
	#[cfg(feature = "quiet-nan")]
	#[inline]
	pub fn recip(self) -> r64 {
		self.checked_recip().unwrap_or(r64::NAN)
	}
	
	/// Returns the maximum of the two numbers.
	/// 
	/// If one of the arguments is `NaN`, then the other argument is returned.
	pub fn max(self, other: r64) -> r64 {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true) => r64::NAN,
			(true, false) => self,
			(false, true) => other,
			(false, false) => match self.partial_cmp(&other).unwrap() {
				Ordering::Less => other,
				_ => self
			}
		}
	}
	
	/// Returns the minimum of the two numbers.
	/// 
	/// If one of the arguments is `NaN`, then the other argument is returned.
	pub fn min(self, other: r64) -> r64 {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true) => r64::NAN,
			(true, false) => self,
			(false, true) => other,
			(false, false) => match self.partial_cmp(&other).unwrap() {
				Ordering::Greater => other,
				_ => self
			}
		}
	}
	
	/// Cancels out common factors between the numerator and the denominator.
	pub fn normalize(self) -> r64 {
		if self.is_nan() {
			return self;
		}
		
		if self.numer() == 0 {
			return r64(0);
		}
		
		let n = self.numer();
		let d = self.denom();
		
		// cancel out common factors
		let gcd = n.unsigned_abs().gcd(d);
		r64::new(n / (gcd as i64), d / gcd)
	}
	
	// BEGIN related integer stuff
	
	/// Checked rational addition. Computes `self + rhs`, returning `None` if
	/// overflow occurred.
	pub fn checked_add(self, rhs: r64) -> Option<r64> {
		// self = a/b, other = c/d
		
		// TODO prove this won't panic/can't overflow.
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
			Some(r64::new(num as i64, den as u64))
		} else {
			None
		}
	}
	
	/// Checked rational subtraction. Computes `self - rhs`, returning `None` if
	/// overflow occurred.
	#[inline]
	pub fn checked_sub(self, rhs: r64) -> Option<r64> {
		self.checked_add(-rhs)
	}
	
	/// Checked rational multiplication. Computes `self * rhs`, returning `None`
	/// if overflow occurred.
	pub fn checked_mul(self, rhs: r64) -> Option<r64> {
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
			Some(r64::new(n as i64, d as u64))
		} else {
			None
		}
	}
	
	/// Checked rational division. Computes `self / rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[inline]
	pub fn checked_div(self, rhs: r64) -> Option<r64> {
		self.checked_mul(rhs.checked_recip()?)
	}
	
	/// Checked rational remainder. Computes `self % rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[doc(hidden)]
	pub fn checked_rem(self, rhs: r64) -> Option<r64> {
		let div = self.checked_div(rhs)?;
		let diff = div.checked_sub(div.floor())?;
		diff.checked_mul(rhs)
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`. Returns `None` if the
	/// numerator is zero.
	#[inline]
	pub fn checked_recip(self) -> Option<r64> {
		if self.numer() == 0 {
			None
		} else {
			let mut denom = self.denom() as i64;
			if self.is_negative() { denom = -denom }
			Some(r64::new(denom, self.numer().unsigned_abs()))
		}
		//assert!(self.denom_size() < FRACTION_SIZE, "subnormal overflow");
	}
	
	/// Takes the *checked* square root of a number.
	/// 
	/// If `self` is positive and both the numerator and denominator are perfect
	/// squares, this returns their square root. Otherwise, returns `None`.
	pub fn checked_sqrt(self) -> Option<r64> {
		if self.is_negative() {
			return None;
		}
		
		let n = self.numer().unsigned_abs().integer_sqrt() as i64;
		let d = self.denom().integer_sqrt();

		if self.numer() == n * n && self.denom() == d * d {
			Some(r64::new(n, d))
		} else {
			None
		}
	}
	/*
	/// Takes the cube root of a number.
	/// 
	/// If `self` is positive and its numerator and denominator are perfect
	/// cubes, this returns their cube root. Otherwise, returns `None`.
	pub fn checked_cbrt(self) -> Option<r32> {
		let n = self.numer().integer_cbrt();
		let d = self.denom().integer_cbrt();

		if self.numer() == n * n * n && self.denom() == d * d * d {
			Some(r32::new(n as i32, d))
		} else {
			None
		}
	}
	*/
	/// Raises a number to an integer power. Returns `None` on overflow.
	pub fn checked_pow(self, exp: i32) -> Option<r64> {
		let exp_is_neg = exp < 0;
		let exp = exp.unsigned_abs();
		
		let num = self.numer().checked_pow(exp)?;
		let den = self.denom().checked_pow(exp)?;
		
		let result = r64::new(num, den);
		
		if exp_is_neg {
			result.checked_recip()
		} else {
			Some(result)
		}
	}
	
	/// Checked conversion to `i64`.
	///
	/// Returns `i64` value if denominator is 1. Otherwise, returns `None`.
	#[inline]
	pub fn to_i64(self) -> Option<i64> {
		if self.denom_size() == 0 {
			Some(self.numer())
		} else {
			None
		}
	}
	
	// BEGIN bitwise representation stuff
	
	/// Raw transmutation to `u64`.
	/// 
	/// Useful if you need access to the payload bits of a NaN value.
	#[inline]
	pub fn to_bits(self) -> u64 { self.0 }
	
	/// Raw transmutation from `u64`.
	#[inline]
	pub fn from_bits(bits: u64) -> r64 { r64(bits) }
}

impl fmt::Display for r64 {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		if self.is_nan() {
			return f.write_str("NaN");
		}
		
		let norm = self.normalize();
		
		norm.numer().fmt(f)?;
		
		if norm.denom_size() > 0 {
			write!(f, "/{}", norm.denom())?;
		}
		
		Ok(())
	}
}

impl fmt::Debug for r64 {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		if self.is_nan() {
			f.write_str("NaN")
		} else {
			write!(f, "{}/{}", self.numer(), self.denom())
		}
	}
}

impl FromStr for r64 {
	type Err = ParseRatioErr;
	
	/// Converts a string in base 10 to a rational.
	/// 
	/// This function accepts strings such as
	/// 
	/// * '157/50'
	/// * '-157/50'
	/// * '25', or equivalently, '25/1'
	/// * 'NaN'
	/// 
	/// Leading and trailing whitespace represent an error.
	/// 
	/// # Return value
	/// 
	/// `Err(ParseRatioError)` if the string did not represent a valid number.
	/// Otherwise, `Ok(n)` where `n` is the floating-bar number represented by
	/// `src`.
	fn from_str(src: &str) -> Result<Self, Self::Err> {
		use core::num::NonZeroU64;
		
		if src.is_empty() {
			return Err(ParseRatioErr::Empty);
		}
		
		// special case NaN
		if src == "NaN" {
			return Ok(r64::NAN);
		}
		
		// lookahead to find dividing bar, if any
		let bar_pos = src.find('/');
		let numer_end = bar_pos.unwrap_or(src.len());
		
		// parse numerator
		let numerator = src[..numer_end]
			.parse::<i64>()
			.map_err(ParseRatioErr::Numerator)?;
		
		// parse optional denominator
		let denominator = bar_pos
			.map(|pos|
				src[pos+1..]
				.parse::<NonZeroU64>()
				.map_err(ParseRatioErr::Denominator)
			) // : Option<Result<u64, ParseRatioErr>>
			.transpose()?
			// : Option<u64>
			.map(NonZeroU64::get)
			.unwrap_or(1);
		
		// ensure parsed numbers fit in fraction field
		let frac_size = r64::get_frac_size(
			numerator as i128,
			denominator as u128
		);
		
		if frac_size > FRACTION_SIZE {
			return Err(ParseRatioErr::Overflow);
		}
		
		Ok(r64::new(numerator, denominator))
	}
}

impl From<u8> for r64 {
	#[inline]
	fn from(v: u8) -> Self { r64(v as u64) }
}

impl From<i8> for r64 {
	fn from(v: i8) -> Self { r64::new(v as i64, 1) }
}

impl From<u16> for r64 {
	#[inline]
	fn from(v: u16) -> Self { r64(v as u64) }
}

impl From<i16> for r64 {
	fn from(v: i16) -> Self { r64::new(v as i64, 1) }
}

impl From<u32> for r64 {
	#[inline]
	fn from(v: u32) -> Self { r64(v as u64) }
}

impl From<i32> for r64 {
	fn from(v: i32) -> Self { r64::new(v as i64, 1) }
}

impl From<r32> for r64 {
	fn from(v: r32) -> Self {
		r64::new(v.numer() as i64, v.denom() as u64)
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
		
		if is_neg {
			r64::new(-result.0, result.1)
		} else {
			r64::new(result.0, result.1)
		}
	}
}

impl From<r64> for f32 {
	fn from(r: r64) -> f32 {
		(r.numer() as f32) / (r.denom() as f32)
	}
}

impl From<r64> for f64 {
	fn from(r: r64) -> f64 {
		(r.numer() as f64) / (r.denom() as f64)
	}
}

impl Neg for r64 {
	type Output = r64;
	
	fn neg(self) -> Self::Output {
		r64::new(-self.numer(), self.denom())
	}
}

impl PartialEq for r64 {
	fn eq(&self, other: &r64) -> bool {
		self.is_nan() && other.is_nan()
		|| self.numer() == 0 && other.numer() == 0
		|| self.normalize().0 == other.normalize().0
	}
}

impl PartialOrd for r64 {
	fn partial_cmp(&self, other: &r64) -> Option<Ordering> {
		// both are nan or both are zero
		if self.is_nan() && other.is_nan()
		|| self.numer() == 0 && other.numer() == 0 {
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

impl Mul for r64 {
	type Output = r64;
	
	fn mul(self, other: r64) -> r64 {
		let result = self.checked_mul(other);
		
		if cfg!(feature = "quiet-nan") {
			result.unwrap_or(r64::NAN)
		} else {
			result.expect("attempt to multiply with overflow")
		}
	}
}

impl Div for r64 {
	type Output = r64;

	fn div(self, other: r64) -> r64 {
		let result = self.checked_div(other);
		
		if cfg!(feature = "quiet-nan") {
			result.unwrap_or(r64::NAN)
		} else {
			result.expect("attempt to divide with overflow")
		}
	}
}

impl Add for r64 {
	type Output = r64;
	
	fn add(self, other: r64) -> r64 {
		let result = self.checked_add(other);
		
		if cfg!(feature = "quiet-nan") {
			result.unwrap_or(r64::NAN)
		} else {
			result.expect("attempt to add with overflow")
		}
	}
}

impl Sub for r64 {
	type Output = r64;

	fn sub(self, other: r64) -> r64 {
		let result = self.checked_sub(other);
		
		if cfg!(feature = "quiet-nan") {
			result.unwrap_or(r64::NAN)
		} else {
			result.expect("attempt to subtract with overflow")
		}
	}
}

#[doc(hidden)]
impl Rem for r64 {
	type Output = r64;
	
	fn rem(self, other: r64) -> r64 {
		todo!()
	}
}

#[cfg(test)]
mod tests {
	#[cfg(feature = "bench")]
	extern crate test;
	
	use super::*;
	
	#[test]
	fn normalize() {
		assert_eq!(r64::new(4, 2).normalize(), r64::new(2, 1));
		assert_eq!(r64::new(-4, 2).normalize(), r64::new(-2, 1));
	}
	
	#[test]
	fn neg() {
		assert_eq!(-r64(0), r64(0));
		assert_eq!(-r64(1), r64::new(-1, 1));
		assert_eq!(-r64::new(-1, 1), r64(1));
	}
	
	#[test]
	fn signum() {
		assert_eq!(r64(0).signum(), r64(0));
		assert_eq!(r64(1).signum(), r64(1));
		assert_eq!(r64(2).signum(), r64(1));
		assert_eq!(r64::new(-1, 1).signum(), r64::new(-1, 1));
		assert_eq!(r64::new(-2, 1).signum(), r64::new(-1, 1));
	}

	#[test]
	fn pow() {
		assert_eq!(r64(0).pow(0),   r64(1));
		assert_eq!(r64::NAN.pow(0), r64(1));
		assert_eq!(r64(1).pow(1),   r64(1));
		
		assert_eq!(r64(3).pow(2),           r64(9));
		assert_eq!(r64(3).pow(-2),          r64::new(1, 9));
		assert_eq!(r64::new(-3, 1).pow(2),  r64(9));
		assert_eq!(r64::new(-3, 1).pow(-2), r64::new(1, 9));
		
		assert_eq!(r64(2).pow(3),          r64(8));
		assert_eq!(r64(2).pow(-3),         r64::new(1, 8));
		assert_eq!(r64::new(1, 2).pow(3),  r64::new(1, 8));
		assert_eq!(r64::new(1, 2).pow(-3), r64(8));
		
		assert_eq!(r64::new(-2, 1).pow(3),  r64::new(-8, 1));
		assert_eq!(r64::new(-2, 1).pow(-3), r64::new(-1, 8));
		assert_eq!(r64::new(-1, 2).pow(3),  r64::new(-1, 8));
		assert_eq!(r64::new(-1, 2).pow(-3), r64::new(-8, 1));
	}

	#[test]
	fn checked_pow() {
		assert_eq!(r64(3).checked_pow(60), None);
	}

	#[test]
	fn checked_sqrt() {
		assert_eq!(r64(0).checked_sqrt(), Some(r64(0)));
		assert_eq!(r64(1).checked_sqrt(), Some(r64(1)));
		assert_eq!(r64(2).checked_sqrt(), None);
		assert_eq!(r64(4).checked_sqrt(), Some(r64(2)));
	}

	#[test]
	fn floor() {
		assert_eq!(r64::new(3, 2).floor(),  r64(1));
		assert_eq!(r64::new(2, 1).floor(),  r64(2));
		assert_eq!(r64::new(-3, 2).floor(), r64::from(-2_i8));
		assert_eq!(r64::new(-2, 1).floor(), r64::from(-2_i8));
	}

	#[test]
	fn ceil() {
		assert_eq!(r64::new(3, 2).ceil(),  r64(2));
		assert_eq!(r64::new(2, 1).ceil(),  r64(2));
		assert_eq!(r64::new(-3, 2).ceil(), r64::from(-1_i8));
		assert_eq!(r64::new(-2, 1).ceil(), r64::from(-2_i8));
	}

	#[test]
	fn round() {
		assert_eq!(r64(1).round(),             r64(1));
		assert_eq!((-r64(1)).round(),          r64::from(-1_i8));
		assert_eq!((r64(3) / r64(2)).round(),  r64(2));
		assert_eq!((-r64(3) / r64(2)).round(), r64::from(-2_i8));
	}
	
	#[test]
	fn fract() {
		assert_eq!(r64(5).fract(),          r64(0));
		assert_eq!(r64::new(3, 2).fract(),  r64::new(1, 2));
		assert_eq!(r64::new(-3, 2).fract(), r64::new(-1, 2));
	}
	
	#[test]
	fn trunc() {
		assert_eq!(r64(5).trunc(),          r64(5));
		assert_eq!(r64::new(1, 2).trunc(),  r64(0));
		assert_eq!(r64::new(-1, 2).trunc(), r64(0));
		assert_eq!(r64::new(3, 2).trunc(),  r64(1));
		assert_eq!(r64::new(-3, 2).trunc(), r64::from(-1 as i8));
	}
	
	#[test]
	fn recip() {
		assert_eq!(r64(5).recip(),         r64::new(1, 5));
		assert_eq!(r64::new(5, 2).recip(), r64::new(2, 5));
		assert_eq!(r64(1).recip(),         r64(1));
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
		
		assert_eq!(-r64(1) * r64(1), -r64(1));
		assert_eq!(r64(1) * -r64(1), -r64(1));
		assert_eq!(-r64(1) * -r64(1), r64(1));
		
		assert_eq!(r64(1) * r64(2), r64(2));
		assert_eq!(r64(2) * r64(2), r64(4));
		
		assert_eq!(
			r64::new(1, 2) * r64::new(1, 2),
			r64::new(1, 4)
		);
		assert_eq!(
			r64::new(-1, 2) * r64::new(1, 2),
			r64::new(-1, 4)
		);
		assert_eq!(
			r64::new(2, 3) * r64::new(2, 3),
			r64::new(4, 9)
		);
		assert_eq!(
			r64::new(3, 2) * r64::new(2, 3),
			r64(1)
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
		
		assert_eq!(r64(1) / r64(2), r64::new(1, 2));
		assert_eq!(r64(2) / r64(1), r64(2));
		assert_eq!(r64(2) / r64(2), r64(1));
	}

	#[test]
	fn rem() {
		assert_eq!(r64(5) % r64(2), r64(1));
		assert_eq!(r64(6) % r64(2), r64(0));
		assert_eq!(r64(8) % (r64(3) / r64(2)), r64(1) / r64(2));
		// Rust modulus gives same sign as dividend, and so do we
		assert_eq!(-r64(5) % r64(2), -r64(1));
		assert_eq!(r64(5) % -r64(2), r64(1));
		assert_eq!(-r64(5) % -r64(2), -r64(1));
	}
	
	#[test]
	fn add() {
		assert_eq!(r64(0) + r64(0), r64(0));
		assert_eq!(-r64(0) + r64(0), r64(0));
		
		assert_eq!(r64(1) + r64(1), r64(2));
		assert_eq!(r64(1) + -r64(1), r64(0));
		assert_eq!(-r64(1) + r64(1), r64(0));
		assert_eq!(-r64(1) + -r64(1), -r64(2));
		
		assert_eq!(r64(2) + r64(2), r64(4));
		assert_eq!(
			r64::new(1, 2) + r64::new(3, 4),
			r64::new(5, 4)
		);
		assert_eq!(
			r64::new(1, 2) + r64::new(-3, 4),
			r64::new(-1, 4)
		);
		assert_eq!(
			r64::new(-1, 2) + r64::new(3, 4),
			r64::new(1, 4)
		);
	}
	
	#[test] #[should_panic]
	fn add_invalid() {
		let _ = r64(1 << FRACTION_SIZE - 1) + r64(1 << FRACTION_SIZE - 1);
	}
	
	#[test]
	fn from_str() {
		assert_eq!("0".parse::<r64>().unwrap(), r64(0));
		assert_eq!("1".parse::<r64>().unwrap(), r64(1));
		assert_eq!("+1".parse::<r64>().unwrap(), r64(1));
		assert_eq!("-1".parse::<r64>().unwrap(), r64::from(-1 as i8));
		assert_eq!("1/1".parse::<r64>().unwrap(), r64(1));
	}
	
	#[test] #[should_panic]
	fn from_str_fail() {
		"1/-1".parse::<r32>().unwrap();
		"/1".parse::<r32>().unwrap();
		"1/".parse::<r32>().unwrap();
		"1/0".parse::<r32>().unwrap();
	}
	
	#[test]
	fn from_f32() {
		//assert_eq!(r64::from(std::f32::consts::E), r64(2850325) / r64(1048576));
		//assert_eq!(r64::from(std::f32::consts::TAU), r64(13176795) / r64(2097152));
	}
	
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
	
	#[test]
	fn debug() {
		assert_eq!(format!("{:?}", r64::new(-0, 1)), "-0/1");
		assert_eq!(format!("{:?}", r64::NAN), "NaN");
	}
	
	#[test]
	fn display() {
		assert_eq!(format!("{}", r64::new(0, 1)), "0");
		assert_eq!(format!("{}", r64::NAN), "NaN");
		assert_eq!(format!("{}", r64::new(-3, 2)), "-3/2");
	}
}

