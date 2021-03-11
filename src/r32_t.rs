#![allow(dead_code)]
#![allow(unused_variables)]

use std::fmt;
use std::cmp::Ordering;
use std::ops::*;
use std::str::FromStr;

use gcd::Gcd;
use integer_sqrt::IntegerSquareRoot;

use super::ParseRatioErr;

/// The 32-bit floating bar type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, Default)]
pub struct r32(u32);

const FRACTION_SIZE: u32 = 26;

const SIGN_BIT: u32 = 0x8000_0000;
const SIZE_FIELD: u32 = SIGN_BIT - 1 << FRACTION_SIZE + 1 >> 1;
const FRACTION_FIELD: u32 = (1 << FRACTION_SIZE) - 1;

impl r32 {
	/// The largest value that can be represented by this rational type.
	pub const MAX: r32 = r32(FRACTION_FIELD);
	
	/// The smallest value that can be represented by this rational type.
	pub const MIN: r32 = r32(SIGN_BIT | FRACTION_FIELD);
	
	/// The smallest positive normal value that can be represented by this
	/// rational type.
	pub const MIN_POSITIVE: r32 = r32(FRACTION_SIZE << FRACTION_SIZE | FRACTION_FIELD);
	
	/// Not a Number (NaN).
	pub const NAN: r32 = r32(SIZE_FIELD);

	#[inline]
	fn from_parts(sign: bool, numer: u32, denom: u32) -> r32 {
		let denom_size = 32 - denom.leading_zeros() - 1;
		let denom_mask = (1 << denom_size) - 1;
		let numer_mask = denom_mask ^ FRACTION_FIELD;
		r32(
			if sign { SIGN_BIT } else { 0 } |
			denom_size << FRACTION_SIZE |
			numer << denom_size & numer_mask | denom & denom_mask
		)
	}
	
	#[inline]
	fn denom_size(self) -> u32 {
		(self.0 & SIZE_FIELD) >> FRACTION_SIZE
	}
	
	#[inline]
	fn get_frac_size(n: u64, d: u64) -> u32 {
		let dsize = 64 - d.leading_zeros() - 1;
		let nsize =
			if cfg!(feature = "denormals") && dsize == FRACTION_SIZE && n == 1 { 0 }
			else { 64 - n.leading_zeros() };
		
		nsize + dsize
	}
	
	/// Creates a rational number from a signed numerator and an unsigned
	/// denominator.
	#[inline]
	pub fn new(numer: i32, denom: u32) -> r32 {
		r32::from_parts(numer.is_negative(), numer.abs() as u32, denom)
	}
	
	/// Returns the numerator of this rational number.
	#[inline]
	pub fn numer(self) -> u32 {
		if cfg!(feature = "denormals") && self.denom_size() == FRACTION_SIZE {
			1
		} else {
			(self.0 & FRACTION_FIELD) >> self.denom_size()
		}
	}
	
	/// Returns the denominator value for this rational number.
	#[inline]
	pub fn denom(self) -> u32 {
		let denom_region = (1 << self.denom_size()) - 1;
		1 << self.denom_size() | self.0 & denom_region
	}
	
	/// Sets sign bit to the value given.
	/// 
	/// If `true`, sign bit is set. Otherwise, it's unset.
	#[inline]
	fn set_sign(self, sign: bool) -> r32 {
		r32(self.0 & !SIGN_BIT | (sign as u32) << 31)
	}
	
	#[inline]
	fn set_fraction(self, numer: u32, denom: u32) -> r32 {
		r32::from_parts(self.is_sign_negative(), numer, denom)
	}
	
	#[inline]
	fn is_sign_positive(self) -> bool {
		self.0 & SIGN_BIT == 0
	}
	
	#[inline]
	pub(crate) fn is_sign_negative(self) -> bool {
		!self.is_sign_positive()
	}
	
	// BEGIN related float stuff
	
	/// Returns the largest integer less than or equal to a number.
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn floor(self) -> r32 {
		if self.is_sign_negative() {
			// if self is a whole number,
			if self.numer() % self.denom() == 0 {
				self
			} else {
				self.set_fraction(self.numer() / self.denom() + 1, 1)
			}
		} else {
			self.trunc()
		}
	}
	
	/// Returns the smallest integer greater than or equal to a number.
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn ceil(self) -> r32 {
		if self.is_sign_negative() {
			self.trunc()
		} else {
			// if self is a whole number,
			if self.numer() % self.denom() == 0 {
				self
			} else {
				self.set_fraction(self.numer() / self.denom() + 1, 1)
			}
		}
	}
	
	/// Returns the nearest integer to a number. Round half-way cases away from
	/// zero.
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn round(self) -> r32 {
		if self.is_sign_negative() {
			(self - r32(1) / r32(2)).ceil()
		} else {
			(self + r32(1) / r32(2)).floor()
		}
	}
	
	/// Returns the integer part of a number.
	#[must_use = "method returns a new number and does not mutate the original value"]
	#[inline]
	pub fn trunc(self) -> r32 {
		self.set_fraction(self.numer() / self.denom(), 1)
	}
	
	/// Returns the fractional part of a number.
	#[must_use = "method returns a new number and does not mutate the original value"]
	#[inline]
	pub fn fract(self) -> r32 {
		let d = self.denom();
		self.set_fraction(self.numer() % d, d)
	}
	
	/// Computes the absolute value of `self`. Returns NaN if the number is NaN.
	#[must_use = "method returns a new number and does not mutate the original value"]
	#[inline]
	pub fn abs(self) -> r32 {
		r32(self.0 & !SIGN_BIT)
	}
	
	/// Returns a number that represents the sign of `self`.
	/// 
	/// * `1` if the number is positive
	/// * `-1` if the number is negative
	/// * `0` if the number is `+0`, `-0`, or `NaN`
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn signum(self) -> r32 {
		if self.numer() == 0 || self.is_nan() {
			r32(0)
		} else {
			r32(self.0 & SIGN_BIT | 1)
		}
	}
	
	/// Raises a number to an integer power.
	// TODO: check that the new values fit in the type.
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn pow(self, p: i32) -> r32 {
		let num = self.numer().pow(p.abs() as u32);
		let den = self.denom().pow(p.abs() as u32);

		// power is positive
		if p >= 0 {
			self.set_fraction(num, den)
		} else {
			// power is negative; switch numbers around
			self.set_fraction(den, num)
		}
	}
	
	/// Takes the square root of a number.
	/// 
	/// **Warning**: This method can give a value that overflows easily. Use
	/// with caution, and discard as soon as you're done with it.
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn sqrt(self) -> r32 {
		// TODO: If `self` is positive, this should approximate its square root
		// by calculating a repeated fraction for a fixed number of steps.
		let f: f32 = self.into();
		r32::from(f.sqrt())
	}
	
	/*
	TODO consider whether to actually add these.
	/// Takes the cube root of a number.
	/// 
	/// If `self` is positive and its numerator and denominator are perfect
	/// cubes, returns their cube root. Otherwise, returns `None`.
	#[inline]
	pub fn checked_cbrt(self) -> Option<r32> {
		unimplemented!()
	}
	*/
	
	/// Returns `true` if this value is `NaN` and `false` otherwise.
	#[inline]
	#[cfg(feature = "denormals")]
	pub fn is_nan(self) -> bool {
		self.denom_size() > FRACTION_SIZE
	}
	
	/// Returns `true` if this value is `NaN` and `false` otherwise.
	#[inline]
	#[cfg(not(feature = "denormals"))]
	pub fn is_nan(self) -> bool {
		self.denom_size() >= FRACTION_SIZE
	}
	
	/// Returns `true` if the number is neither zero, denormal, or `NaN`.
	#[inline]
	pub fn is_normal(self) -> bool {
		self.numer() != 0
		&& self.denom_size() < FRACTION_SIZE
	}
	
	/// Returns `true` if and only if `self` has a positive sign, including
	/// `+0.0` (but not NaNs with positive sign bit).
	#[inline]
	pub fn is_positive(self) -> bool {
		self.numer() != 0 && self.is_sign_positive()
	}
	
	/// Returns `true` if and only if self has a negative sign, including
	/// `-0.0` (but not NaNs with negative sign bit).
	#[inline]
	pub fn is_negative(self) -> bool {
		self.numer() != 0 && self.is_sign_negative()
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`.
	/// 
	/// # Panics
	/// 
	/// Panics when trying to set a numerator of zero as denominator.
	#[must_use = "method returns a new number and does not mutate the original value"]
	#[inline]
	pub fn recip(self) -> r32 {
		self.checked_recip().expect("attempt to divide by zero")
	}
	
	/// Returns the maximum of the two numbers.
	/// 
	/// If one of the arguments is `NaN`, then the other argument is returned.
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn max(self, other: r32) -> r32 {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true)   => r32::NAN,
			(true, false)  => self,
			(false, true)  => other,
			(false, false) => match self.partial_cmp(&other).unwrap() {
				Ordering::Less => other,
				// return self by default
				_ => self
			}
		}
	}
	
	/// Returns the minimum of the two numbers.
	/// 
	/// If one of the arguments is `NaN`, then the other argument is returned.
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn min(self, other: r32) -> r32 {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true)   => r32::NAN,
			(true, false)  => self,
			(false, true)  => other,
			(false, false) => match self.partial_cmp(&other).unwrap() {
				Ordering::Greater => other,
				// return self by default
				_ => self
			}
		}
	}
	
	/// Cancels out common factors between the numerator and the denominator.
	#[must_use = "method returns a new number and does not mutate the original value"]
	pub fn normalize(self) -> r32 {
		if self.is_nan() {
			return self;
		}
		
		if self.numer() == 0 {
			return r32(0);
		}
		
		let n = self.numer();
		let d = self.denom();
		
		// cancel out common factors by dividing numerator and denominator by
		// their greatest common divisor.
		let gcd = n.gcd(d);
		self.set_fraction(n / gcd, d / gcd)
	}
	
	// BEGIN related integer stuff
	
	/// Checked rational addition. Computes `self + rhs`, returning `None` if
	/// overflow occurred.
	#[must_use = "this returns the result of the operation, without modifying the original"]
	pub fn checked_add(self, rhs: r32) -> Option<r32> {
		// self = a/b, other = c/d
		
		let selfsign = (self.signum().0 as i32).signum() as i64;
		let othersign = (rhs.signum().0 as i32).signum() as i64;
		
		// TODO prove this won't panic/can't overflow.
		// num = ad + bc
		let num =
			(self.numer() as i64 * selfsign) * rhs.denom() as i64
			+ self.denom() as i64 * (rhs.numer() as i64 * othersign);
		// den = bd
		let mut den = self.denom() as u64 * rhs.denom() as u64;
		let s = num.is_negative();
		let mut num = num.abs() as u64;
		
		let mut size = r32::get_frac_size(num, den);
		
		if size > FRACTION_SIZE {
			let gcd = num.gcd(den);
			num /= gcd;
			den /= gcd;
			size = r32::get_frac_size(num, den);
		}
		
		if size <= FRACTION_SIZE {
			Some(r32::from_parts(s, num as u32, den as u32))
		} else {
			None
		}
	}
	
	/// Checked rational subtraction. Computes `self - rhs`, returning `None` if
	/// overflow occurred.
	#[must_use = "this returns the result of the operation, without modifying the original"]
	pub fn checked_sub(self, rhs: r32) -> Option<r32> {
		self.checked_add(-rhs)
	}
	
	/// Checked rational multiplication. Computes `self * rhs`, returning `None`
	/// if overflow occurred.
	#[must_use = "this returns the result of the operation, without modifying the original"]
	pub fn checked_mul(self, rhs: r32) -> Option<r32> {
		let s = self.is_sign_negative() != rhs.is_sign_negative();
		let mut n = self.numer() as u64 * rhs.numer() as u64;
		let mut d = self.denom() as u64 * rhs.denom() as u64;
		
		let mut size = r32::get_frac_size(n, d);
		
		if size > FRACTION_SIZE {
			let gcd = n.gcd(d);
			n /= gcd;
			d /= gcd;
			size = r32::get_frac_size(n, d);
		}
		
		if size <= FRACTION_SIZE {
			Some(r32::from_parts(s, n as u32, d as u32))
		} else {
			None
		}
	}
	
	/// Checked rational division. Computes `self / rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[must_use = "this returns the result of the operation, without modifying the original"]
	#[inline]
	pub fn checked_div(self, rhs: r32) -> Option<r32> {
		self.checked_mul(rhs.checked_recip()?)
	}
	
	/// Checked rational remainder. Computes `self % rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[must_use = "this returns the result of the operation, without modifying the original"]
	#[doc(hidden)]
	pub fn checked_rem(self, rhs: r32) -> Option<r32> {
		todo!()
	}
	
	/// Checked rational reciprocal. Computes `1 / self`, returning `None` if
	/// `self.numer() == 0`.
	#[must_use = "this returns the result of the operation, without modifying the original"]
	pub fn checked_recip(self) -> Option<r32> {
		if self.numer() == 0 {
			None
		} else {
			Some(self.set_fraction(self.denom(), self.numer()))
		}
		//assert!(self.denom_size() < FRACTION_SIZE, "subnormal overflow");
	}
	
	/// Takes the *checked* square root of a number.
	/// 
	/// If `self` is positive and both the numerator and denominator are perfect
	/// squares, this returns their square root. Otherwise, returns `None`.
	#[must_use = "this returns the result of the operation, without modifying the original"]
	pub fn checked_sqrt(self) -> Option<r32> {
		let nsqrt = self.numer().integer_sqrt();
		let dsqrt = self.denom().integer_sqrt();

		if self.numer() == nsqrt * nsqrt && self.denom() == dsqrt * dsqrt {
			Some(r32::from_parts(self.is_negative(), nsqrt, dsqrt))
		} else {
			None
		}
	}
	
	/// Raises a number to an integer power.
	// TODO: check that the new values fit in the type.
	#[must_use = "this returns the result of the operation, without modifying the original"]
	pub fn checked_pow(self, p: i32) -> Option<r32> {
		let num = self.numer().checked_pow(p.abs() as u32);
		let den = self.denom().checked_pow(p.abs() as u32);

		match (num, den) {
			(Some(num), Some(den)) => Some(
				// power is positive
				if p >= 0 {
					r32::from_parts(self.is_negative(), num, den)
				} else {
					// power is negative; switch numbers around
					r32::from_parts(self.is_negative(), den, num)
				}
			),
			_ => None
		}
	}
	
	/// Raw transmutation to `u32`.
	#[inline]
	pub fn to_bits(self) -> u32 { self.0 }
	
	/// Raw transmutation from `u32`.
	#[inline]
	pub fn from_bits(bits: u32) -> r32 { r32(bits) }
}

impl fmt::Display for r32 {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		if self.is_nan() {
			return f.write_str("NaN");
		}
		
		let norm = self.normalize();
		
		if norm.is_negative() {
			f.write_str("-")?;
		}
		
		norm.numer().fmt(f)?;
		
		if norm.denom_size() > 0 {
			write!(f, "/{}", norm.denom())?;
		}
		
		Ok(())
	}
}

impl fmt::Debug for r32 {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		if self.is_nan() {
			return f.write_str("NaN");
		}
		
		if self.is_sign_negative() {
			f.write_str("-")?;
		}
		
		write!(f, "{}/{}", self.numer(), self.denom())
	}
}

impl FromStr for r32 {
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
		if src.is_empty() {
			return Err(ParseRatioErr::Empty);
		}
		
		if src == "NaN" {
			return Ok(r32::NAN);
		}
		
		// if bar exists, parse as fraction
		if let Some(pos) = src.find('/') {
			// bar is at the end. invalid.
			if pos == src.len() - 1 {
				return Err(ParseRatioErr::Invalid);
			}
			
			let numerator: i32 = src[0..pos].parse()?;
			let denominator: u32 = src[pos+1..].parse()?;
			
			if denominator == 0 {
				return Err(ParseRatioErr::Invalid);
			}
			
			let denom_size = 32 - denominator.leading_zeros() - 1;
			
			// if denormal, return early
			#[cfg(feature = "denormals")]
			if numerator.abs() == 1 && denom_size == FRACTION_SIZE {
				let sign = numerator < 0;
				let denominator = denominator & FRACTION_FIELD;
				
				return Ok(r32::from_parts(sign, 1, denominator));
			}

			// ensure both fragments fit in the fraction field
			// TODO this uh, likely won't work if numerator is negative
			let frac_size = denom_size + (32 - numerator.leading_zeros());
			
			if frac_size > FRACTION_SIZE {
				return Err(ParseRatioErr::Overflow);
			}
			
			Ok(r32::new(numerator, denominator))
		} else {
			// otherwise, parse as integer.
			let numerator: i32 = src.parse()?;
			
			let mag = numerator.checked_abs()
				.ok_or(ParseRatioErr::Overflow)?;
			let frac_size = 32 - mag.leading_zeros();
			
			if frac_size > FRACTION_SIZE {
				return Err(ParseRatioErr::Overflow);
			}
			
			Ok(r32::new(numerator, 1))
		}
	}
}

impl From<u8> for r32 {
	#[inline]
	fn from(v: u8) -> Self { r32(v as u32) }
}

impl From<i8> for r32 {
	fn from(v: i8) -> Self {
		let n = if v == i8::MIN { 128 } else { v.abs() as u32 };
		r32::from_parts(v.is_negative(), n, 1)
	}
}

impl From<u16> for r32 {
	#[inline]
	fn from(v: u16) -> Self { r32(v as u32) }
}

impl From<i16> for r32 {
	fn from(v: i16) -> Self {
		let n = if v == i16::MIN { 32768 } else { v.abs() as u32 };
		r32::from_parts(v.is_negative(), n, 1)
	}
}

impl From<f32> for r32 {
	/// Based on: https://www.johndcook.com/blog/2010/10/20/best-rational-approximation/
	fn from(mut f: f32) -> Self {
		// why 13? bc it's fraction_size / 2
		// div by 2 is to have enough space for both numer and denom.
		// don't count implicit bit because then we can only represent 0 - 0.5
		// in a number that could be 0 - 1.
		const N: u32 = (1 << 13) - 1; // 2^13 - 1 = 8191
		//let is_lorge = f.abs() > 1.0;
		let is_neg = f < 0.0;
		
		if f.is_nan() || f.is_infinite() {
			return r32::NAN;
		}
		
		//if is_lorge { f = f.recip(); }
		if is_neg   { f = f.abs();   }
		
		let (mut a, mut b) = (0, 1); // lower
		let (mut c, mut d) = (1, 0); // upper
		let mut is_mediant = false;
		
		// while neither denoms are too big,
		while b <= N && d <= N {
			let mediant = (a + c) as f32 / (b + d) as f32;
			
			if f == mediant {
				is_mediant = true;
				break;
			} else if f > mediant {
				a = a + c;
				b = b + d;
			} else {
				c = a + c;
				d = b + d;
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
		
		// use reciprocal if original number wasn't between 0 and 1
		/*if is_lorge {
			return r32::from_parts(is_neg, result.1, result.0);
		}
		else {*/
			return r32::from_parts(is_neg, result.0, result.1);
		//}
	}
}

impl Into<f32> for r32 {
	fn into(self) -> f32 {
		let s = if self.is_negative() { -1.0 } else { 1.0 };
		s * self.numer() as f32 / self.denom() as f32
	}
}

impl Into<f64> for r32 {
	fn into(self) -> f64 {
		let s = if self.is_negative() { -1.0 } else { 1.0 };
		s * self.numer() as f64 / self.denom() as f64
	}
}

impl Neg for r32 {
	type Output = r32;
	
	fn neg(self) -> Self::Output {
		r32(self.0 ^ SIGN_BIT)
	}
}

impl PartialEq for r32 {
	fn eq(&self, other: &r32) -> bool {
		self.is_nan() && other.is_nan()
		|| self.numer() == 0 && other.numer() == 0
		|| self.normalize().0 == other.normalize().0
	}
}

impl PartialOrd for r32 {
	fn partial_cmp(&self, other: &r32) -> Option<Ordering> {
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
		self.is_sign_positive()
		.partial_cmp(&other.is_sign_positive())
		.map(|c| c.then(
			// compare numbers
			// a/b = c/d <=> ad = bc
			// when a, b, c, and d are all > 0
			(self.numer() as u64 * other.denom() as u64)
			.cmp(&(self.denom() as u64 * other.numer() as u64))
		))
	}
}

impl Mul for r32 {
	type Output = r32;
	
	fn mul(self, other: r32) -> r32 {
		self.checked_mul(other).expect("attempt to multiply with overflow")
	}
}

impl Div for r32 {
	type Output = r32;

	fn div(self, other: r32) -> r32 {
		self.checked_div(other).expect("attempt to divide with overflow")
	}
}

impl Add for r32 {
	type Output = r32;
	
	fn add(self, other: r32) -> r32 {
		self.checked_add(other).expect("attempt to add with overflow")
	}
}

impl Sub for r32 {
	type Output = r32;

	fn sub(self, other: r32) -> r32 {
		self.checked_sub(other).expect("attempt to subtract with overflow")
	}
}

impl Rem for r32 {
	type Output = r32;
	
	fn rem(self, other: r32) -> r32 {
		let div = self / other;
		// TODO do we really need to be consistent with rust's % ?
		// if not, we can maybe remove set_sign() below.
		((div - div.floor()) * other).set_sign(self.is_negative())
	}
}

#[cfg(test)]
mod tests {
	#[cfg(feature = "bench")]
	extern crate test;
	
	use super::*;
	
	#[test]
	fn normalize() {
		assert_eq!(r32::from_parts(false, 4, 2).normalize(), r32::from_parts(false, 2, 1));
		assert_eq!(r32::from_parts(true, 4, 2).normalize(), r32::from_parts(true, 2, 1));
	}
	
	#[test]
	fn neg() {
		assert_eq!((-r32(0)).0, SIGN_BIT);
		assert_eq!((-r32(SIGN_BIT)).0, 0);
		assert_eq!(-r32(1), r32::from_parts(true, 1, 1));
	}
	
	#[test]
	fn signum() {
		assert_eq!(r32(0).signum(), r32(0));
		assert_eq!(r32(SIGN_BIT).signum(), r32(0));
		assert_eq!(r32(1).signum(), r32(1));
		assert_eq!(r32(2).signum(), r32(1));
		assert_eq!(r32::from_parts(true, 1, 1).signum(), r32::from_parts(true, 1, 1));
		assert_eq!(r32::from_parts(true, 2, 1).signum(), r32::from_parts(true, 1, 1));
	}

	#[test]
	fn pow() {
		assert_eq!(r32(0).pow(0), r32(1));
		assert_eq!(r32(1).pow(1), r32(1));
		assert_eq!(r32(2).pow(3), r32(8));
		assert_eq!(r32(2).pow(-3), r32::from_str("1/8").unwrap());
	}

	#[test]
	fn checked_pow() {
		assert_eq!(r32(0).checked_pow(0), Some(r32(1)));
		assert_eq!(r32(1).checked_pow(1), Some(r32(1)));
		assert_eq!(r32(2).checked_pow(3), Some(r32(8)));
		assert_eq!(r32(2).checked_pow(-3), Some(r32::from_str("1/8").unwrap()));
		assert_eq!(r32(3).checked_pow(30), None);
	}

	#[test]
	fn checked_sqrt() {
		assert_eq!(r32(0).checked_sqrt(), Some(r32(0)));
		assert_eq!(r32(1).checked_sqrt(), Some(r32(1)));
		assert_eq!(r32(2).checked_sqrt(), None);
		assert_eq!(r32(4).checked_sqrt(), Some(r32(2)));
	}

	#[test]
	fn floor() {
		assert_eq!(r32(1).floor(), r32(1));
		assert_eq!((-r32(1)).floor(), r32::from(-1_i8));
		assert_eq!((r32(3) / r32(2)).floor(), r32(1));
		assert_eq!((-r32(3) / r32(2)).floor(), r32::from(-2_i8));
	}

	#[test]
	fn ceil() {
		assert_eq!(r32(1).ceil(), r32(1));
		assert_eq!((-r32(1)).ceil(), r32::from(-1_i8));
		assert_eq!((r32(3) / r32(2)).ceil(), r32(2));
		assert_eq!((-r32(3) / r32(2)).ceil(), r32::from(-1_i8));
	}

	#[test]
	fn round() {
		assert_eq!(r32(1).round(), r32(1));
		assert_eq!((-r32(1)).round(), r32::from(-1_i8));
		assert_eq!((r32(3) / r32(2)).round(), r32(2));
		assert_eq!((-r32(3) / r32(2)).round(), r32::from(-2_i8));
	}
	
	#[test]
	fn fract() {
		assert_eq!(r32(5).fract(), r32(0));
		assert_eq!(r32::from_parts(false, 3, 2).fract(), r32::from_parts(false, 1, 2));
		assert_eq!(r32::from_parts(true, 3, 2).fract(), r32::from_parts(true, 1, 2));
	}
	
	#[test]
	fn trunc() {
		assert_eq!(r32(5).trunc(), r32(5));
		assert_eq!(r32::from_parts(false, 1, 2).trunc(), r32(0));
		assert_eq!(r32::from_parts(true, 1, 2).trunc(), r32(0));
		assert_eq!(r32::from_parts(false, 3, 2).trunc(), r32(1));
		assert_eq!(r32::from_parts(true, 3, 2).trunc(), r32::from(-1 as i8));
	}
	
	#[test]
	fn recip() {
		//println!("{:b}", r32(5));
		assert_eq!(dbg![r32(5)].recip(), r32::from_parts(false, 1, 5));
		assert_eq!(r32::from_parts(false, 5, 2).recip(), r32::from_parts(false, 2, 5));
		assert_eq!(r32(1).recip(), r32(1));
	}
	
	#[test]
	fn cmp() {
		assert!(r32(0) == r32(0));
		assert!(r32(0) == -r32(0));
		
		assert!(r32(0) < r32(1));
		assert!(r32(2) < r32(3));
		assert!(r32(0) > -r32(1));
		assert!(r32(2) > -r32(3));
	}
	
	#[test]
	fn mul() {
		assert_eq!(r32(0) * r32(0), r32(0));
		
		assert_eq!(r32(0) * r32(1), r32(0));
		assert_eq!(r32(1) * r32(0), r32(0));
		assert_eq!(r32(1) * r32(1), r32(1));
		
		assert_eq!(-r32(1) * r32(1), -r32(1));
		assert_eq!(r32(1) * -r32(1), -r32(1));
		assert_eq!(-r32(1) * -r32(1), r32(1));
		
		assert_eq!(r32(1) * r32(2), r32(2));
		assert_eq!(r32(2) * r32(2), r32(4));
		
		assert_eq!(
			r32::from_parts(false, 1, 2) * r32::from_parts(false, 1, 2),
			r32::from_parts(false, 1, 4)
		);
		assert_eq!(
			r32::from_parts(true, 1, 2) * r32::from_parts(false, 1, 2),
			r32::from_parts(true, 1, 4)
		);
		assert_eq!(
			r32::from_parts(false, 2, 3) * r32::from_parts(false, 2, 3),
			r32::from_parts(false, 4, 9)
		);
		assert_eq!(
			r32::from_parts(false, 3, 2) * r32::from_parts(false, 2, 3),
			r32(1)
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
		
		assert_eq!(-r32(1) / r32(1), -r32(1));
		assert_eq!(r32(1) / -r32(1), -r32(1));
		assert_eq!(-r32(1) / -r32(1), r32(1));
		
		assert_eq!(r32(1) / r32(2), r32::from_parts(false, 1, 2));
		assert_eq!(r32(2) / r32(1), r32(2));
		assert_eq!(r32(2) / r32(2), r32(1));
	}

	#[test]
	fn rem() {
		assert_eq!(r32(5) % r32(2), r32(1));
		assert_eq!(r32(6) % r32(2), r32(0));
		assert_eq!(r32(8) % (r32(3) / r32(2)), r32(1) / r32(2));
		// Rust modulus gives same sign as dividend, and so do we
		assert_eq!(-r32(5) % r32(2), -r32(1));
		assert_eq!(r32(5) % -r32(2), r32(1));
		assert_eq!(-r32(5) % -r32(2), -r32(1));
	}
	
	#[test]
	fn add() {
		assert_eq!(r32(0) + r32(0), r32(0));
		assert_eq!(-r32(0) + r32(0), r32(0));
		
		assert_eq!(r32(1) + r32(1), r32(2));
		assert_eq!(r32(1) + -r32(1), r32(0));
		assert_eq!(-r32(1) + r32(1), r32(0));
		assert_eq!(-r32(1) + -r32(1), -r32(2));
		
		assert_eq!(r32(2) + r32(2), r32(4));
		assert_eq!(
			r32::from_parts(false, 1, 2) + r32::from_parts(false, 3, 4),
			r32::from_parts(false, 5, 4)
		);
		assert_eq!(
			r32::from_parts(false, 1, 2) + r32::from_parts(true, 3, 4),
			r32::from_parts(true, 1, 4)
		);
		assert_eq!(
			r32::from_parts(true, 1, 2) + r32::from_parts(false, 3, 4),
			r32::from_parts(false, 1, 4)
		);
	}
	
	#[test] #[should_panic]
	fn add_invalid() {
		let _ = r32(1 << FRACTION_SIZE - 1) + r32(1 << FRACTION_SIZE - 1);
	}
	
	#[test]
	fn from_str() {
		assert_eq!("0".parse::<r32>().unwrap(), r32(0));
		assert_eq!("1".parse::<r32>().unwrap(), r32(1));
		assert_eq!("+1".parse::<r32>().unwrap(), r32(1));
		assert_eq!("-1".parse::<r32>().unwrap(), r32::from(-1 as i8));
		assert_eq!("1/1".parse::<r32>().unwrap(), r32(1));
	}
	
	// TODO
	#[test] #[ignore]
	fn from_f32() {
		assert_eq!("0".parse::<r32>().unwrap(), r32(0));
		assert_eq!("1".parse::<r32>().unwrap(), r32(1));
		assert_eq!("+1".parse::<r32>().unwrap(), r32(1));
		assert_eq!("-1".parse::<r32>().unwrap(), r32::from(-1 as i8));
		assert_eq!("1/1".parse::<r32>().unwrap(), r32(1));
	}
	
	#[test]
	fn debug() {
		assert_eq!(format!("{:?}", r32::from_parts(true, 0, 1)), "-0/1");
		assert_eq!(format!("{:?}", r32::NAN), "NaN");
	}
	
	#[test]
	fn display() {
		assert_eq!(format!("{}", r32::from_parts(false, 0, 1)), "0");
		assert_eq!(format!("{}", r32::NAN), "NaN");
		assert_eq!(format!("{}", r32::from_parts(true, 3, 2)), "-3/2");
	}
}
