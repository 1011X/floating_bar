use core::fmt;
use core::cmp::Ordering;
use core::convert::TryFrom;
use core::ops::*;
use core::str::FromStr;

use gcd::Gcd;

use super::{ParseRatioErr, TryFromRatioError};

/// The 32-bit floating bar type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, Default)]
pub struct r32(u32);

const DSIZE_SIZE: u32 = 5;
const FRACTION_SIZE: u32 = 27;

const FRACTION_FIELD: u32 = (1 << FRACTION_SIZE) - 1;


impl r32 {
	/// The highest value that can be represented by this rational type.
	pub const MAX: r32 = r32((1 << (FRACTION_SIZE - 1)) - 1);
	
	/// The lowest value that can be represented by this rational type.
	pub const MIN: r32 = r32(1 << (FRACTION_SIZE - 1));
	
	/// The smallest positive value that can be represented by this rational
	/// type.
	pub const MIN_POSITIVE: r32 = r32(FRACTION_SIZE << FRACTION_SIZE | FRACTION_FIELD);
	
	/// Not a Number (NaN).
	pub const NAN: r32 = r32(u32::MAX);
	
	// PRIVATE API
	
	#[inline]
	fn denom_size(self) -> u32 {
		self.0 >> FRACTION_SIZE
	}
	
	#[inline]
	fn denom_mask(self) -> u32 {
		(1 << self.denom_size()) - 1
	}
	
	#[inline]
	fn numer_mask(self) -> u32 {
		FRACTION_FIELD & !self.denom_mask()
	}
	
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

	/// Returns the numerator of this rational number. `self` cannot be NaN.
	#[inline]
	pub(crate) fn numer(self) -> i32 {
		// apparently this does sign-extension
		(self.0 as i32)
		.wrapping_shl(DSIZE_SIZE)
		.wrapping_shr(DSIZE_SIZE + (self.denom_size() as u32))
	}
	
	/// Returns the denominator of this rational number. `self` cannot be NaN.
	#[inline]
	pub(crate) fn denom(self) -> u32 {
		1 << self.denom_size() | (self.0 & self.denom_mask())
	}
	
	/// Creates a rational number without checking the values.
	/// 
	/// # Safety
	/// 
	/// The values must fit in the fraction field.
	#[inline]
	pub const unsafe fn new_unchecked(numer: i32, denom: u32) -> r32 {
		let denom_size = 32 - denom.leading_zeros() - 1;
		let denom_mask = (1 << denom_size as u32) - 1;
		let numer_mask = FRACTION_FIELD & !denom_mask;
		
		r32(
			(denom_size as u32) << FRACTION_SIZE |
			((numer << denom_size) as u32) & numer_mask |
			denom & denom_mask
		)
	}
	
	/// Creates a rational number if the given values both fit in the fraction
	/// field.
	pub const fn new(numer: i32, denom: u32) -> Option<r32> {
		let denom_size = 32 - denom.leading_zeros() - 1;
		let numer_size = if numer >= 0 {
			32 - numer.leading_zeros() + 1
		} else {
			32 - numer.leading_ones() + 1
		};
		
		if numer_size + denom_size > FRACTION_SIZE as u32 {
			return None;
		}
		
		// SAFETY: we just checked if the values fit.
		unsafe {
			Some(r32::new_unchecked(numer, denom))
		}
	}
	
	/// Creates a rational number if the given values can be reduced to fit in
	/// the fraction field.
	/// 
	/// This will run the GCD algorithm and remove common factors, if any. If
	/// the result does not fit in the fraction field, this returns `None`.
	pub fn new_reduced(mut numer: i32, mut denom: u32) -> Option<r32> {
		let gcd = numer.unsigned_abs().gcd(denom);
		numer /= gcd as i32;
		denom /= gcd;
		
		r32::new(numer, denom)
	}
	
	/// Returns `true` if this value is `NAN` and `false` otherwise.
	#[inline]
	pub fn is_nan(self) -> bool {
		self.denom_size() >= FRACTION_SIZE
	}

	/// Returns `true` if `self` is positive and `false` if the number is zero,
	/// negative, or `NAN`.
	#[inline]
	pub fn is_positive(self) -> bool {
		!self.is_nan() && self.numer().is_positive()
	}

	/// Returns `true` if `self` is negative and `false` if the number is zero,
	/// positive, or `NAN`.
	#[inline]
	pub fn is_negative(self) -> bool {
		!self.is_nan() && self.numer().is_negative()
	}
	
	/// Returns the integer part of a number, or NaN if `self` is NaN.
	#[inline]
	pub fn trunc(self) -> r32 {
		if self.is_nan() { return self }
		
		let numer = self.numer() / (self.denom() as i32);
		// the `& FRACTION_FIELD` is for negative results.
		r32((numer as u32) & FRACTION_FIELD)
	}
	
	/// Returns the fractional part of a number, or NaN if `self` is NaN.
	#[inline]
	pub fn fract(self) -> r32 {
		if self.is_nan() { return self }
		
		let numer = (self.numer() % (self.denom() as i32)) as u32;
		// we can do this because all of self's bits will stay the same, apart
		// from the numerator.
		r32(
			self.0 & !self.numer_mask()
			| (numer << self.denom_size()) & FRACTION_FIELD
		)
	}
	
	/// Returns the largest integer less than or equal to a number.
	#[inline]
	pub fn floor(self) -> r32 {
		if self.is_negative() {
			// if self is a whole number,
			if self.numer() % (self.denom() as i32) == 0 {
				self
			} else {
				self.trunc() - r32(1)
			}
		} else {
			self.trunc()
		}
	}
	
	/// Returns the smallest integer greater than or equal to a number.
	#[inline]
	pub fn ceil(self) -> r32 {
		if self.is_positive() {
			// if self is a whole number,
			if self.numer() % (self.denom() as i32) == 0 {
				self
			} else {
				self.trunc() + r32(1)
			}
		} else {
			self.trunc()
		}
	}
	
	/// Returns the nearest integer to a number. Round half-way cases away from
	/// zero.
	#[inline]
	pub fn round(self) -> r32 {
		if self.is_negative() {
			unsafe { self - r32::new_unchecked(1, 2) }
		} else if self.is_positive() {
			unsafe { self + r32::new_unchecked(1, 2) }
		} else {
			self
		}
		.trunc()
	}
	
	/// Computes the absolute value of `self`.
	#[inline]
	pub fn abs(self) -> r32 {
		if self.is_negative() {
			-self
		} else {
			self
		}
	}
	
	/// Returns a number that represents the sign of `self`.
	/// 
	/// * `1` if the number is positive
	/// * `-1` if the number is negative
	/// * `0` if the number is `0`
	/// * `NAN` if the number is `NAN`.
	#[inline]
	pub fn signum(self) -> r32 {
		if self.is_nan() {
			self
		} else if self.is_negative() {
			unsafe { r32::new_unchecked(-1, 1) }
		} else if self.is_positive() {
			r32(1)
		} else {
			r32(0)
		}
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`.
	/// 
	/// # Panics
	/// 
	/// Panics when the numerator is zero.
	#[inline]
	pub fn recip(self) -> r32 {
		self.checked_recip().expect("attempt to divide by zero")
	}
	
	/// Cancels out common factors between the numerator and the denominator.
	#[inline]
	pub fn normalize(self) -> r32 {
		if self.is_nan() { return self }
		
		let n = self.numer();
		let d = self.denom();
		
		// cancel out common factors by dividing numerator and denominator by
		// their greatest common divisor.
		let gcd = n.unsigned_abs().gcd(d);
		
		// SAFETY: an integer will always be smaller when divided by another
		// integer, and thus will always fit.
		unsafe {
			r32::new_unchecked(n / (gcd as i32), d / gcd)
		}
	}
	
	/// Raises self to the power of `exp`.
	#[inline]
	pub fn pow(self, exp: i32) -> r32 {
		self.checked_pow(exp).expect("attempt to multiply with overflow")
	}
	
	/// Returns the maximum of the two numbers.
	/// 
	/// If one of the arguments is `NaN`, then the other argument is returned.
	pub fn max(self, other: r32) -> r32 {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true)   => r32::NAN,
			(true, false)  => other,
			(false, true)  => self,
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
	pub fn min(self, other: r32) -> r32 {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true)   => r32::NAN,
			(true, false)  => other,
			(false, true)  => self,
			(false, false) => match self.partial_cmp(&other).unwrap() {
				Ordering::Greater => other,
				// return self by default
				_ => self
			}
		}
	}
	
	/// Checked rational negation. Computes `-self`, returning `None` if the
	/// numerator would overflow.
	#[inline]
	pub fn checked_neg(self) -> Option<r32> {
		if self.is_nan() { return Some(self) }
		// yes, this is the simplest and quickest way.
		r32::new(-self.numer(), self.denom())
	}
	
	/// Checked absolute value. Computes `self.abs()`, returning `None` if the
	/// numerator would overflow.
	#[inline]
	pub fn checked_abs(self) -> Option<r32> {
		if self.is_negative() {
			self.checked_neg()
		} else {
			Some(self)
		}
	}
	
	/// Checked reciprocal. Computes `1/self`, returning `None` if the
	/// numerator is zero.
	#[inline]
	pub fn checked_recip(self) -> Option<r32> {
		if self.is_nan() {
			Some(self)
		} else if self.numer() == 0 {
			None
		} else {
			let mut denom = self.denom() as i32;
			if self.is_negative() { denom = -denom }
			r32::new(denom, self.numer().unsigned_abs())
		}
	}
	
	/*
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
	*/
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
		if self.is_nan() || rhs.is_nan() {
			return Some(r32::NAN);
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
	
	/// Checked subtraction. Computes `self - rhs`, returning `None` if
	/// overflow occurred.
	#[inline]
	pub fn checked_sub(self, rhs: r32) -> Option<r32> {
		self.checked_add(rhs.checked_neg()?)
	}
	
	/// Checked rational division. Computes `self / rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[inline]
	pub fn checked_div(self, rhs: r32) -> Option<r32> {
		self.checked_mul(rhs.checked_recip()?)
	}
	
	/// Checked rational remainder. Computes `self % rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[inline]
	pub fn checked_rem(self, rhs: r32) -> Option<r32> {
		let div = self.checked_div(rhs)?;
		div.checked_sub(div.floor())?.checked_mul(rhs)
	}
	
	/// Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
	/// overflow occurred.
	#[inline]
	pub fn checked_pow(self, exp: i32) -> Option<r32> {
		if exp == 0 { return Some(r32(1)) }
		if self.is_nan() { return Some(r32::NAN) }
		
		let exp_is_neg = exp < 0;
		let exp = exp.unsigned_abs();
		
		// TODO run gcd early to reduce multiple factors later?
		let num = self.numer().checked_pow(exp)?;
		let den = self.denom().checked_pow(exp)?;
		
		if exp_is_neg {
			r32::new(num, den)?.checked_recip()
		} else {
			r32::new(num, den)
		}
	}
	
	/// Raw transmutation to `u32`.
	/// 
	/// Useful if you need access to the payload bits of a `NAN` value.
	#[inline]
	pub fn to_bits(self) -> u32 { self.0 }
	
	/// Raw transmutation from `u32`.
	#[inline]
	pub fn from_bits(bits: u32) -> r32 { r32(bits) }
}

crate::impl_ratio_traits! { r32 u32 i32 NonZeroU32 }

impl From<u16> for r32 {
	#[inline]
	fn from(v: u16) -> Self { r32(v as u32) }
}

impl From<i16> for r32 {
	#[inline]
	fn from(v: i16) -> Self {
		// SAFETY: all i16 values fits in r32.
		unsafe { r32::new_unchecked(v as i32, 1) }
	}
}

impl TryFrom<r32> for u32 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r32) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 && norm.numer() >= 0 {
			Ok(norm.numer() as u32)
		} else {
			Err(TryFromRatioError)
		}
	}
}

impl TryFrom<r32> for i32 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r32) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 {
			Ok(norm.numer())
		} else {
			Err(TryFromRatioError)
		}
	}
}

impl TryFrom<r32> for u64 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r32) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 && norm.numer() >= 0 {
			Ok(norm.numer() as u64)
		} else {
			Err(TryFromRatioError)
		}
	}
}

impl TryFrom<r32> for i64 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r32) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 {
			Ok(norm.numer() as i64)
		} else {
			Err(TryFromRatioError)
		}
	}
}

impl TryFrom<r32> for u128 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r32) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 && norm.numer() >= 0 {
			Ok(norm.numer() as u128)
		} else {
			Err(TryFromRatioError)
		}
	}
}

impl TryFrom<r32> for i128 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r32) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 {
			Ok(norm.numer() as i128)
		} else {
			Err(TryFromRatioError)
		}
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

crate::impl_ratio_tests!(r32);

