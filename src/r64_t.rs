use core::fmt;
use core::cmp::Ordering;
use core::ops::*;
use core::str::FromStr;
use core::convert::TryFrom;

use gcd::Gcd;

use super::{ParseRatioErr, TryFromRatioError, r32};

/// The 64-bit floating bar type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, Default)]
pub struct r64(u64);

const DSIZE_SIZE: u32 = 6;
const FRACTION_SIZE: u64 = 58;

const FRACTION_FIELD: u64 = (1 << FRACTION_SIZE) - 1;


impl r64 {
	/// The highest value that can be represented by this rational type.
	pub const MAX: r64 = r64(FRACTION_FIELD >> 1);
	
	/// The lowest value that can be represented by this rational type.
	pub const MIN: r64 = r64(1 << (FRACTION_SIZE - 1));
	
	/// The smallest positive value that can be represented by this rational
	/// type.
	pub const MIN_POSITIVE: r64 = r64((FRACTION_SIZE - 1) << FRACTION_SIZE | FRACTION_FIELD);
	
	/// Not a Number (NaN).
	pub const NAN: r64 = r64(u64::MAX);
	
	// PRIVATE methods
	
	#[inline]
	fn denom_size(self) -> u64 {
		self.0 >> FRACTION_SIZE
	}
	
	#[inline]
	fn denom_mask(self) -> u64 {
		(1 << self.denom_size()) - 1
	}
	
	#[inline]
	fn numer_mask(self) -> u64 {
		FRACTION_FIELD & !self.denom_mask()
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
	
	// PUBLIC API

	/// Returns the numerator of this rational number. `self` cannot be NaN.
	#[inline]
	pub(crate) fn numer(self) -> i64 {
		// apparently this does sign-extension
		(self.0 as i64)
		.wrapping_shl(DSIZE_SIZE)
		.wrapping_shr(DSIZE_SIZE + (self.denom_size() as u32))
	}
	
	/// Returns the denominator of this rational number. `self` cannot be NaN.
	#[inline]
	pub(crate) fn denom(self) -> u64 {
		1 << self.denom_size() | (self.0 & self.denom_mask())
	}
	
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
	pub const fn new(numer: i64, denom: u64) -> Option<r64> {
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
	
	/// Creates a rational number if the given values can be reduced to fit in
	/// the fraction field.
	/// 
	/// This will run the GCD algorithm and remove common factors, if any. If
	/// the result does not fit in the fraction field, this returns `None`.
	pub fn new_reduced(mut numer: i64, mut denom: u64) -> Option<r64> {
		let gcd = numer.unsigned_abs().gcd(denom);
		numer /= gcd as i64;
		denom /= gcd;
		
		r64::new(numer, denom)
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
	pub fn trunc(self) -> r64 {
		if self.is_nan() { return self }
		
		let numer = self.numer() / (self.denom() as i64);
		// the `& FRACTION_FIELD` is for negative results.
		r64((numer as u64) & FRACTION_FIELD)
	}
	
	/// Returns the fractional part of a number, or NaN if `self` is NaN.
	#[inline]
	pub fn fract(self) -> r64 {
		if self.is_nan() { return self }
		
		let numer = (self.numer() % (self.denom() as i64)) as u64;
		// we can do this because all of self's bits will stay the same, apart
		// from the numerator.
		r64(
			self.0 & !self.numer_mask()
			| (numer << self.denom_size()) & FRACTION_FIELD
		)
	}
	
	/// Returns the largest integer less than or equal to a number.
	#[inline]
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
	#[inline]
	pub fn ceil(self) -> r64 {
		if self.is_positive() {
			// if self is a whole number,
			if self.numer() % (self.denom() as i64) == 0 {
				self
			} else {
				self.trunc() + r64(1)
			}
		} else {
			self.trunc()
		}
	}
	
	/// Returns the nearest integer to a number. Round half-way cases away from
	/// zero.
	#[inline]
	pub fn round(self) -> r64 {
		if self.is_negative() {
			unsafe { self - r64::new_unchecked(1, 2) }
		} else if self.is_positive() {
			unsafe { self + r64::new_unchecked(1, 2) }
		} else {
			self
		}
		.trunc()
	}
	
	/// Computes the absolute value of `self`.
	#[inline]
	pub fn abs(self) -> r64 {
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
	pub fn signum(self) -> r64 {
		if self.is_nan() {
			self
		} else if self.is_negative() {
			unsafe { r64::new_unchecked(-1, 1) }
		} else if self.is_positive() {
			r64(1)
		} else {
			r64(0)
		}
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`.
	/// 
	/// # Panics
	/// 
	/// Panics when the numerator is zero.
	#[inline]
	pub fn recip(self) -> r64 {
		self.checked_recip().expect("attempt to divide by zero")
	}
	
	/// Cancels out common factors between the numerator and the denominator.
	#[inline]
	pub fn normalize(self) -> r64 {
		if self.is_nan() { return self }
		
		let n = self.numer();
		let d = self.denom();
		
		// cancel out common factors by dividing numerator and denominator by
		// their greatest common divisor.
		let gcd = n.unsigned_abs().gcd(d);
		
		// SAFETY: an integer will always be smaller when divided by another
		// integer, and thus will always fit.
		unsafe {
			r64::new_unchecked(n / (gcd as i64), d / gcd)
		}
	}
	
	/// Raises self to the power of `exp`.
	#[inline]
	pub fn pow(self, exp: i32) -> r64 {
		self.checked_pow(exp).expect("attempt to multiply with overflow")
	}
	
	/// Returns the maximum of the two numbers.
	/// 
	/// If one of the arguments is `NaN`, then the other argument is returned.
	pub fn max(self, other: r64) -> r64 {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true)   => r64::NAN,
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
	pub fn min(self, other: r64) -> r64 {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true)   => r64::NAN,
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
	pub fn checked_neg(self) -> Option<r64> {
		if self.is_nan() { return Some(self) }
		// yes, this is the simplest and quickest way.
		r64::new(-self.numer(), self.denom())
	}
	
	/// Checked absolute value. Computes `self.abs()`, returning `None` if the
	/// numerator would overflow.
	#[inline]
	pub fn checked_abs(self) -> Option<r64> {
		if self.is_negative() {
			self.checked_neg()
		} else {
			Some(self)
		}
	}
	
	/// Checked reciprocal. Computes `1/self`, returning `None` if the
	/// numerator is zero.
	#[inline]
	pub fn checked_recip(self) -> Option<r64> {
		if self.is_nan() {
			Some(self)
		} else if self.numer() == 0 {
			None
		} else {
			let mut denom = self.denom() as i64;
			if self.is_negative() { denom = -denom }
			r64::new(denom, self.numer().unsigned_abs())
		}
	}
	
	/// Checked rational addition. Computes `self + rhs`, returning `None` if
	/// overflow occurred.
	pub fn checked_add(self, rhs: r64) -> Option<r64> {
		if self.is_nan() || rhs.is_nan() {
			return Some(r64::NAN);
		}
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
	
	/// Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
	/// overflow occurred.
	#[inline]
	pub fn checked_pow(self, exp: i32) -> Option<r64> {
		if exp == 0 { return Some(r64(1)) }
		if self.is_nan() { return Some(r64::NAN) }
		
		let exp_is_neg = exp < 0;
		let exp = exp.unsigned_abs();
		
		// TODO run gcd early to reduce multiple factors later?
		let num = self.numer().checked_pow(exp)?;
		let den = self.denom().checked_pow(exp)?;
		
		if exp_is_neg {
			r64::new(num, den)?.checked_recip()
		} else {
			r64::new(num, den)
		}
	}
	
	/// Checked subtraction. Computes `self - rhs`, returning `None` if
	/// overflow occurred.
	#[inline]
	pub fn checked_sub(self, rhs: r64) -> Option<r64> {
		self.checked_add(rhs.checked_neg()?)
	}
	
	/// Checked rational division. Computes `self / rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[inline]
	pub fn checked_div(self, rhs: r64) -> Option<r64> {
		self.checked_mul(rhs.checked_recip()?)
	}
	
	/// Checked rational remainder. Computes `self % rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[inline]
	pub fn checked_rem(self, rhs: r64) -> Option<r64> {
		let div = self.checked_div(rhs)?;
		div.checked_sub(div.floor())?.checked_mul(rhs)
	}
	
	/// Raw transmutation to `u64`.
	/// 
	/// Useful if you need access to the payload bits of a `NAN` value.
	#[inline]
	pub fn to_bits(self) -> u64 { self.0 }
	
	/// Raw transmutation from `u64`.
	#[inline]
	pub fn from_bits(bits: u64) -> r64 { r64(bits) }
}

crate::impl_ratio_traits! { r64 u64 i64 NonZeroU64 }

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

impl TryFrom<r64> for u64 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r64) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 && norm.numer() >= 0 {
			Ok(norm.numer() as u64)
		} else {
			Err(TryFromRatioError)
		}
	}
}

impl TryFrom<r64> for i64 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r64) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 {
			Ok(norm.numer())
		} else {
			Err(TryFromRatioError)
		}
	}
}

impl TryFrom<r64> for u128 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r64) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 && norm.numer() >= 0 {
			Ok(norm.numer() as u128)
		} else {
			Err(TryFromRatioError)
		}
	}
}

impl TryFrom<r64> for i128 {
	type Error = TryFromRatioError;
	
	#[inline]
	fn try_from(value: r64) -> Result<Self, Self::Error> {
		let norm = value.normalize();
		if norm.denom_size() == 0 {
			Ok(norm.numer() as i128)
		} else {
			Err(TryFromRatioError)
		}
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

crate::impl_ratio_tests!(r64);

