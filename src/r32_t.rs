#![allow(dead_code)]
#![allow(unused_variables)]

use core::fmt;
use core::cmp::Ordering;
use core::ops::*;
use core::str::FromStr;

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
	
	/// Returns the denominator of this rational number.
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
		r32(self.abs().0 | (sign as u32) << 31)
	}
	
	#[inline]
	fn set_fraction(self, numer: u32, denom: u32) -> r32 {
		r32::from_parts(self.is_sign_negative(), numer, denom)
	}
	
	// BEGIN related float stuff
	
	/// Returns the integer part of a number.
	#[inline]
	pub fn trunc(self) -> r32 {
		self.set_fraction(self.numer() / self.denom(), 1)
	}
	
	/// Returns the fractional part of a number.
	#[inline]
	pub fn fract(self) -> r32 {
		let d = self.denom();
		self.set_fraction(self.numer() % d, d)
	}
	
	/// Returns the largest integer less than or equal to a number.
	pub fn floor(self) -> r32 {
		if self.is_sign_negative() {
			// if self is a whole number,
			if self.numer() % self.denom() == 0 {
				self
			} else {
				self.trunc() - r32(1)
			}
		} else {
			self.trunc()
		}
	}
	
	/// Returns the smallest integer greater than or equal to a number.
	pub fn ceil(self) -> r32 {
		if self.is_sign_negative() {
			self.trunc()
		} else {
			// if self is a whole number,
			if self.numer() % self.denom() == 0 {
				self
			} else {
				self.trunc() + r32(1)
			}
		}
	}
	
	/// Returns the nearest integer to a number. Round half-way cases away from
	/// zero.
	pub fn round(self) -> r32 {
		if self.is_sign_negative() {
			self - r32(1) / r32(2)
		} else {
			self + r32(1) / r32(2)
		}
		.trunc()
	}
	
	/// Computes the absolute value of `self`. Returns NaN if the number is NaN.
	#[inline]
	pub fn abs(self) -> r32 {
		r32(self.0 & !SIGN_BIT)
	}
	
	/// Returns a number that represents the sign of `self`.
	/// 
	/// * `1` if the number is positive
	/// * `-1` if the number is negative
	/// * `0` if the number is `+0`, `-0`, or `NaN`
	pub fn signum(self) -> r32 {
		if self.numer() == 0 || self.is_nan() {
			r32(0)
		} else {
			r32(self.0 & SIGN_BIT | 1)
		}
	}
	
	/// Raises a number to an integer power.
	/// 
	/// # Panics
	/// 
	/// Panics on overflow.
	#[cfg(not(feature = "quiet-nan"))]
	#[inline]
	pub fn pow(self, exp: i32) -> r32 {
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
	/// **Warning**: This method can give a value that overflows easily. Use
	/// with caution, and discard as soon as you're done with it.
	fn sqrt(self) -> r32 {
		// TODO: If `self` is positive, this should approximate its square root
		// by calculating a repeated fraction for a fixed number of steps.
		let f: f32 = self.into();
		r32::from(f.sqrt())
	}
	
	/// Calculates the approximate cube root of the value.
	/// 
	/// **Warning**: This method can give a value that overflows easily. Use
	/// with caution, and discard as soon as you're done with it.
	fn cbrt(self) -> r32 {
		todo!()
	}
	
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
	fn is_normal(self) -> bool {
		self.numer() != 0
		&& self.denom_size() < FRACTION_SIZE
	}
	
	/// Returns `true` if `self` is positive and `false` if the number is zero,
	/// negative, or `NaN`.
	#[inline]
	pub fn is_positive(self) -> bool {
		!self.is_nan()
		&& self.numer() > 0
		&& self.is_sign_positive()
	}
	
	/// Returns `true` if `self` is negative and `false` if the number is zero,
	/// positive, or `NaN`.
	#[inline]
	pub fn is_negative(self) -> bool {
		!self.is_nan()
		&& self.numer() > 0
		&& self.is_sign_negative()
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`.
	/// 
	/// # Panics
	/// 
	/// Panics when trying to set a numerator of zero as the denominator.
	#[cfg(not(feature = "quiet-nan"))]
	#[inline]
	pub fn recip(self) -> r32 {
		self.checked_recip().expect("attempt to divide by zero")
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`.
	///
	/// If the numerator is zero, this will return `NaN`.
	#[cfg(feature = "quiet-nan")]
	#[inline]
	pub fn recip(self) -> r32 {
		self.checked_recip().unwrap_or(r32::NAN)
	}
	
	/// Returns the maximum of the two numbers.
	/// 
	/// If one of the arguments is `NaN`, then the other argument is returned.
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
	pub fn checked_sub(self, rhs: r32) -> Option<r32> {
		self.checked_add(-rhs)
	}
	
	/// Checked rational multiplication. Computes `self * rhs`, returning `None`
	/// if overflow occurred.
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
	#[inline]
	pub fn checked_div(self, rhs: r32) -> Option<r32> {
		self.checked_mul(rhs.checked_recip()?)
	}
	
	/// Checked rational remainder. Computes `self % rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[doc(hidden)]
	pub fn checked_rem(self, rhs: r32) -> Option<r32> {
		let div = self.checked_div(rhs)?;
		let diff = div.checked_sub(div.floor())?;
		// TODO do we really need to be consistent with rust's % ?
		// if not, we can maybe remove set_sign() below.
		Some(diff.checked_mul(rhs)?.set_sign(self.is_negative()))
	}
	
	/// Checked rational reciprocal. Computes `1 / self`, returning `None` if
	/// the numerator is zero.
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
	pub fn checked_sqrt(self) -> Option<r32> {
		if self.is_negative() {
			return None;
		}
		
		let n = self.numer().integer_sqrt();
		let d = self.denom().integer_sqrt();

		if self.numer() == n * n && self.denom() == d * d {
			Some(r32::new(n as i32, d))
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
	pub fn checked_pow(self, exp: i32) -> Option<r32> {
		let exp_is_neg = exp < 0;
		let exp_is_odd = (exp & 1) == 1;
		let exp = exp.checked_abs()? as u32;
		
		let sign = exp_is_odd && self.is_sign_negative();
		
		let mut num = self.numer().checked_pow(exp)?;
		let mut den = self.denom().checked_pow(exp)?;
		
		if exp_is_neg {
			core::mem::swap(&mut num, &mut den);
		}
		
		Some(r32::from_parts(sign, num, den))
	}
	
	/// Checked conversion to `i32`.
	///
	/// Returns the numeric value as an `i32` if the denominator is 1.
	/// Otherwise, returns `None`.
	#[inline]
	pub fn to_i32(self) -> Option<i32> {
		if self.denom_size() != 0 {
			return None;
		}
		
		Some(
			if self.is_sign_negative() {
				-(self.abs().0 as i32)
			} else {
				self.0 as i32
			}
		)
	}
	
	// BEGIN bitwise representation stuff
	
	/// Raw transmutation to `u32`.
	/// 
	/// Useful if you need access to the payload bits of a NaN value.
	#[inline]
	pub fn to_bits(self) -> u32 { self.0 }
	
	/// Raw transmutation from `u32`.
	#[inline]
	pub fn from_bits(bits: u32) -> r32 { r32(bits) }
	
	/// Returns `true` if `self` has a positive sign, including positive zero,
	/// and `NaN`s with positive sign bit.
	/// 
	/// If you'd prefer to know if the value is strictly positive, consider
	/// using [`r32::is_positive`] instead.
	#[inline]
	pub fn is_sign_positive(self) -> bool {
		(self.0 & SIGN_BIT) == 0
	}
	
	/// Returns `true` if `self` has a negative sign, including negative zero,
	/// and `NaN`s with negative sign bit.
	/// 
	/// If you'd prefer to know if the value is strictly negative, consider
	/// using [`r32::is_negative`] instead.
	#[inline]
	pub fn is_sign_negative(self) -> bool {
		!self.is_sign_positive()
	}
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
		if self.is_sign_negative() {
			f.write_str("-")?;
		}
		
		if self.is_nan() {
			f.write_str("NaN")
		} else {
			write!(f, "{}/{}", self.numer(), self.denom())
		}
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
	/// `Err(ParseRatioError)` if the string did not contain a valid rational
	/// number. Otherwise, `Ok(n)` where `n` is the floating-bar number
	/// represented by `src`.
	fn from_str(mut src: &str) -> Result<Self, Self::Err> {
		use core::num::NonZeroU32;
		
		if src.is_empty() {
			return Err(ParseRatioErr::Empty);
		}
		
		// check for sign
		let sign = src.starts_with('-');
		
		if sign { src = &src[1..]; }
		
		// special case NaN
		if src == "NaN" {
			return Ok(r32::NAN.set_sign(sign));
		}
		
		// lookahead to find dividing bar, if any
		let bar_pos = src.find('/');
		let numer_end = bar_pos.unwrap_or(src.len());
		
		// parse numerator
		let numerator = src[..numer_end]
			.parse::<u32>()
			.map_err(ParseRatioErr::Numerator)?;
		
		// parse optional denominator
		let denominator = bar_pos
			.map(|pos|
				src[pos+1..]
				.parse::<NonZeroU32>()
				.map_err(ParseRatioErr::Denominator)
			) // : Option<Result<u32, ParseRatioErr>>
			.transpose()?
			// : Option<u32>
			.map(NonZeroU32::get)
			.unwrap_or(1);
		
		// ensure parsed numbers fit in fraction field
		let frac_size = r32::get_frac_size(
			numerator as u64,
			denominator as u64
		);
		
		if frac_size > FRACTION_SIZE {
			return Err(ParseRatioErr::Overflow);
		}
		
		Ok(r32::from_parts(sign, numerator, denominator))
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
	/// Based on [John D. Cook's Best Rational Approximation post](https://www.johndcook.com/blog/2010/10/20/best-rational-approximation/)
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
		
		r32::from_parts(is_neg, result.0, result.1)
	}
}

// TODO document wrt precision
// TODO add TryFrom version
impl From<r32> for f32 {
	fn from(r: r32) -> f32 {
		let s = if r.is_negative() { -1.0 } else { 1.0 };
		s * r.numer() as f32 / r.denom() as f32
	}
}

impl From<r32> for f64 {
	fn from(r: r32) -> f64 {
		let s = if r.is_negative() { -1.0 } else { 1.0 };
		s * r.numer() as f64 / r.denom() as f64
	}
}

impl Neg for r32 {
	type Output = r32;
	
	fn neg(self) -> Self::Output { r32(self.0 ^ SIGN_BIT) }
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
		let result = self.checked_mul(other);
		
		if cfg!(feature = "quiet-nan") {
			result.unwrap_or(r32::NAN)
		} else {
			result.expect("attempt to multiply with overflow")
		}
	}
}

impl Div for r32 {
	type Output = r32;

	fn div(self, other: r32) -> r32 {
		let result = self.checked_div(other);
		
		if cfg!(feature = "quiet-nan") {
			result.unwrap_or(r32::NAN)
		} else {
			result.expect("attempt to divide with overflow")
		}
	}
}

impl Add for r32 {
	type Output = r32;
	
	fn add(self, other: r32) -> r32 {
		let result = self.checked_add(other);
		
		if cfg!(feature = "quiet-nan") {
			result.unwrap_or(r32::NAN)
		} else {
			result.expect("attempt to add with overflow")
		}
	}
}

impl Sub for r32 {
	type Output = r32;

	fn sub(self, other: r32) -> r32 {
		let result = self.checked_sub(other);
		
		if cfg!(feature = "quiet-nan") {
			result.unwrap_or(r32::NAN)
		} else {
			result.expect("attempt to subtract with overflow")
		}
	}
}

#[doc(hidden)]
impl Rem for r32 {
	type Output = r32;
	
	fn rem(self, other: r32) -> r32 {
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
		assert_eq!(r32(0).pow(0),   r32(1));
		assert_eq!(r32::NAN.pow(0), r32(1));
		assert_eq!(r32(1).pow(1),   r32(1));
		
		assert_eq!(r32(3).pow(2),           r32(9));
		assert_eq!(r32(3).pow(-2),          r32::new(1, 9));
		assert_eq!(r32::new(-3, 1).pow(2),  r32(9));
		assert_eq!(r32::new(-3, 1).pow(-2), r32::new(1, 9));
		
		assert_eq!(r32(2).pow(3),          r32(8));
		assert_eq!(r32(2).pow(-3),         r32::new(1, 8));
		assert_eq!(r32::new(1, 2).pow(3),  r32::new(1, 8));
		assert_eq!(r32::new(1, 2).pow(-3), r32(8));
		
		assert_eq!(r32::new(-2, 1).pow(3),  r32::new(-8, 1));
		assert_eq!(r32::new(-2, 1).pow(-3), r32::new(-1, 8));
		assert_eq!(r32::new(-1, 2).pow(3),  r32::new(-1, 8));
		assert_eq!(r32::new(-1, 2).pow(-3), r32::new(-8, 1));
	}

	#[test]
	fn checked_pow() {
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
		assert_eq!(r32(5).recip(), r32::from_parts(false, 1, 5));
		assert_eq!(r32::new(5, 2).recip(), r32::new(2, 5));
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
	
	#[test] #[should_panic]
	fn from_str_fail() {
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
		assert_eq!(r32::from(std::f32::consts::E), r32(2721) / r32(1001));
		assert_eq!(r32::from(std::f32::consts::TAU), r32(710) / r32(113));
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
