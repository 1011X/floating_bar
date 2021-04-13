/*!
This library provides the floating-bar type, which allows for efficient
representation of rational numbers without loss of precision. It is based on
[Inigo Quilez's blog post exploring the concept](http://www.iquilezles.org/www/articles/floatingbar/floatingbar.htm).

```rust
use floating_bar::r32;

let fullscreen = r32!(4 / 3);
let widescreen = r32!(16 / 9);

assert_eq!(fullscreen, r32!(800 / 600));
assert_eq!(widescreen, r32!(1280 / 720));
assert_eq!(widescreen, r32!(1920 / 1080));
```

## Structure

The floating-bar types follow a general structure:
* the **denominator-size field**: always log<sub>2</sub> of the type's total size, stored in the highest bits.
* the **fraction field**: stored in the remaining bits.

Here is a visual aid, where each character corresponds to one bit and the least significant bit is on the right:
```txt
d = denominator size field, f = fraction field

r32: dddddfffffffffffffffffffffffffff
r64: ddddddffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
```

The fraction field stores both the numerator and the denominator. The size of
the denominator is determined by the denominator-size field, which gives the position of the partition (the "fraction bar") from the right.

The numerator is stored as a two's complement signed integer on the left side of
the partition. The denominator is stored as an unsigned integer on the right
side. The denominator has an implicit 1 bit in front of its stored value,
similar to the implicit bit convention followed by floating-point. Thus, a size
field of zero has an implicit denominator value of 1.

## Value space

There are three distinct categories that a floating-bar number can fall into:
normal, reducible, and not-a-number (also known as NaNs).

**NaN** values are those with a denominator size greater than or equal to the
size of the entire fraction field. The library mostly ignores these values, and
only uses one particular value to provide a `NAN` constant. They can be used to
store payloads if desired using the `.to_bits()` and `from_bits()` methods.
Effort is put into not clobbering possible payload values, but no guarantees are
made.

**Reducible** values are those where the numerator and denominator share some
common factor that has not been canceled out, and thus take up more space than
their normalized form. Due to the performance cost of finding and canceling out
common factors, reducible values are only normalized when absolutely necessary,
such as when the result would otherwise overflow.

**Normal** values are those where the numerator and denominator don't share any
common factors, and could not be any smaller while still accurately representing
its value.

## Behavior

**Equality** is performed by the following rules:
1. If both numbers are NaN, they are equal.
2. If only one of the numbers is NaN, they are not equal.
3. Otherwise, both values are normalized and their raw representations are
   checked for equality.

**Comparison** is performed by the following rules:
1. If both numbers are NaN, they compare equal.
2. If only one number is NaN, they're incomparable.
3. Otherwise, the values are calculated into order-preserving integers which are
   then compared.

Note that floating-bar numbers only implement `PartialOrd` and not `Ord` due to the (currently) unspecified ordering of NaNs. This may change in the future.

## Float conversions

The algorithm for converting a floating-point number to a floating-bar number is
described by [John D. Cook's Best Rational Approximation post](https://www.johndcook.com/blog/2010/10/20/best-rational-approximation/),
with some minor tweaks to improve accuracy and performance. The algorithm splits
the space provided for the fraction into two for the numerator and denominator,
and then repeatedly calculates an upper and lower bound for the number until it
finds the closest approximation that will fit in that space.

Converting from floats in practice has shown to be accurate up to about 7 decimal digits.
*/

#![no_std]

#![cfg_attr(feature = "bench", feature(test))]

#[cfg(feature = "std")]
extern crate std;

use core::fmt;
use core::num::ParseIntError;

mod r32_t;
mod r64_t;

pub use r32_t::r32;
pub use r64_t::r64;

/// An error which can be returned when parsing a ratio.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseRatioErr {
	/// Value being parsed is empty.
	///
	/// Among other causes, this variant will be constructed when parsing an
	/// empty string.
	Empty,
	
	/// Numbers are too large to store together in the fraction field.
	Overflow,
	
	/// Error when parsing numerator.
	Numerator(ParseIntError),
	
	/// Error when parsing denominator.
	/// 
	/// This will contain an error kind of `Zero` if the denominator is `0`.
	Denominator(ParseIntError),
}

impl fmt::Display for ParseRatioErr {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			ParseRatioErr::Empty =>
				f.write_str("cannot parse rational from empty string"),
			
			ParseRatioErr::Overflow =>
				f.write_str("numbers are too large to fit in fraction"),
			
			ParseRatioErr::Numerator(pie) =>
				write!(f, "numerator error: {}", pie),
			
			ParseRatioErr::Denominator(pie) =>
				write!(f, "denominator error: {}", pie),
		}
	}
}

#[cfg(feature = "std")]
impl std::error::Error for ParseRatioErr {
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		match self {
			ParseRatioErr::Numerator(pie) => Some(pie),
			ParseRatioErr::Denominator(pie) => Some(pie),
			_ => None
		}
	}
}

/// Convenience macro for `r32` literals.
#[macro_export]
macro_rules! r32 {
	($numer:literal) => { r32!($numer / 1) };
	($numer:literal / $denom:literal) => {
		r32::new($numer, $denom).expect("literal out of range for `r32`")
	};
}

/// Convenience macro for `r64` literals.
#[macro_export]
macro_rules! r64 {
	($numer:literal) => { r64!($numer / 1) };
	($numer:literal / $denom:literal) => {
		r64::new($numer, $denom).expect("literal out of range for `r64`")
	};
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_ratio_type {
	($name:ident $uint:ident $int:ident $nzuint:ident) => {
	
impl $name {
	/// The highest value that can be represented by this rational type.
	pub const MAX: $name = $name(1 << (FRACTION_SIZE - 1));
	
	/// The lowest value that can be represented by this rational type.
	pub const MIN: $name = $name(1 << FRACTION_SIZE);
	
	/// The smallest positive value that can be represented by this rational
	/// type.
	pub const MIN_POSITIVE: $name = $name(FRACTION_SIZE << FRACTION_SIZE | FRACTION_FIELD);
	
	/// Not a Number (NaN).
	pub const NAN: $name = $name($uint::MAX);
	
	#[inline]
	fn denom_size(self) -> $uint {
		self.0 >> FRACTION_SIZE
	}
	
	#[inline]
	fn denom_mask(self) -> $uint {
		(1 << self.denom_size()) - 1
	}
	
	#[inline]
	fn numer_mask(self) -> $uint {
		FRACTION_FIELD & !self.denom_mask()
	}

	/// Returns the numerator of this rational number. `self` cannot be NaN.
	#[inline]
	pub(crate) fn numer(self) -> $int {
		// apparently this does sign-extension
		(self.0 as $int)
		.wrapping_shl(DSIZE_SIZE)
		.wrapping_shr(DSIZE_SIZE + (self.denom_size() as u32))
	}
	
	/// Returns the denominator of this rational number. `self` cannot be NaN.
	#[inline]
	pub(crate) fn denom(self) -> $uint {
		1 << self.denom_size() | (self.0 & self.denom_mask())
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
	pub fn trunc(self) -> $name {
		if self.is_nan() { return self }
		
		let numer = self.numer() / (self.denom() as $int);
		// the `& FRACTION_FIELD` is for negative results.
		$name((numer as $uint) & FRACTION_FIELD)
	}
	
	/// Returns the fractional part of a number, or NaN if `self` is NaN.
	#[inline]
	pub fn fract(self) -> $name {
		if self.is_nan() { return self }
		
		let numer = (self.numer() % (self.denom() as $int)) as $uint;
		// we can do this because all of self's bits will stay the same, apart
		// from the numerator.
		$name(
			self.0 & !self.numer_mask()
			| (numer << self.denom_size()) & FRACTION_FIELD
		)
	}
	
	/// Returns the largest integer less than or equal to a number.
	pub fn floor(self) -> $name {
		if self.is_negative() {
			// if self is a whole number,
			if self.numer() % (self.denom() as $int) == 0 {
				self
			} else {
				self.trunc() - $name(1)
			}
		} else {
			self.trunc()
		}
	}
	
	/// Returns the smallest integer greater than or equal to a number.
	pub fn ceil(self) -> $name {
		if self.is_positive() {
			// if self is a whole number,
			if self.numer() % (self.denom() as $int) == 0 {
				self
			} else {
				self.trunc() + $name(1)
			}
		} else {
			self.trunc()
		}
	}
	
	/// Returns the nearest integer to a number. Round half-way cases away from
	/// zero.
	pub fn round(self) -> $name {
		if self.is_negative() {
			unsafe { self - $name::new_unchecked(1, 2) }
		} else if self.is_positive() {
			unsafe { self + $name::new_unchecked(1, 2) }
		} else {
			self
		}
		.trunc()
	}
	
	/// Computes the absolute value of `self`.
	#[inline]
	pub fn abs(self) -> $name {
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
	pub fn signum(self) -> $name {
		if self.is_nan() {
			self
		} else if self.is_negative() {
			unsafe { $name::new_unchecked(-1, 1) }
		} else if self.is_positive() {
			$name(1)
		} else {
			$name(0)
		}
	}
	
	/// Takes the reciprocal (inverse) of a number, `1/x`.
	/// 
	/// # Panics
	/// 
	/// Panics when the numerator is zero.
	#[inline]
	pub fn recip(self) -> $name {
		self.checked_recip().expect("attempt to divide by zero")
	}
	
	/// Cancels out common factors between the numerator and the denominator.
	pub fn normalize(self) -> $name {
		if self.is_nan() { return self }
		
		if self.numer() == 0 {
			return $name(0);
		}
		
		let n = self.numer();
		let d = self.denom();
		
		// cancel out common factors by dividing numerator and denominator by
		// their greatest common divisor.
		let gcd = n.unsigned_abs().gcd(d);
		unsafe { $name::new_unchecked(n / (gcd as $int), d / gcd) }
	}
	
	/// Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
	/// overflow occurred.
	pub fn pow(self, exp: i32) -> $name {
		if exp == 0 { return $name(1) }
		if self.is_nan() { return self }
		
		let exp_is_neg = exp < 0;
		let exp = exp.unsigned_abs();
		
		let num = self.numer().pow(exp);
		let den = self.denom().pow(exp);
		
		if exp_is_neg {
			$name::new(num, den).map($name::recip)
		} else {
			$name::new(num, den)
		}
		.expect("attempt to multiply with overflow")
	}
	
	/// Returns the maximum of the two numbers.
	/// 
	/// If one of the arguments is `NaN`, then the other argument is returned.
	pub fn max(self, other: $name) -> $name {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true)   => $name::NAN,
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
	pub fn min(self, other: $name) -> $name {
		match (self.is_nan(), other.is_nan()) {
			// this clobbers any "payload" bits being used.
			(true, true)   => $name::NAN,
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
	pub fn checked_neg(self) -> Option<$name> {
		if self.is_nan() { return Some(self) }
		// yes, this is the simplest and quickest way.
		$name::new(-self.numer(), self.denom())
	}
	
	/// Checked absolute value. Computes `self.abs()`, returning `None` if the
	/// numerator would overflow.
	#[inline]
	pub fn checked_abs(self) -> Option<$name> {
		if self.is_negative() {
			self.checked_neg()
		} else {
			Some(self)
		}
	}
	
	/// Checked reciprocal. Computes `1/self`, returning `None` if the
	/// numerator is zero.
	pub fn checked_recip(self) -> Option<$name> {
		if self.is_nan() {
			Some(self)
		} else if self.numer() == 0 {
			None
		} else {
			let mut denom = self.denom() as $int;
			if self.is_negative() { denom = -denom }
			$name::new(denom, self.numer().unsigned_abs())
		}
	}
	
	/// Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
	/// overflow occurred.
	pub fn checked_pow(self, exp: i32) -> Option<$name> {
		if exp == 0 { return Some($name(1)) }
		if self.is_nan() { return Some($name::NAN) }
		
		let exp_is_neg = exp < 0;
		let exp = exp.unsigned_abs();
		
		let num = self.numer().checked_pow(exp)?;
		let den = self.denom().checked_pow(exp)?;
		
		if exp_is_neg {
			$name::new(num, den)?.checked_recip()
		} else {
			$name::new(num, den)
		}
	}
	
	/// Checked subtraction. Computes `self - rhs`, returning `None` if
	/// overflow occurred.
	pub fn checked_sub(self, rhs: $name) -> Option<$name> {
		self.checked_add(rhs.checked_neg()?)
	}
	
	/// Checked rational division. Computes `self / rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[inline]
	pub fn checked_div(self, rhs: $name) -> Option<$name> {
		self.checked_mul(rhs.checked_recip()?)
	}
	
	/// Checked rational remainder. Computes `self % rhs`, returning `None` if
	/// `rhs == 0` or the division results in overflow.
	#[inline]
	pub fn checked_rem(self, rhs: $name) -> Option<$name> {
		let div = self.checked_div(rhs)?;
		div.checked_sub(div.floor())?.checked_mul(rhs)
	}
	
	/// Raw transmutation to `u64`.
	/// 
	/// Useful if you need access to the payload bits of a NaN value.
	#[inline]
	pub fn to_bits(self) -> $uint { self.0 }
	
	/// Raw transmutation from `u64`.
	#[inline]
	pub fn from_bits(bits: $uint) -> $name { $name(bits) }
}

impl fmt::Display for $name {
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

impl fmt::Debug for $name {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		if self.is_nan() {
			f.write_str("NaN")
		} else {
			write!(f, "{}/{}", self.numer(), self.denom())
		}
	}
}

impl FromStr for $name {
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
	fn from_str(src: &str) -> Result<Self, Self::Err> {
		use core::num::$nzuint;
		
		if src.is_empty() {
			return Err(ParseRatioErr::Empty);
		}
		
		// special case NaN
		if src == "NaN" {
			return Ok($name::NAN);
		}
		
		// lookahead to find dividing bar, if any
		let bar_pos = src.find('/');
		let numer_end = bar_pos.unwrap_or(src.len());
		
		// parse numerator
		let numerator = src[..numer_end]
			.parse::<$int>()
			.map_err(ParseRatioErr::Numerator)?;
		
		// parse optional denominator
		let denominator = bar_pos
			.map(|pos|
				src[pos+1..]
				.parse::<$nzuint>()
				.map_err(ParseRatioErr::Denominator)
			) // : Option<Result<u32, ParseRatioErr>>
			.transpose()?
			// : Option<u32>
			.map($nzuint::get)
			.unwrap_or(1);
		
		// ensures parsed numbers fit in fraction field
		$name::new(numerator, denominator)
		.ok_or(ParseRatioErr::Overflow)
	}
}

impl From<u8> for $name {
	#[inline]
	fn from(v: u8) -> Self { $name(v as $uint) }
}

impl From<i8> for $name {
	fn from(v: i8) -> Self {
		unsafe { $name::new_unchecked(v as $int, 1) }
	}
}

impl PartialEq for $name {
	fn eq(&self, other: &$name) -> bool {
		self.is_nan() && other.is_nan()
		|| self.normalize().0 == other.normalize().0
	}
}

impl Neg for $name {
	type Output = $name;
	
	fn neg(self) -> Self::Output {
		self.checked_neg().expect("attempt to negate with overflow")
	}
}

impl Add for $name {
	type Output = $name;
	
	fn add(self, other: $name) -> Self::Output {
		self.checked_add(other).expect("attempt to add with overflow")
	}
}

impl AddAssign for $name {
	fn add_assign(&mut self, other: $name) {
		*self = *self + other
	}
}

impl Sub for $name {
	type Output = $name;

	fn sub(self, other: $name) -> Self::Output {
		self.checked_sub(other).expect("attempt to subtract with overflow")
	}
}

impl SubAssign for $name {
	fn sub_assign(&mut self, other: $name) {
		*self = *self - other
	}
}

impl Mul for $name {
	type Output = $name;
	
	fn mul(self, other: $name) -> Self::Output {
		self.checked_mul(other).expect("attempt to multiply with overflow")
	}
}

impl MulAssign for $name {
	fn mul_assign(&mut self, other: $name) {
		*self = *self * other
	}
}

impl Div for $name {
	type Output = $name;

	fn div(self, other: $name) -> Self::Output {
		self.checked_div(other).expect("attempt to divide with overflow")
	}
}

impl DivAssign for $name {
	fn div_assign(&mut self, other: $name) {
		*self = *self / other
	}
}

impl Rem for $name {
	type Output = $name;
	
	fn rem(self, other: $name) -> Self::Output {
		self.checked_rem(other).expect("attempt to divide with overflow")
	}
}

impl RemAssign for $name {
	fn rem_assign(&mut self, other: $name) {
		*self = *self % other
	}
}

	} // end of macro case
} // end of macro
