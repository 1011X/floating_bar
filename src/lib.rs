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
* the **denominator-size field**: always log<sub>2</sub> of the type's total
  size, stored in the highest bits.
* the **fraction field**: stored in the remaining bits.

Here is a visual aid, where each character corresponds to one bit and the least
significant bit is on the right:
```txt
d = denominator size field, f = fraction field

r32: dddddfffffffffffffffffffffffffff
r64: ddddddffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
```

The fraction field stores both the numerator and the denominator. The size of
the denominator is determined by the denominator-size field, which gives the
position of the partition (the "fraction bar") from the right.

The numerator is stored as a two's complement signed integer on the left side of
the partition. The denominator is stored as an unsigned integer on the right
side. The denominator has an implicit 1 bit in front of its stored value,
similar to the implicit bit convention followed by floating-point. Thus, a size
field of zero has an implicit denominator value of 1.

## Value space

There are three categories that a floating-bar number can fall into: normal,
reducible, and not-a-number (also known as NaNs).

**NaN** values are those with an overly large denominator size as to leave no
room for the numerator. The library mostly ignores these values, and only uses
one particular value to provide a `NAN` constant. They can be used to store
payloads if desired using the `.to_bits()` and `from_bits()` functions. Effort
is put into not clobbering possible payload values during calculations, but no
guarantees are made.

**Reducible** values are those where the numerator and denominator share some
common factor that has not been canceled out, and thus use more bits than their
normalized form. Due to the performance cost of finding and canceling out common
factors, reducible values are only normalized when absolutely necessary, such as
when the result would otherwise overflow.

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

/// The error type returned when a checked rational type conversion fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryFromRatioError;

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
macro_rules! impl_ratio_traits {
	($name:ident $uint:ident $int:ident $nzuint:ident) => {

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
	#[inline]
	fn from(v: i8) -> Self {
		// SAFETY: I assure you that a signed byte will fit.
		unsafe { $name::new_unchecked(v as $int, 1) }
	}
}

impl PartialEq for $name {
	#[inline]
	fn eq(&self, other: &$name) -> bool {
		self.is_nan() && other.is_nan()
		|| self.normalize().0 == other.normalize().0
	}
}

impl Neg for $name {
	type Output = $name;
	
	#[inline]
	fn neg(self) -> Self::Output {
		self.checked_neg().expect("attempt to negate with overflow")
	}
}

impl AddAssign for $name {
	#[inline]
	fn add_assign(&mut self, other: $name) {
		*self = *self + other
	}
}

impl Sub for $name {
	type Output = $name;

	#[inline]
	fn sub(self, other: $name) -> Self::Output {
		self + (-other)
		//self.checked_sub(other).expect("attempt to subtract with overflow")
	}
}

impl SubAssign for $name {
	#[inline]
	fn sub_assign(&mut self, other: $name) {
		*self = *self - other
	}
}

impl MulAssign for $name {
	#[inline]
	fn mul_assign(&mut self, other: $name) {
		*self = *self * other
	}
}

impl Div for $name {
	type Output = $name;

	#[inline]
	fn div(self, other: $name) -> Self::Output {
		self * other.recip()
		//self.checked_div(other).expect("attempt to divide with overflow")
	}
}

impl DivAssign for $name {
	#[inline]
	fn div_assign(&mut self, other: $name) {
		*self = *self / other
	}
}

impl Rem for $name {
	type Output = $name;
	
	#[inline]
	fn rem(self, other: $name) -> Self::Output {
		// TODO: this inherits self's sign. is this what we want?
		(self / other).fract() * other
		// This will inherit other's sign.
		//let div = self / other;
		//(div - div.floor()) * other
	}
}

impl RemAssign for $name {
	#[inline]
	fn rem_assign(&mut self, other: $name) {
		*self = *self % other
	}
}

}}



#[doc(hidden)]
#[macro_export]
macro_rules! impl_ratio_tests {
	($ratio:ident) => {


#[cfg(test)]
mod tests {
	#[cfg(feature = "bench")]
	extern crate test;

	use super::*;
	use crate::$ratio;

	#[test]
	fn checked_pow() {
		assert_eq!($ratio(2).checked_pow(FRACTION_SIZE as i32), None);
	}
	/*
	#[test]
	#[cfg(feature = "roots")]
	fn checked_sqrt() {
		assert_eq!(r64(0).checked_sqrt(), Some(r64(0)));
		assert_eq!(r64(1).checked_sqrt(), Some(r64(1)));
		assert_eq!(r64(2).checked_sqrt(), None);
		assert_eq!(r64(4).checked_sqrt(), Some(r64(2)));
	}
	*/
	#[test]
	fn trunc() {
		assert_eq!($ratio::NAN.trunc(), $ratio::NAN);
		
		assert_eq!($ratio(5).trunc(),     $ratio(5));
		assert_eq!($ratio!( 1/2).trunc(), $ratio(0));
		assert_eq!($ratio!(-1/2).trunc(), $ratio(0));
		assert_eq!($ratio!( 3/2).trunc(), $ratio(1));
		assert_eq!($ratio!(-3/2).trunc(), $ratio!(-1));
	}
	
	#[test]
	fn fract() {
		assert_eq!($ratio::NAN.fract(), $ratio::NAN);
		
		assert_eq!($ratio(5).fract(),     $ratio(0));
		assert_eq!($ratio!( 3/2).fract(), $ratio!( 1/2));
		assert_eq!($ratio!(-3/2).fract(), $ratio!(-1/2));
	}

	#[test]
	fn floor() {
		assert_eq!($ratio::NAN.floor(), $ratio::NAN);
		
		assert_eq!($ratio(1).floor(),     $ratio(1));
		assert_eq!($ratio!(-1).floor(),   $ratio!(-1));
		assert_eq!($ratio!( 3/2).floor(), $ratio(1));
		assert_eq!($ratio!(-3/2).floor(), $ratio!(-2));
	}

	#[test]
	fn ceil() {
		assert_eq!($ratio::NAN.ceil(), $ratio::NAN);
		
		assert_eq!($ratio(1).ceil(),     $ratio(1));
		assert_eq!($ratio!(-1).ceil(),   $ratio!(-1));
		assert_eq!($ratio!( 3/2).ceil(), $ratio(2));
		assert_eq!($ratio!(-3/2).ceil(), $ratio!(-1));
	}

	#[test]
	fn round() {
		assert_eq!($ratio::NAN.round(), $ratio::NAN);
		
		assert_eq!($ratio(1).round(),     $ratio(1));
		assert_eq!($ratio!(-1).round(),   $ratio!(-1));
		assert_eq!($ratio!( 3/2).round(), $ratio(2));
		assert_eq!($ratio!(-3/2).round(), $ratio!(-2));
	}
	
	#[test]
	fn min() {
		assert_eq!($ratio::NAN.min($ratio::NAN), $ratio::NAN);
		assert_eq!($ratio::NAN.min($ratio(0)),   $ratio(0));
		assert_eq!($ratio(0).min($ratio::NAN),   $ratio(0));
		assert_eq!($ratio(0).min($ratio(1)),     $ratio(0));
	}
	
	#[test]
	fn max() {
		assert_eq!($ratio::NAN.max($ratio::NAN), $ratio::NAN);
		assert_eq!($ratio::NAN.max($ratio(0)),   $ratio(0));
		assert_eq!($ratio(0).max($ratio::NAN),   $ratio(0));
		assert_eq!($ratio(0).max($ratio(1)),     $ratio(1));
	}
	
	#[test]
	fn abs() {
		assert_eq!($ratio::NAN.abs(), $ratio::NAN);
		assert_eq!($ratio(0).abs(), $ratio(0));
		assert_eq!($ratio(1).abs(), $ratio(1));
		
		assert_eq!($ratio!(-1).abs(), $ratio(1));
	}
	
	#[test]
	fn signum() {
		assert_eq!($ratio::NAN.signum(), $ratio::NAN);
		
		assert_eq!($ratio(0).signum(), $ratio(0));
		assert_eq!($ratio(1).signum(), $ratio(1));
		assert_eq!($ratio(2).signum(), $ratio(1));
		
		assert_eq!($ratio!(-1).signum(), $ratio!(-1));
		assert_eq!($ratio!(-2).signum(), $ratio!(-1));
	}
	
	#[test]
	fn recip() {
		assert_eq!($ratio::NAN.recip(), $ratio::NAN);
		
		assert_eq!($ratio(5).recip(), $ratio!(1/5));
		assert_eq!($ratio!(5/2).recip(), $ratio!(2/5));
		assert_eq!($ratio(1).recip(), $ratio(1));
	}
	
	#[test]
	fn normalize() {
		assert_eq!($ratio!( 4 / 2).normalize(), $ratio!( 2));
		assert_eq!($ratio!(-4 / 2).normalize(), $ratio!(-2));
	}

	#[test]
	fn pow() {
		assert_eq!( $ratio::NAN.pow(0), $ratio(1) );
		
		assert_eq!( $ratio(0).pow(0),   $ratio(1) );
		assert_eq!( $ratio(1).pow(1),   $ratio(1) );
		
		assert_eq!( $ratio(3).pow( 2),   $ratio(9)    );
		assert_eq!( $ratio(3).pow(-2),   $ratio!(1/9) );
		assert_eq!( $ratio!(-3).pow( 2), $ratio(9)    );
		assert_eq!( $ratio!(-3).pow(-2), $ratio!(1/9) );
		
		assert_eq!( $ratio(2).pow( 3),    $ratio(8)    );
		assert_eq!( $ratio(2).pow(-3),    $ratio!(1/8) );
		assert_eq!( $ratio!(1/2).pow( 3), $ratio!(1/8) );
		assert_eq!( $ratio!(1/2).pow(-3), $ratio(8)    );
		
		assert_eq!( $ratio!(-2).pow( 3),   $ratio!(-8)   );
		assert_eq!( $ratio!(-2).pow(-3),   $ratio!(-1/8) );
		assert_eq!( $ratio!(-1/2).pow( 3), $ratio!(-1/8) );
		assert_eq!( $ratio!(-1/2).pow(-3), $ratio!(-8)   );
	}
	
	#[test]
	fn cmp() {
		assert!($ratio(0) == $ratio(0));
		
		assert!($ratio(0) < $ratio(1));
		assert!($ratio(2) < $ratio(3));
		assert!($ratio(0) > -$ratio(1));
		assert!($ratio(2) > -$ratio(3));
		
		// TODO more assertions here
	}
	
	#[test]
	fn neg() {
		assert_eq!(-$ratio!( 0), $ratio!( 0));
		assert_eq!(-$ratio!( 1), $ratio!(-1));
		assert_eq!(-$ratio!(-1), $ratio!( 1));
	}
	
	#[test]
	fn checked_neg() {
		let den = 1 << (FRACTION_SIZE - 1);
		assert_eq!($ratio::new(-1, den).unwrap().checked_neg(), None);
	}
	
	#[test]
	fn add() {
		assert_eq!($ratio(0) + $ratio(0), $ratio(0));
		
		assert_eq!($ratio(1) + $ratio(1), $ratio(2));
		assert_eq!($ratio(1) + $ratio!(-1), $ratio(0));
		assert_eq!($ratio!(-1) + $ratio(1), $ratio(0));
		assert_eq!($ratio!(-1) + $ratio!(-1), $ratio!(-2));
		
		assert_eq!($ratio(2)     + $ratio(2),     $ratio(4));
		assert_eq!($ratio!(1/2)  + $ratio!(3/4),  $ratio!(5/4));
		assert_eq!($ratio!(1/2)  + $ratio!(-3/4), $ratio!(-1/4));
		assert_eq!($ratio!(-1/2) + $ratio!(3/4),  $ratio!(1/4));
	}
	
	#[test]
	fn checked_add() {
		assert_eq!($ratio(0).checked_add($ratio(0)), Some($ratio(0)));
		
		assert_eq!($ratio(1).checked_add($ratio(1)), Some($ratio(2)));
		assert_eq!($ratio(1).checked_add($ratio!(-1)), Some($ratio(0)));
		assert_eq!($ratio!(-1).checked_add($ratio(1)), Some($ratio(0)));
		assert_eq!($ratio!(-1).checked_add($ratio!(-1)), Some($ratio!(-2)));
		
		assert_eq!($ratio(2).checked_add($ratio(2)), Some($ratio(4)));
		assert_eq!($ratio!(1/2).checked_add($ratio!(3/4)), Some($ratio!(5/4)));
		assert_eq!($ratio!(1/2).checked_add($ratio!(-3/4)), Some($ratio!(-1/4)));
		assert_eq!($ratio!(-1/2).checked_add($ratio!(3/4)), Some($ratio!(1/4)));
		
		assert_eq!($ratio::MAX.checked_add($ratio(1)), None);
	}
	
	#[test]
	fn mul() {
		assert_eq!($ratio(0) * $ratio(0), $ratio(0));
		
		assert_eq!($ratio(0) * $ratio(1), $ratio(0));
		assert_eq!($ratio(1) * $ratio(0), $ratio(0));
		assert_eq!($ratio(1) * $ratio(1), $ratio(1));
		
		assert_eq!(-$ratio(1) *  $ratio(1), -$ratio(1));
		assert_eq!( $ratio(1) * -$ratio(1), -$ratio(1));
		assert_eq!(-$ratio(1) * -$ratio(1),  $ratio(1));
		
		assert_eq!($ratio(1) * $ratio(2), $ratio(2));
		assert_eq!($ratio(2) * $ratio(2), $ratio(4));
		
		assert_eq!(
			$ratio!(1/2) * $ratio!(1/2), $ratio!(1/4)
		);
		assert_eq!(
			$ratio!(-1/2) * $ratio!(1/2), $ratio!(-1/4)
		);
		assert_eq!(
			$ratio!(2/3) * $ratio!(2/3), $ratio!(4/9)
		);
		assert_eq!(
			$ratio!(3/2) * $ratio!(2/3), $ratio(1)
		);
	}
	
	#[test]
	fn div() {
		assert_eq!($ratio(0) / $ratio(1), $ratio(0));
		assert_eq!($ratio(0) / $ratio(2), $ratio(0));
		assert_eq!($ratio(1) / $ratio(1), $ratio(1));
		
		assert_eq!(-$ratio(1) /  $ratio(1), -$ratio(1));
		assert_eq!( $ratio(1) / -$ratio(1), -$ratio(1));
		assert_eq!(-$ratio(1) / -$ratio(1),  $ratio(1));
		
		assert_eq!($ratio(1) / $ratio(2), $ratio!(1/2));
		assert_eq!($ratio(2) / $ratio(1), $ratio(2));
		assert_eq!($ratio(2) / $ratio(2), $ratio(1));
	}

	#[test]
	fn rem() {
		assert_eq!($ratio(5) % $ratio(2), $ratio(1));
		assert_eq!($ratio(6) % $ratio(2), $ratio(0));
		assert_eq!($ratio(8) % $ratio!(3 / 2), $ratio!(1 / 2));
		
		// always returns sign of dividend (1st number)
		assert_eq!(-$ratio(5) %  $ratio(2), -$ratio(1));
		assert_eq!( $ratio(5) % -$ratio(2),  $ratio(1));
		assert_eq!(-$ratio(5) % -$ratio(2), -$ratio(1));
	}
	
	#[test]
	fn from_str() {
		assert_eq!("NaN".parse::<$ratio>().unwrap(), $ratio::NAN);
		assert_eq!("0".parse::<$ratio>().unwrap(),   $ratio(0));
		assert_eq!("1".parse::<$ratio>().unwrap(),   $ratio(1));
		assert_eq!("+1".parse::<$ratio>().unwrap(),  $ratio(1));
		assert_eq!("-1".parse::<$ratio>().unwrap(),  $ratio!(-1));
		assert_eq!("1/1".parse::<$ratio>().unwrap(), $ratio(1));
	}
	
	#[test] #[should_panic]
	fn from_str_invalid() {
		"1/-1".parse::<$ratio>().unwrap();
		"/1".parse::<$ratio>().unwrap();
		"1/".parse::<$ratio>().unwrap();
		"1/0".parse::<$ratio>().unwrap();
	}
	/*
	#[test]
	fn from_f32() {
		assert_eq!($ratio::from(0.0), $ratio(0));
		assert_eq!($ratio::from(1.0), $ratio(1));
		assert_eq!($ratio::from(-1.0), -$ratio(1));
		assert_eq!($ratio::from(0.2), $ratio(1) / $ratio(5));
		assert_eq!($ratio::from(1.0 - 0.7), $ratio(3) / $ratio(10));
		//assert_eq!($ratio::from(std::f32::consts::E), $ratio(15062/5541));
		//assert_eq!($ratio::from(std::f32::consts::TAU), $ratio!(710/113));
		//assert_eq!($ratio::from(1.618033988749894), $ratio!(4181/2584));
		//assert_eq!($ratio::from(std::f32::consts::SQRT_2), $ratio(4756/3363));
	}
	
	#[test]
	fn from_f64() {
		assert_eq!($ratio::from(0.0), $ratio(0));
		assert_eq!($ratio::from(1.0), $ratio(1));
		assert_eq!($ratio::from(-1.0), -$ratio(1));
		assert_eq!($ratio::from(0.2), $ratio!(1/5));
		assert_eq!($ratio::from(1.0 - 0.7), $ratio!(3/10));
		//assert_eq!(r64::from(std::f64::consts::E), r64!(268876667 / 98914198));
		//assert_eq!(r64::from(std::f64::consts::TAU), r64!(411557987 / 65501488));
		//assert_eq!(r64::from(1.618033988749894), r64!(39088169 / 24157817));
		assert_eq!($ratio::from(std::f64::consts::SQRT_2), $ratio(11863283/8388608));
	}
	*/
}

}}
