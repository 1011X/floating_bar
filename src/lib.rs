/*!
This library provides the floating bar type, which allows for efficient
representation of rational numbers without loss of precision. It is based on
[Inigo Quilez's blog post exploring the concept](http://www.iquilezles.org/www/articles/floatingbar/floatingbar.htm).

## Examples

```rust
use floating_bar::r32;

let done = 42;
let total = 100;
let progress = r32::new(done, total);
println!("{}", progress);
```

## Structure

The floating bar types follow a general structure:
* the denominator **size field** (always log<sub>2</sub> of the type's total size)
* the **fraction field** (uses all the remaining bits)

More concretely, they are represented like this:

```txt
d = denominator size, f = fraction field

r32: dddddfffffffffffffffffffffffffff
r64: ddddddffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
```

The fraction field stores both the numerator and the denominator. The size of
the denominator is determined by the denominator-size field, which gives the position of the partition (the "fraction bar") from the right.

The denominator has an implicit 1 bit which goes in front of the actual value
stored. Thus, a size field of zero has an implicit denominator value of 1,
making it compatible with integers. This is similar to the implicit bit
convention followed by floating-point numbers.

## NaNs

It's possible to have invalid values in this format, which are denoted as NaN
for "Not a Number". Invalid values are those whose value in the denominator-size
field is greater than or equal to the size of the entire fraction field.

This library focuses on the numeric value of this type, and is meant to limit
the propagation of NaNs. Any operation that could give an undefined value (e.g.
when overflowing or dividing by zero) will panic instead of returning a NaN.
Effort is put in to not clobber possible payload values in NaNs, but no
guarantees are made.

### Comparisons

By default, floating-bar numbers implement only `PartialOrd` and not `Ord`.

For `PartialOrd`, floating-bar numbers follow these rules:
1. If both numbers are NaN, they're equal.
2. If both numbers are zero, they're equal.
3. If only one number is NaN, they're incomparable and `None` is returned.
4. Their signs are then checked and,
   - If they're different, a comparison of the signs is returned.
   - If they're the same, a comparison of the numbers is returned.

## Float conversions

Converting to float in practice has shown to be accurate up to 7 digits.
*/

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "bench", feature(test))]

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
