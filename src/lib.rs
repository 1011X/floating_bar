/*!
This library provides the floating bar type, which allows for efficient
representation of rational numbers without loss of precision. It is based on
[this blog post](http://www.iquilezles.org/www/articles/floatingbar/floatingbar.htm).

## Structure

The floating bar types follow a general structure:
* the **sign bit**
* the denominator **size field** (always log_2 of the type's total size)
* the **fraction field** (uses all the remaining bits)

More concretely, they are represented like this:

```ignore
s = sign, d = denominator size, f = fraction field

r32: sdddddffffffffffffffffffffffffff
r64: sddddddfffffffffffffffffffffffffffffffffffffffffffffffffffffffff
```

The fraction field stores both the numerator and the denominator as separate
values. Their exact size at any given moment depends on the size field, which
gives the position of the partition (the "bar") from the right between the two
values.

The denominator has an implicit 1 bit which goes in front of the actual value
stored. Thus, a size field of zero has an implicit denominator value of 1,
making it compatible with integers.

## Features

### `denormals`

This enables denormal values. When the value of the denominator takes up the
whole fraction field, the numerator will take an implicit value of 1.

Due to the performance penalty of calculating with denormal values, this is
disabled by default.

## NaN's

Unfortunately, it's possible to have invalid values with this format. Invalid
values are those which have a denominator size larger than the number of bits in
the fraction field, and are represented as `NaN`.

This library focuses on the numeric value of the format and is meant to limit
the propagation of NaNs. Any operation that could give an undefined value (e.g.
when overflowing or dividing by zero) will panic instead of returning a NaN.
Effort is put in to not clobber possible payload values in NaNs, but no
guarantees are made.
*/

#![cfg_attr(feature = "bench", feature(test))]

use std::fmt;
use std::error;
use std::num::ParseIntError;

mod r32_t;
mod r64_t;

pub use r32_t::r32;
pub use r64_t::r64;

/// An error which can be returned when parsing a rational number.
/// 
/// # Potential causes
/// 
/// Among other causes, `ParseRatioErr` can be thrown because of leading or
/// trailing whitespace in the string e.g. when it is obtained from the standard
/// input. Using the `str.trim()` method ensures that no whitespace remains
/// before parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseRatioErr {
    kind: RatioErrKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RatioErrKind {
    Empty,
    Invalid,
    Overflow,
    Int(ParseIntError),
}

impl fmt::Display for ParseRatioErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.kind {
            RatioErrKind::Empty =>
            	f.write_str("cannot parse rational from empty string"),
            RatioErrKind::Invalid =>
            	f.write_str("invalid rational literal"),
            RatioErrKind::Overflow =>
            	f.write_str("number too large to fit in target type"),
            RatioErrKind::Int(pie) =>
            	pie.fmt(f),
        }
    }
}

impl error::Error for ParseRatioErr {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.kind {
            RatioErrKind::Int(pie) => Some(pie),
            _ => None
        }
    }
}

impl From<ParseIntError> for ParseRatioErr {
    fn from(pie: ParseIntError) -> ParseRatioErr {
        ParseRatioErr { kind: RatioErrKind::Int(pie) }
    }
}
