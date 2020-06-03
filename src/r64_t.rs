#![allow(dead_code)]
#![allow(unused_variables)]

use std::fmt;
use std::cmp::Ordering;
use std::ops::*;
use std::str::FromStr;

use gcd::Gcd;
use integer_sqrt::IntegerSquareRoot;

use super::{ParseRatioErr, RatioErrKind, r32};

/// The 64-bit floating bar type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, Default)]
pub struct r64(u64);

const SIGN_BIT: u64 = 0x8000_0000_0000_0000;
const SIZE_FIELD: u64 = SIGN_BIT - 1 << FRACTION_SIZE + 1 >> 1;
const FRACTION_FIELD: u64 = (1 << FRACTION_SIZE) - 1;

const FRACTION_SIZE: u64 = 57;

pub const NAN: r64 = r64(SIZE_FIELD);
pub const MAX: r64 = r64(FRACTION_FIELD);
pub const MIN: r64 = r64(SIGN_BIT | FRACTION_FIELD);
pub const MIN_POSITIVE: r64 = r64(FRACTION_SIZE << FRACTION_SIZE | FRACTION_FIELD);

impl r64 {
    // TODO unfinished; check input values that overflow
    #[inline]
    fn new(num: i64, den: u64) -> r64 {
        let size = 64 - den.leading_zeros() - 1;
        let denom_field = (1 << size) - 1;
        
        r64(0).set_sign(num.is_negative())
        .set_denom_size(size as u64)
        .set_fraction(
            ((denom_field ^ FRACTION_FIELD) >> size & num.abs() as u64) << size |
            den & denom_field
        )
    }
    
    #[inline]
    fn denom_size(self) -> u64 {
        (self.0 & SIZE_FIELD) >> FRACTION_SIZE
    }
    
    /// Returns the numerator value for this rational number.
    #[inline]
    pub fn numer(self) -> u64 {
        if self.denom_size() == FRACTION_SIZE { 1 }
        else { (self.0 & FRACTION_FIELD) >> self.denom_size() }
    }
    
    /// Returns the denominator value for this rational number.
    #[inline]
    pub fn denom(self) -> u64 {
        let denom_region = (1 << self.denom_size()) - 1;
        1 << self.denom_size() | self.0 & denom_region
    }
    
    /// Sets sign bit to the value given.
    /// 
    /// If `true`, sign bit is set. Otherwise, it's unset.
    #[inline]
    fn set_sign(self, sign: bool) -> r64 {
        r64(self.0 & !SIGN_BIT | (sign as u64) << 63)
    }
    
    #[inline]
    fn set_denom_size(self, size: u64) -> r64 {
        r64(self.0 & !SIZE_FIELD | (size & 0x3f) << FRACTION_SIZE)
    }
    
    #[inline]
    fn set_fraction(self, frac: u64) -> r64 {
        r64(self.0 & !FRACTION_FIELD | frac & FRACTION_FIELD)
    }
    
    #[inline]
    fn from_parts(sign: bool, numer: u64, denom: u64) -> r64 {
        let size = 64 - denom.leading_zeros() - 1;
        let denom_field = (1 << size) - 1;
        r64(
            if sign { SIGN_BIT } else { 0 } |
            (size as u64) << FRACTION_SIZE |
            ((denom_field ^ FRACTION_FIELD) >> size & numer) << size |
            denom & denom_field
        )
    }
    
    #[inline]
    fn is_sign_positive(self) -> bool {
        self.0 & SIGN_BIT == 0
    }
    
    #[inline]
    fn is_sign_negative(self) -> bool {
        self.0 & SIGN_BIT != 0
    }
    
    // BEGIN related float stuff
    
    /// Returns the largest integer less than or equal to a number.
    pub fn floor(self) -> r64 {
        if self.is_negative() {
            // if self is a whole number,
            if self.numer() % self.denom() == 0 {
                self
            } else {
                r64::from_parts(self.is_negative(), self.numer() / self.denom() + 1, 1)
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
            if self.numer() % self.denom() == 0 {
                self
            } else {
                r64::from_parts(self.is_negative(), self.numer() / self.denom() + 1, 1)
            }
        }
    }
    
    /// Returns the nearest integer to a number. Round half-way cases away from
    /// zero.
    pub fn round(self) -> r64 {
        if self.is_negative() {
            (self - r64(1) / r64(2)).ceil()
        } else {
            (self + r64(1) / r64(2)).floor()
        }
    }
    
    /// Returns the integer part of a number.
    #[inline]
    pub fn trunc(self) -> r64 {
        r64::from_parts(self.is_negative(), self.numer() / self.denom(), 1)
    }
    
    /// Returns the fractional part of a number.
    #[inline]
    pub fn fract(self) -> r64 {
        let d = self.denom();
        r64::from_parts(self.is_negative(), self.numer() % d, d)
    }
    
    /// Computes the absolute value of `self`. Returns NaN if the number is NaN.
    #[inline]
    pub fn abs(self) -> r64 {
        self.set_sign(false)
    }
    
    /// Returns a number that represents the sign of `self`.
    /// 
    /// * `1` if the number is positive
    /// * `-1` if the number is negative
    /// * `0` if the number is `+0`, `-0`, or `NaN`
    pub fn signum(self) -> r64 {
        if self.numer() == 0 || self.is_nan() {
            r64(0)
        } else {
            r64(self.0 & SIGN_BIT | 1)
        }
    }
    
    /// Raises a number to an integer power.
    // TODO: check that the new values fit in the type.
    pub fn pow(self, p: i32) -> r64 {
        let num = self.numer().pow(p.abs() as u32);
        let den = self.denom().pow(p.abs() as u32);

        // power is positive
        if p >= 0 {
            r64::from_parts(self.is_negative(), num, den)
        } else {
		    // power is negative; switch numbers around
            r64::from_parts(self.is_negative(), den, num)
        }
    }
    
    /// Raises a number to an integer power.
    // TODO: check that the new values fit in the type.
    pub fn checked_pow(self, p: i32) -> Option<r64> {
        let num = self.numer().checked_pow(p.abs() as u32);
        let den = self.denom().checked_pow(p.abs() as u32);

        match (num, den) {
            (Some(num), Some(den)) =>
                // power is positive
                Some(if p >= 0 {
                    r64::from_parts(self.is_negative(), num, den)
                }
                // power is negative; switch numbers around
                else {
                    r64::from_parts(self.is_negative(), den, num)
                }),
            _ => None
        }
    }
    
    /// Takes the *checked* square root of a number.
    /// 
    /// If `self` is positive and both the numerator and denominator are perfect
    /// squares, this returns their square root. Otherwise, returns `None`.
    pub fn checked_sqrt(self) -> Option<r64> {
        let nsqrt = self.numer().integer_sqrt();
        let dsqrt = self.denom().integer_sqrt();

        if self.numer() == nsqrt * nsqrt && self.denom() == dsqrt * dsqrt {
            Some(r64::from_parts(self.is_negative(), nsqrt, dsqrt))
        } else {
            None
        }
    }
    
    /// Takes the square root of a number.
    /// 
    /// If `self` is positive, it approximates its square root by calculating
    /// a repeated fraction for a fixed number of steps.
    /// 
    /// **Warning**: This method can give a number that overflows easily, so
    /// use it with caution, and discard it as soon as you're done with it.
    #[doc(hidden)]
    pub fn sqrt(self) -> r64 {
        // TODO: maybe there's a better way?
        let f: f64 = self.into();
        r64::from(f.sqrt())
    }
    /*
    TODO consider whether to actually add these.
    /// Takes the cube root of a number.
    /// 
    /// If `self` is positive and its numerator and denominator are perfect
    /// cubes, returns their cube root. Otherwise, returns `None`.
    #[inline]
    pub fn checked_cbrt(self) -> Option<r64> {
        unimplemented!()
    }
    */
    /// Returns `true` if this value is `NaN` and `false` otherwise.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.denom_size() > FRACTION_SIZE
    }
    
    /// Returns `true` if the number is neither zero, subnormal, or `NaN`.
    #[inline]
    pub fn is_normal(self) -> bool {
        self.numer() != 0
        && !(self.numer() == 1 && self.denom() == FRACTION_SIZE)
        && !self.is_nan()
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
    #[inline]
    pub fn recip(self) -> r64 {
        assert!(self.numer() != 0, "attempt to divide by zero");
        assert!(self.denom_size() < FRACTION_SIZE, "subnormal overflow");
        r64::from_parts(self.is_negative(), self.denom(), self.numer())
    }
    
    /// Returns the maximum of the two numbers.
    /// 
    /// If one of the arguments is `NaN`, then the other argument is returned.
    pub fn max(self, other: r64) -> r64 {
        match (self.is_nan(), other.is_nan()) {
            // this clobbers any "payload" bits being used.
            (true, true) => NAN,
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
            (true, true) => NAN,
            (true, false) => self,
            (false, true) => other,
            (false, false) => match self.partial_cmp(&other).unwrap() {
                Ordering::Greater => other,
                _ => self
            }
        }
    }
    
    /// Raw transmutation to `u64`.
    #[inline]
    pub fn to_bits(self) -> u64 { self.0 }
    
    /// Raw transmutation from `u64`.
    #[inline]
    pub fn from_bits(bits: u64) -> r64 { r64(bits) }
    
    /// Cancels out common factors between the numerator and the denominator.
    pub fn simplify(self) -> r64 {
        if self.is_nan() {
            return self;
        }
        
        if self.numer() == 0 {
            return r64(0);
        }
        
        let n = self.numer();
        let d = self.denom();
        
        // cancel out common factors
        let gcd = n.gcd(d);
        r64::from_parts(self.is_negative(), n / gcd, d / gcd)
    }
    
    // BEGIN related integer stuff
    
    /// Checked integer addition. Computes `self + rhs`, returning `None` if
    /// overflow occurred.
    #[doc(hidden)]
    pub fn checked_add(self, rhs: r64) -> Option<r64> {
        unimplemented!()
    }
    
    /// Checked integer subtraction. Computes `self - rhs`, returning `None` if
    /// overflow occurred.
    #[doc(hidden)]
    pub fn checked_sub(self, rhs: r64) -> Option<r64> {
        unimplemented!()
    }
    
    /// Checked integer multiplication. Computes `self * rhs`, returning `None`
    /// if overflow occurred.
    #[doc(hidden)]
    pub fn checked_mul(self, rhs: r64) -> Option<r64> {
        unimplemented!()
    }
    
    /// Checked integer division. Computes `self / rhs`, returning `None` if
    /// `rhs == 0` or the division results in overflow.
    #[doc(hidden)]
    pub fn checked_div(self, rhs: r64) -> Option<r64> {
        unimplemented!()
    }
    
    /// Checked integer remainder. Computes `self % rhs`, returning `None` if
    /// `rhs == 0` or the division results in overflow.
    #[doc(hidden)]
    pub fn checked_rem(self, rhs: r64) -> Option<r64> {
        unimplemented!()
    }
}

impl fmt::Display for r64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_nan() {
            return f.write_str("NaN");
        }
        
        let norm = self.simplify();
        
        if norm.is_negative() {
            f.write_str("-")?;
        }
        
        write!(f, "{}", norm.numer())?;
        
        if norm.denom_size() > 0 {
            write!(f, "/{}", norm.denom())?;
        }
        
        Ok(())
    }
}

impl fmt::Debug for r64 {
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
        if src.is_empty() {
            return Err(ParseRatioErr { kind: RatioErrKind::Empty });
        }
        
        if src == "NaN" {
            return Ok(NAN);
        }
        
        // if bar exists, parse as fraction
        if let Some(pos) = src.find('/') {
            // bar is at the end. invalid.
            if pos == src.len() - 1 {
                return Err(ParseRatioErr { kind: RatioErrKind::Invalid });
            }
            
            let numerator: i64 = src[0..pos].parse()?;
            let denominator: u64 = src[pos+1..].parse()?;
            
            if denominator == 0 {
                return Err(ParseRatioErr { kind: RatioErrKind::Invalid });
            }
            
            let sign = numerator < 0;
            let denom_size = 64 - denominator.leading_zeros() - 1;
            
            // if subnormal, return early
            if numerator.abs() == 1 && denom_size as u64 == FRACTION_SIZE {
                let denominator = denominator & FRACTION_FIELD;
                return Ok(r64::from_parts(sign, 1, denominator));
            }
            
            // ensure both fragments fit in the fraction field
            let frac_size = denom_size + (64 - numerator.leading_zeros());
            
            if frac_size as u64 > FRACTION_SIZE {
                Err(ParseRatioErr { kind: RatioErrKind::Overflow })
            }
            else {
                Ok(r64::from_parts(sign, numerator.abs() as u64, denominator))
            }
        }
        // otherwise, parse as integer.
        else {
            let numerator: i64 = src.parse()?;
            let mag = numerator.checked_abs()
                .ok_or(ParseRatioErr { kind: RatioErrKind::Overflow })?;
            let frac_size = 64 - mag.leading_zeros();
            
            if frac_size as u64 > FRACTION_SIZE {
                return Err(ParseRatioErr { kind: RatioErrKind::Overflow });
            }
            
            Ok(r64::from_parts(numerator < 0, mag as u64, 1))
        }
    }
}

impl From<u8> for r64 {
    #[inline]
    fn from(v: u8) -> Self { r64(v as u64) }
}

impl From<i8> for r64 {
    fn from(v: i8) -> Self {
        let n = if v == i8::min_value() { 128 } else { v.abs() as u64 };
        r64::from_parts(v.is_negative(), n, 1)
    }
}

impl From<u16> for r64 {
    #[inline]
    fn from(v: u16) -> Self { r64(v as u64) }
}

impl From<i16> for r64 {
    fn from(v: i16) -> Self {
        let n = if v == i16::min_value() { 32768 } else { v.abs() as u64 };
        r64::from_parts(v.is_negative(), n, 1)
    }
}

impl From<u32> for r64 {
    #[inline]
    fn from(v: u32) -> Self { r64(v as u64) }
}

impl From<i32> for r64 {
    fn from(v: i32) -> Self {
        let n = if v == i32::min_value() { 4294967296 } else { v.abs() as u64 };
        r64::from_parts(v.is_negative(), n, 1)
    }
}

impl From<r32> for r64 {
    fn from(v: r32) -> Self {
        r64::from_parts(v.is_negative(), v.numer() as u64, v.denom() as u64)
    }
}

impl From<f32> for r64 {
    fn from(f: f32) -> Self {
        r32::from(f).into()
    }
}

impl From<f64> for r64 {
    /// Based on: https://www.johndcook.com/blog/2010/10/20/best-rational-approximation/
    fn from(mut f: f64) -> Self {
        // why 29? bc it's fraction_size / 2 + 1
        // div by 2 is to have enough space for both numer and denom.
        // plus 1 is to count implicit bit bc numer and denom can both have 29
        // bits of precision here.
        let N = (1 << 29) - 1; // 2^29 - 1 = 536870911
        let is_lorge = f.abs() > 1.0;
        let is_neg = f < 0.0;
        
        if f.is_nan() || f.is_infinite() {
            return NAN;
        }
        
        if is_lorge { f = f.recip(); }
        if is_neg   { f = f.abs();   }
        
        let (mut a, mut b) = (0, 1); // lower
        let (mut c, mut d) = (1, 1); // upper
        let mut is_mediant = false;
        
        // while neither denoms are too big,
        while b <= N && d <= N {
            let mediant = (a + c) as f64 / (b + d) as f64;
            
            if f == mediant {
                is_mediant = true;
                break;
            }
            else if f > mediant {
                a = a + c;
                b = b + d;
            }
            else {
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
        if is_lorge {
            return r64::from_parts(is_neg, result.1, result.0);
        } else {
            return r64::from_parts(is_neg, result.0, result.1);
        }
    }
}

impl Into<f32> for r64 {
    fn into(self) -> f32 {
        let s = if self.is_negative() { -1.0 } else { 1.0 };
        s * self.numer() as f32 / self.denom() as f32
    }
}

impl Into<f64> for r64 {
    fn into(self) -> f64 {
        let s = if self.is_negative() { -1.0 } else { 1.0 };
        s * self.numer() as f64 / self.denom() as f64
    }
}

impl Neg for r64 {
    type Output = r64;
    
    fn neg(self) -> Self::Output {
        r64(self.0 ^ SIGN_BIT)
    }
}

impl PartialEq for r64 {
    fn eq(&self, other: &r64) -> bool {
        self.is_nan() && other.is_nan()
        || self.numer() == 0 && other.numer() == 0
        || self.simplify().0 == other.simplify().0
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
        self.is_sign_positive()
        .partial_cmp(&other.is_sign_positive())
        .map(|c| c.then(
            // compare numbers
            // a/b = c/d <=> ad = bc
            // when a, b, c, and d are all > 0
            (self.numer() as u128 * other.denom() as u128)
            .cmp(&(self.denom() as u128 * other.numer() as u128))
        ))
    }
}

impl Mul for r64 {
    type Output = r64;
    
    fn mul(self, other: r64) -> r64 {
        let s = self.is_negative() != other.is_negative();
        let mut n = self.numer() as u128 * other.numer() as u128;
        let mut d = self.denom() as u128 * other.denom() as u128;
        
        let mut min_size =
        	((128 - d.leading_zeros() - 1) + (128 - n.leading_zeros())) as u64;
        
        if min_size > FRACTION_SIZE {
		    let gcd = n.gcd(d);
		    n /= gcd;
		    d /= gcd;
		    min_size = ((128 - d.leading_zeros() - 1) + (128 - n.leading_zeros())) as u64;
	    }
        
        debug_assert!(min_size <= FRACTION_SIZE, "attempt to multiply with overflow");
        
        r64::from_parts(s, n as u64, d as u64)
    }
}

impl Div for r64 {
    type Output = r64;

    fn div(self, other: r64) -> r64 {
        self * other.recip()
    }
}

impl Add for r64 {
    type Output = r64;
    
    fn add(self, other: r64) -> r64 {
        // self = a/b, other = c/d
        
        let selfsign = (self.signum().0 as i64).signum();
        let othersign = (other.signum().0 as i64).signum();
        
        // TODO prove this won't panic/can't overflow.
        // num = ad + bc
        let num =
            (self.numer() as i64 * selfsign) * other.denom() as i64
            + self.denom() as i64 * (other.numer() as i64 * othersign);
        // den = bd
        let mut den = self.denom() as u128 * other.denom() as u128;
        let s = num.is_negative();
        let mut num = num.abs() as u128;
        
        let mut min_size =
        	((128 - den.leading_zeros() - 1) + (128 - num.leading_zeros())) as u64;
        
        if min_size > FRACTION_SIZE {
		    let gcd = num.gcd(den);
		    num /= gcd;
		    den /= gcd;
		    min_size = ((128 - den.leading_zeros() - 1) + (128 - num.leading_zeros())) as u64;
	    }
        
        debug_assert!(min_size <= FRACTION_SIZE, "attempt to add with overflow");
        
        r64::from_parts(s, num as u64, den as u64)
    }
}

impl Sub for r64 {
    type Output = r64;

    fn sub(self, other: r64) -> r64 {
        self + -other
    }
}

impl Rem for r64 {
    type Output = r64;
    
    fn rem(self, other: r64) -> r64 {
        let div = self / other;
        ((div - div.floor()) * other).set_sign(self.is_negative())
    }
}

#[cfg(test)]
mod tests {
	extern crate test;
	
	use test::Bencher;
    use super::*;
    
    #[test]
    fn simplify() {
        assert_eq!(r64::from_parts(false, 4, 2).simplify(), r64::from_parts(false, 2, 1));
        assert_eq!(r64::from_parts(true, 4, 2).simplify(), r64::from_parts(true, 2, 1));
    }
    
    #[test]
    fn neg() {
        assert_eq!((-r64(0)).0, SIGN_BIT);
        assert_eq!((-r64(SIGN_BIT)).0, 0);
    }
    
    #[test]
    fn signum() {
        assert_eq!(r64(0).signum(), r64(0));
        assert_eq!(r64(SIGN_BIT).signum(), r64(0));
        assert_eq!(r64(1).signum(), r64(1));
        assert_eq!(r64(2).signum(), r64(1));
        assert_eq!(r64::from_parts(true, 1, 1).signum(), r64::from_parts(true, 1, 1));
        assert_eq!(r64::from_parts(true, 2, 1).signum(), r64::from_parts(true, 1, 1));
    }

    #[test]
    fn pow() {
        assert_eq!(r64(0).pow(0), r64(1));
        assert_eq!(r64(1).pow(1), r64(1));
        assert_eq!(r64(2).pow(3), r64(8));
        assert_eq!(r64(2).pow(-3), r64::from_str("1/8").unwrap());
    }

    #[test]
    fn checked_pow() {
        assert_eq!(r64(0).checked_pow(0), Some(r64(1)));
        assert_eq!(r64(1).checked_pow(1), Some(r64(1)));
        assert_eq!(r64(2).checked_pow(3), Some(r64(8)));
        assert_eq!(r64(2).checked_pow(-3), Some(r64::from_str("1/8").unwrap()));
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
        assert_eq!(r64::from_parts(false, 3, 2).floor(), r64(1));
        assert_eq!(r64::from_parts(false, 2, 1).floor(), r64(2));
        assert_eq!(r64::from_parts(true, 3, 2).floor(), r64::from(-2_i8));
        assert_eq!(r64::from_parts(true, 2, 1).floor(), r64::from(-2_i8));
    }

    #[test]
    fn ceil() {
        assert_eq!(r64::from_parts(false, 3, 2).ceil(), r64(2));
        assert_eq!(r64::from_parts(false, 2, 1).ceil(), r64(2));
        assert_eq!(r64::from_parts(true, 3, 2).ceil(), r64::from(-1_i8));
        assert_eq!(r64::from_parts(true, 2, 1).ceil(), r64::from(-2_i8));
    }

    #[test]
    fn round() {
        assert_eq!(r64(1).round(), r64(1));
        assert_eq!((-r64(1)).round(), r64::from(-1_i8));
        assert_eq!((r64(3) / r64(2)).round(), r64(2));
        assert_eq!((-r64(3) / r64(2)).round(), r64::from(-2_i8));
    }
    
    #[test]
    fn fract() {
        assert_eq!(r64(5).fract(), r64(0));
        assert_eq!(r64::from_parts(false, 3, 2).fract(), r64::from_parts(false, 1, 2));
        assert_eq!(r64::from_parts(true, 3, 2).fract(), r64::from_parts(true, 1, 2));
    }
    
    #[test]
    fn trunc() {
        assert_eq!(r64(5).trunc(), r64(5));
        assert_eq!(r64::from_parts(false, 1, 2).trunc(), r64(0));
        assert_eq!(r64::from_parts(true, 1, 2).trunc(), r64(0));
        assert_eq!(r64::from_parts(false, 3, 2).trunc(), r64(1));
        assert_eq!(r64::from_parts(true, 3, 2).trunc(), r64::from(-1 as i8));
    }
    
    #[test]
    fn recip() {
        assert_eq!(r64(5).recip(), r64::from_parts(false, 1, 5));
        assert_eq!(r64::from_parts(false, 5, 2).recip(), r64::from_parts(false, 2, 5));
        assert_eq!(r64(1).recip(), r64(1));
    }
    
    #[test]
    fn cmp() {
        assert!(r64(0) == r64(0));
        assert!(r64(0) == -r64(0));
        
        assert!(r64(0) < r64(1));
        assert!(r64(2) < r64(3));
        assert!(r64(0) > -r64(1));
        assert!(r64(2) > -r64(3));
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
            r64::from_parts(false, 1, 2) * r64::from_parts(false, 1, 2),
            r64::from_parts(false, 1, 4)
        );
        assert_eq!(
            r64::from_parts(true, 1, 2) * r64::from_parts(false, 1, 2),
            r64::from_parts(true, 1, 4)
        );
        assert_eq!(
            r64::from_parts(false, 2, 3) * r64::from_parts(false, 2, 3),
            r64::from_parts(false, 4, 9)
        );
        assert_eq!(
            r64::from_parts(false, 3, 2) * r64::from_parts(false, 2, 3),
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
        
        assert_eq!(r64(1) / r64(2), r64::from_parts(false, 1, 2));
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
            r64::from_parts(false, 1, 2) + r64::from_parts(false, 3, 4),
            r64::from_parts(false, 5, 4)
        );
        assert_eq!(
            r64::from_parts(false, 1, 2) + r64::from_parts(true, 3, 4),
            r64::from_parts(true, 1, 4)
        );
        assert_eq!(
            r64::from_parts(true, 1, 2) + r64::from_parts(false, 3, 4),
            r64::from_parts(false, 1, 4)
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
    
    #[test]
    fn debug() {
        assert_eq!(format!("{:?}", r64::from_parts(true, 0, 1)), "-0/1");
        assert_eq!(format!("{:?}", NAN), "NaN");
    }
    
    #[test]
    fn display() {
        assert_eq!(format!("{}", r64::from_parts(false, 0, 1)), "0");
        assert_eq!(format!("{}", NAN), "NaN");
        assert_eq!(format!("{}", r64::from_parts(true, 3, 2)), "-3/2");
    }
    
    
    
    #[bench]
    fn f64_addition(b: &mut Bencher) {
    	let mut floats = Vec::new();
    	for i in 0..1000 {
    		floats.push(i as f64);
		}
    	b.iter(|| {
    		floats.iter().fold(0.0, |a, b| a + *b)
    	})
    }
    
    #[bench]
    fn r64_addition(b: &mut Bencher) {
    	let mut ratios = Vec::new();
    	for i in 0_i16..1000 {
    		ratios.push(r64::from(i));
		}
    	b.iter(|| {
    		ratios.iter().fold(r64(0), |a, b| a + *b)
    	})
    }
    
    #[bench]
    fn f64_multiplication(b: &mut Bencher) {
    	let mut floats = Vec::new();
    	for i in 0_i16..1000 {
    		floats.push(i as f64);
		}
    	b.iter(|| {
    		floats.iter().fold(0.0, |a, &b| a * b)
    	})
    }
    
    #[bench]
    fn r64_multiplication(b: &mut Bencher) {
    	let mut ratios = Vec::new();
    	for i in 0_i16..1000 {
    		ratios.push(r64::from(i));
		}
    	b.iter(|| {
    		ratios.iter().fold(r64(0), |a, &b| a * b)
    	})
    }
}
