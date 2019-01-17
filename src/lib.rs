/*!
The 32-bit floating bar type.
*/

use std::fmt;
use std::cmp::Ordering;
use std::ops::*;

use gcd::Gcd;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, Default, Hash)]
pub struct r32(u32);

const SIGN_BIT: u32 = 0x8000_0000;
const SIZE_FIELD: u32 = 0x7c00_0000;
const FRACTION_FIELD: u32 = 0x03ff_ffff;

const FRACTION_SIZE: u32 = 26;

pub const MAX: r32 = r32(FRACTION_FIELD);
pub const MIN: r32 = r32(SIGN_BIT | FRACTION_FIELD);
pub const MIN_POSITIVE: r32 = r32(25 << FRACTION_SIZE | FRACTION_FIELD);

impl r32 {
    #[inline]
    fn denom_size(self) -> u32 {
        (self.0 & SIZE_FIELD) >> FRACTION_SIZE
    }
    
    #[inline]
    fn numer(self) -> u32 {
        (self.0 & FRACTION_FIELD) >> self.denom_size()
    }
    
    #[inline]
    fn denom(self) -> u32 {
        let denom_region = (1 << self.denom_size()) - 1;
        self.0 & denom_region | 1 << self.denom_size()
    }
    /*
    #[inline]
    fn set_sign(self, sign: bool) -> r32 {
        let r = self.0 & !SIGN_BIT
        r | if sign { SIGN_BIT } else { 0 }
    }
    
    #[inline]
    fn set_size(self, size: u32) -> r32 {
        let r = self.0 & !SIZE_FIELD;
        r | sign << FRACTION_SIZE & SIZE_FIELD
    }
    
    #[inline]
    fn set_fraction(self, frac: u32) -> r32 {
        let r = self.0 & !FRACTION_FIELD;
        r | frac & FRACTION_FIELD
    }
    
    #[inline]
    fn set_denom(self, den: u32) -> r32 {
        let denom_field = (1 << self.denom_size()) - 1;
        self.0 & !denom_field | den & denom_field
    }
    
    #[inline]
    fn set_numer(self, num: u32) -> r32 {
        let size = self.denom_size();
        self.0 & (!FRACTION_FIELD | (1 << size) - 1) | num << size & FRACTION_FIELD
    }
    */
    #[inline]
    fn from_parts(sign: bool, numer: u32, denom: u32) -> r32 {
        let size = 32 - denom.leading_zeros() - 1;
        r32(
            if sign { SIGN_BIT } else { 0 } |
            size << FRACTION_SIZE |
            numer << size |
            denom & ((1 << size) - 1)
        )
    }
    /*
    pub fn floor(self) -> r32 {
        let n = self.numer() as i32 * if self.is_sign_negative() { -1 } else { 1 };
        let d = self.denom() as i32;
        unimplemented!();
        r32::from_parts(self.is_sign_negative(), n / d, d)
    }
    */
    #[inline]
    pub fn fract(self) -> r32 {
        let d = self.denom();
        r32::from_parts(self.is_sign_negative(), self.numer() % d, d)
    }
    
    #[inline]
    pub fn abs(self) -> r32 {
        r32(self.0 & !SIGN_BIT)
    }
    
    // always returns normal
    #[inline]
    pub fn signum(self) -> r32 {
        if self.numer() == 0 {
            r32(0)
        }
        else {
            r32(self.0 & SIGN_BIT | 1)
        }
    }
    
    #[inline]
    pub fn is_sign_positive(self) -> bool {
        self.0 & SIGN_BIT == 0
    }
    
    #[inline]
    pub fn is_sign_negative(self) -> bool {
        !self.is_sign_positive()
    }
    
    #[inline]
    pub fn recip(self) -> r32 {
        r32::from_parts(self.is_sign_negative(), self.denom(), self.numer())
    }
    
    #[inline]
    pub fn to_bits(self) -> u32 { self.0 }
    
    #[inline]
    pub fn from_bits(bits: u32) -> r32 { r32(bits) }
    
    pub fn normalize(self) -> r32 {
        // normalize -0 (or anything with 0 in numerator) to +0
        if self.numer() == 0 {
            return r32(0);
        }
        
        let n = self.numer();
        let d = self.denom();
        
        // cancel out common factors
        let gcd = n.gcd(d);
        r32::from_parts(self.is_sign_negative(), n / gcd, d / gcd)
    }
}

impl fmt::Display for r32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let norm = self.normalize();
        
        if norm.is_sign_negative() {
            f.write_str("-")?;
        }
        
        write!(f, "{}", norm.numer())?;
        
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
        
        write!(f, "{}/{}", self.numer(), self.denom())
    }
}
/*
impl FromStr for r32 {
    type Err = ParseRationErr;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        unimplemented!()
    }
}
*/
impl From<u8> for r32 {
    #[inline]
    fn from(v: u8) -> Self { r32(v as u32) }
}

impl From<i8> for r32 {
    fn from(v: i8) -> Self {
        let n = if v == i8::min_value() { 128 } else { v.abs() as u32 };
        r32::from_parts(v.is_negative(), n, 1)
    }
}

impl From<u16> for r32 {
    #[inline]
    fn from(v: u16) -> Self { r32(v as u32) }
}

impl From<i16> for r32 {
    fn from(v: i16) -> Self {
        let n = if v == i16::min_value() { 32768 } else { v.abs() as u32 };
        r32::from_parts(v.is_negative(), n, 1)
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
        /* TODO check performance of including these
        // self = a/b, other = c/d
        // a = 0 & c = 0
        sn == 0 && on == 0
        // sign(self) = sign(other) & ...
        || self.is_sign_positive() == other.is_sign_positive()
        && (
            // a = c & b = d
            sn == on && sd == od
            // b | a  &  d | c  &  a/b = c/d
            || sd % sn == 0 && od % sn == 0
            && sn / sd == on / od
        ) ||
        */
        self.normalize().0 == other.normalize().0
    }
}

impl Ord for r32 {
    fn cmp(&self, other: &Self) -> Ordering {
        // TODO this better
        (self.signum().0 as i32)
        .cmp(&(other.signum().0 as i32))
        .then_with(||
            // a/b > c/d <=> ad > bc
            // when a, b, c, and d are all > 0
            (self.numer() as u64 * other.denom() as u64)
            .cmp(&(self.denom() as u64 * other.numer() as u64))
        )
    }
}

impl PartialOrd for r32 {
    fn partial_cmp(&self, other: &r32) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Mul for r32 {
    type Output = r32;
    
    fn mul(self, other: r32) -> r32 {
        let s = self.is_sign_positive() != other.is_sign_positive();
        let mut n = self.numer() as u64 * other.numer() as u64;
        let mut d = self.denom() as u64 * other.denom() as u64;
        
        let gcd = n.gcd(d);
        n /= gcd;
        d /= gcd;
        
        debug_assert!(
            (64 - d.leading_zeros() - 1) + (64 - n.leading_zeros()) <= FRACTION_SIZE,
            "attempt to multiply with overflow"
        );
        
        r32::from_parts(s, n as u32, d as u32)
    }
}

impl Div for r32 {
    type Output = r32;

    fn div(self, other: r32) -> r32 {
        self * other.recip()
    }
}

impl Add for r32 {
    type Output = r32;
    
    fn add(self, other: r32) -> r32 {
        // self = a/b, other = c/d
        
        let selfsign = (self.signum().0 as i32).signum() as i64;
        let othersign = (other.signum().0 as i32).signum() as i64;
        
        // TODO prove this won't panic/can't overflow.
        // num = ad + bc
        let num =
            (self.numer() as i64 * selfsign) * other.denom() as i64
            + self.denom() as i64 * (other.numer() as i64 * othersign);
        // den = bd
        let mut den = self.denom() as u64 * other.denom() as u64;
        let s = num.is_negative();
        let mut num = num.abs() as u64;
        
        let gcd = num.gcd(den);
        num /= gcd;
        den /= gcd;
        
        debug_assert!(
            (64 - den.leading_zeros() - 1) + (64 - num.leading_zeros()) <= FRACTION_SIZE,
            "attempt to add with overflow"
        );
        
        r32::from_parts(s, num as u32, den as u32)
    }
}

impl Sub for r32 {
    type Output = r32;

    fn sub(self, other: r32) -> r32 {
        self + -other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn normalize() {
        assert_eq!(r32::from_parts(true, 0, 1).normalize(), r32(0));
        assert_eq!(r32::from_parts(false, 4, 2).normalize(), r32::from_parts(false, 2, 1));
    }
    
    #[test]
    fn neg() {
        assert_eq!((-r32(0)).0, SIGN_BIT);
        assert_eq!((-r32(SIGN_BIT)).0, 0);
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
    fn fract() {
        assert_eq!(r32(5).fract(), r32(0));
        assert_eq!(r32::from_parts(false, 5, 2).fract(), r32::from_parts(false, 1, 2));
    }
    
    #[test]
    fn recip() {
        assert_eq!(r32(5).recip(), r32::from_parts(false, 1, 5));
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
        let _ = r32(1 << 25) * r32(1 << 25);
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
        let _ = r32(1 << 25) + r32(1 << 25);
    }
    
    #[test]
    fn debug() {
        let r = r32::from_parts(true, 0, 1);
        assert_eq!(format!("{:?}", r), "-0/1");
    }
    
    #[test]
    fn display() {
        let r = r32::from_parts(true, 3, 2);
        assert_eq!(format!("{}", r), "-3/2");
        let r = r32::from_parts(false, 0, 1);
        assert_eq!(format!("{}", r), "0");
    }
}
