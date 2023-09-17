use super::algebra_tools::exponentiate;

use std::cmp::Ordering;
use std::{fmt, ops};

fn gcd(a: u32, b: u32) -> u32 {
    let mut x = a;
    let mut y = b;
    let mut r = x % y;
    while r > 0 {
        x = y;
        y = r;
        r = x % y;
    }
    y
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Rational { n: i32, d: u32 }

impl Rational {
    pub const ZERO: Rational = Rational { n: 0, d: 1 };
    pub const ONE: Rational = Rational { n: 1, d: 1 };

    pub fn numerator(&self) -> i32 {
        self.n
    }
    pub fn denominator(&self) -> u32 {
        self.d
    }

    pub fn new(num: i32, den: i32) -> Rational {
        let (n, d): (i32, u32) =
            if den < 0 {(num * -1, (den * -1) as u32)}
            else { (num, den as u32) };
        let gcd_nd: u32 = gcd(n.unsigned_abs(), d);
        return Rational { n: n / (gcd_nd as i32), d: d / gcd_nd };
    }
    pub fn abs(self) -> Rational {
        Rational { n: self.n.abs(), d: self.d }
    }
    pub fn floor(self) -> i32 {
        if self.n.is_positive() || self.n % self.d as i32 == 0 {
            self.n / self.d as i32
        }
        else {
            (self.n / self.d as i32) - 1
        }
    }
    pub fn ceil(self) -> i32 {
        if self.n.is_negative() || self.n % self.d as i32 == 0 {
            self.n / self.d as i32
        }
        else {
            (self.n / self.d as i32) + 1
        }
    }
    pub fn round_to_zero(self) -> i32 {
        self.n / self.d as i32
    }
    pub fn round_from_zero(self) -> i32 {
        if self.n % self.d as i32 == 0 {
            self.n / self.d as i32
        }
        else {
            (self.n / self.d as i32) + if self.n.is_negative() {-1} else {1}
        }
    }
}

impl From<i32> for Rational {
    fn from(integer: i32) -> Self {
        Rational { n: integer, d: 1 }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 { f.write_fmt(format_args!("{}", self.n)) }
        else { f.write_fmt(format_args!("({}/{})", self.n, self.d)) }
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        let diff_n: i32 = (self.clone() - other.clone()).n;
        if diff_n < 0 { return Ordering::Less; }
        if diff_n > 0 { return Ordering::Greater; }
        Ordering::Equal
    }
}
impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ops::Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Rational::new(self.n * rhs.d as i32 + rhs.n * self.d as i32,
                      (self.d * rhs.d) as i32)
    }
}

impl ops::Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Rational {n: -self.n, d:self.d}
    }
}
impl ops::Sub for Rational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Rational::new(self.n * rhs.d as i32 - rhs.n * self.d as i32,
                      (self.d * rhs.d) as i32)
    }
}

impl ops::Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Rational::new(self.n * rhs.n, (self.d * rhs.d) as i32)
    }
}
impl ops::Div for Rational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Rational::new(self.n * rhs.d as i32, self.d as i32 * rhs.n)
    }
}

impl ops::BitXor<i32> for Rational {
    type Output = Self;
    fn bitxor(self, exp: i32) -> Self {
        exponentiate(if exp < 0 { Rational::ONE / self } else { self }, 
                     exp.unsigned_abs(), Rational::ONE)
    }
}