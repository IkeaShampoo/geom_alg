use super::algebra_tools::*;

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

impl AddIdentity for Rational {
    const ZERO: Rational = Rational { n: 0, d: 1 };
}
impl MulIdentity for Rational {
    const ONE: Rational = Rational { n: 1, d: 1 };
}

impl Rational {
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
    pub fn is_zero(&self) -> bool {
        self.numerator() == 0
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
impl ops::AddAssign for Rational {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
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
impl ops::SubAssign for Rational {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ops::Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Rational::new(self.n * rhs.n, (self.d * rhs.d) as i32)
    }
}
impl ops::MulAssign for Rational {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl ops::Div for Rational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Rational::new(self.n * rhs.d as i32, self.d as i32 * rhs.n)
    }
}
impl ops::DivAssign for Rational {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl ops::BitXor<i32> for Rational {
    type Output = Self;
    fn bitxor(self, exp: i32) -> Self::Output {
        let b = if exp < 0 { Rational::ONE / self } else { self };
        Rational { 
            n: exponentiate(b.n, exp.unsigned_abs()), 
            d: exponentiate(b.d, exp.unsigned_abs())
        }
    }
}

impl ops::BitXor<Rational> for Rational {
    type Output = Option<Self>;
    fn bitxor(self, exp: Rational) -> Self::Output {
        match exp.d {
            1 => Some(self ^ exp.n),
            exp_d => match (root_i64(self.n as i64, exp_d), root_u64(self.d as u64, exp_d)) {
                (Some(Ok(n_root)), Ok(d_root)) =>
                    Some(Rational { n: n_root as i32, d: d_root as u32, } ^ exp.n),
                _ => None,
            }
        }
    }
}

impl Rational {
    /// Simplifies a rational raised to another rational
    /// into some root of a rational, returned as (rational, root index).
    /// Returns None if the root does not exist
    pub fn simplify_exp(self, exp: Self) -> Option<(Self, u32)> {
        if exp.d == 1 {
            Some((self ^ exp.n, 1))
        }
        else if (self.n < 0) && (exp.d & 1 == 0) {
            None
        }
        else {
            let sign = self.n.signum();
            let mut root_index = exp.d;
            let (mut n, mut d): (u64, u64) = (self.n.unsigned_abs() as u64, self.d as u64);
            let mut unused_factors = 1;
            loop {
                match prime_factor(root_index) {
                    Some(index_factor) => {
                        root_index /= index_factor;
                        match (root_u64(n, index_factor), root_u64(d, index_factor)) {
                            (Ok(n_root), Ok(d_root)) => (n, d) = (n_root, d_root),
                            _ => unused_factors *= index_factor
                        }
                    }
                    None => break
                }
            }
            Some((Rational { n: sign * n as i32, d: d as u32, } ^ exp.n, 
                root_index * unused_factors))
        }
    }
}