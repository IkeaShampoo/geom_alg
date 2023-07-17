use std::cmp;
use std::fmt;
use std::ops;
use std::collections::VecDeque;

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

#[derive(Copy, Clone)]
pub struct Rational { n: i32, d: u32 }
const ZERO: Rational = Rational {n: 0, d: 1};
const ONE: Rational = Rational {n: 1, d: 1};

impl Rational {
    pub fn new(num: i32, den: i32) -> Rational {
        let (n, d): (i32, u32) =
            if den < 0 {(num * -1, (den * -1) as u32)}
            else { (num, den as u32) };
        let gcd_nd: u32 = gcd(n.abs() as u32, d);
        return Rational {n: n / (gcd_nd as i32), d: d / gcd_nd};
    }
}

impl From<i32> for Rational {
    fn from(integer: i32) -> Self {
        Rational {n: integer, d: 1 }
    }
}

impl ops::Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Rational::new(self.n * rhs.d as i32 + rhs.n * self.d as i32,
                      (self.d * rhs.d) as i32)
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

impl ops::Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Rational {n: -self.n, d:self.d}
    }
}

impl PartialEq<Self> for Rational {
    fn eq(&self, other: &Self) -> bool {
        (self.n == other.n) && (self.d == other.d)
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        let diff_n: i32 = (self.clone() - other.clone()).n;
        if diff_n < 0 { return Some(cmp::Ordering::Less); }
        if diff_n > 0 { return Some(cmp::Ordering::Greater); }
        Some(cmp::Ordering::Equal)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 { f.write_fmt(format_args!("{}", self.n)) }
        else { f.write_fmt(format_args!("({}/{})", self.n, self.d)) }
    }
}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct Exponential {
    b: Scalar, e: Rational
}

impl fmt::Display for Exponential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.e == ONE { Scalar::fmt(&self.b, f) }
        else { f.write_fmt(format_args!("({}^{})", self.b, self.e)) }
    }
}

#[derive(Clone, PartialEq, PartialOrd)]
pub enum Scalar {
    Variable(String),
    Rational(Rational),
    Sum(Rational, Vec<Scalar>),
    Product(Vec<Exponential>),
}

impl From<Scalar> for Exponential {
    fn from(value: Scalar) -> Self {
        Exponential {b: value, e: ONE}
    }
}

fn to_ordering(n: i32) -> cmp::Ordering {
    if n < 0 { cmp::Ordering::Less }
    else if n > 0 { cmp::Ordering::Greater }
    else { cmp::Ordering::Equal }
}

impl From<String> for Scalar {
    fn from(name: String) -> Self {
        Scalar::Variable(name)
    }
}

impl From<Rational> for Scalar {
    fn from(fraction: Rational) -> Self {
        Scalar::Rational(fraction)
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Rational(s) => Rational::fmt(s, f),
            Scalar::Variable(s) => f.write_str(s.as_str()),
            Scalar::Sum(r, s) => {
                f.write_str("(")?;
                if *r != ZERO {
                    Rational::fmt(r, f)?;
                    if s.len() > 0 { f.write_str(" + ")?; }
                }
                for i in 0..s.len() {
                    if i > 0 { f.write_str(" + ")?; }
                    Scalar::fmt(&s[i],f)?;
                }
                f.write_str(")")
            },
            Scalar::Product(s) => {
                f.write_str("(")?;
                for i in 0..s.len() {
                    if i > 0 { f.write_str(" * ")?; }
                    Exponential::fmt(&s[i], f)?;
                }
                f.write_str(")")
            }
        }
    }
}

impl ops::Add for Scalar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Scalar::Rational(lhs), Scalar::Rational(rhs)) => Scalar::Rational(lhs + rhs),
            (Scalar::Rational(lhs), Scalar::Sum(rhs_rat, rhs)) => Scalar::Sum(lhs + rhs_rat, rhs),
            (Scalar::Sum(lhs_rat, lhs), Scalar::Rational(rhs)) => Scalar::Sum(lhs_rat + rhs, lhs),
            (Scalar::Rational(lhs), rhs) => Scalar::Sum(lhs, vec![rhs]),
            (lhs, Scalar::Rational(rhs)) => Scalar::Sum(rhs, vec![lhs]),
            (Scalar::Sum(lhs_rat, lhs), Scalar::Sum(rhs_rat, rhs)) => {
                let mut lhs: VecDeque<Scalar> = VecDeque::from(lhs);
                let mut rhs: VecDeque<Scalar> = VecDeque::from(rhs);
                let mut new_terms: Vec<Scalar> =
                    Vec::with_capacity(lhs.len() + rhs.len());
                let mut lhs_next: Option<Scalar>;
                let mut rhs_next: Option<Scalar>;
                while {
                    lhs_next = lhs.pop_front();
                    rhs_next = rhs.pop_front();
                    lhs_next != None || rhs_next != None
                } {
                    if let (Some(lhs_next), Some(rhs_next)) = (lhs_next, rhs_next)  {
                        new_terms.push(if lhs_next < rhs_next {lhs_next} else {rhs_next});
                    }
                }
                new_terms.append(&mut Vec::from(lhs));
                new_terms.append(&mut Vec::from(rhs));
                Scalar::Sum(lhs_rat + rhs_rat, new_terms)
            }
            (lhs, rhs) => {
                let (rat, mut old_terms, new_term): (Rational, Vec<Scalar>, Scalar) =
                    if let Scalar::Sum(lhs_rat, lhs) = lhs { (lhs_rat, lhs, rhs) }
                    else if let Scalar::Sum(rhs_rat, rhs) = rhs { (rhs_rat, rhs, lhs) }
                    else { (ZERO, vec!{lhs}, rhs) };
                let mut i: usize = 0;
                let num_terms = old_terms.len();
                while {
                    if i < num_terms {old_terms[i] < new_term}
                    else {false}
                } {i += 1;}
                old_terms.insert(i, new_term);
                Scalar::Sum(rat, old_terms)
            }
        }
    }
}