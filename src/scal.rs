use std::ops;
use std::cmp;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::convert;

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

#[derive(Clone, PartialEq, PartialOrd)]
pub struct Exponential {
    b: Scalar, e: Rational
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

impl ops::Add for Scalar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Scalar::Rational(lhs), Scalar::Rational(rhs)) => Scalar::Rational(lhs + rhs),
            (Scalar::Rational(lhs), Scalar::Sum(rhs_rat, rhs)) => Scalar::Sum(lhs + rhs_rat, rhs),
            (Scalar::Sum(lhs_rat, lhs), Scalar::Rational(rhs)) => Scalar::Sum(lhs_rat + rhs, lhs),
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
                /*
                let (old_terms, new_term): (Vec<Scalar>, Scalar) =
                    if let Scalar::Sum(lhs) = lhs { (lhs, rhs) }
                    else if let Scalar::Sum(rhs) = rhs { (rhs, lhs) }
                    else { (vec!{lhs}, rhs) };
                let mut new_terms: Vec<Scalar>;

                 */
                lhs
            }
        }
    }
}
/*

    impl Add for SimpleScalar {
        type Output = Self;

        fn add(&self, rhs: &Self) -> Self::Output {
            if let SimpleScalar::Rational(n, d) = self {
                if let SimpleScalar::Rational(rhs_n, rhs_d) = rhs {
                    let new_n: i32 = n * rhs_d as i32 + rhs_n * d as i32;
                    let new_d: u32 = d * rhs_d;
                    let gcd: u32 = gcd(*d, *rhs_d);
                    SimpleScalar::Rational { n: (new_n / gcd as i32), d: new_d / gcd }
                }
            }
            if let SimpleScalar::Sum(terms) = self {
                if let SimpleScalar::Sum(rhs_terms) = self {
                    let mut rhs_copy = rhs_terms.to_vec();
                    SimpleScalar::Sum(terms.to_vec().append(&mut rhs_copy));
                }
            }
            SimpleScalar::Sum(vec![*self, *rhs])
        }
    }

 */