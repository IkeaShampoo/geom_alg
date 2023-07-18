use std::cmp;
use std::cmp::Ordering;
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

#[derive(Copy, Clone, Eq, Ord, Debug)]
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

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 { f.write_fmt(format_args!("{}", self.n)) }
        else { f.write_fmt(format_args!("({}/{})", self.n, self.d)) }
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

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct Exponential {
    b: Scalar, e: Rational
}

impl From<Scalar> for Exponential {
    fn from(scalar: Scalar) -> Exponential {
        Exponential{b: scalar, e: ONE}
    }
}

impl fmt::Display for Exponential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.e == ONE { Scalar::fmt(&self.b, f) }
        else { f.write_fmt(format_args!("({}^{})", self.b, self.e)) }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub enum Scalar {
    Variable(String),
    Rational(Rational),
    Sum(Vec<Scalar>),
    Product(Vec<Exponential>),
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
            Scalar::Sum(s) => {
                f.write_str("(")?;
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

impl PartialEq<Rational> for Scalar {
    fn eq(&self, other: &Rational) -> bool {
        *self == Scalar::from(*other)
    }
}

impl ops::Add for Scalar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Scalar::Rational(lhs), Scalar::Rational(rhs)) => Scalar::Rational(lhs + rhs),
            (Scalar::Sum(lhs), Scalar::Sum(rhs)) => {
                let mut lhs: VecDeque<Scalar> = VecDeque::from(lhs);
                let mut rhs: VecDeque<Scalar> = VecDeque::from(rhs);
                let mut new_terms: Vec<Scalar> = Vec::with_capacity(lhs.len() + rhs.len());
                while let (Some(lhs_next), Some(rhs_next)) = (lhs.front(), rhs.front()) {
                    new_terms.push(
                        match (lhs_next, rhs_next) {
                            // Each sum will only have up to one rational term
                            (Scalar::Rational(lhs_next), Scalar::Rational(rhs_next)) => {
                                let rat_sum = *lhs_next + *rhs_next;
                                lhs.pop_front();
                                rhs.pop_front();
                                Scalar::Rational(rat_sum)
                            }
                            (lhs_next, rhs_next) =>
                                if *lhs_next < *rhs_next {lhs.pop_front()} else {rhs.pop_front()}
                                    .expect("term should not be empty")
                        }
                    );
                }
                new_terms.append(&mut Vec::from(lhs));
                new_terms.append(&mut Vec::from(rhs));
                println!("Final sum: {:?}", new_terms);
                Scalar::Sum(new_terms)
            }
            (lhs, rhs) => {
                let (old_terms, new_term): (Vec<Scalar>, Scalar) =
                    if let Scalar::Sum(lhs) = lhs { (lhs, rhs) }
                    else if let Scalar::Sum(rhs) = rhs { (rhs, lhs) }
                    else { (vec!{lhs}, rhs) };
                Scalar::Sum(old_terms) + Scalar::Sum(vec![new_term])
            }
        }
    }
}

impl ops::Sub for Scalar {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        if let (Scalar::Rational(lhs), Scalar::Rational(rhs)) = (&self, &rhs) {
            if *lhs == -rhs.clone() { return Scalar::from(ZERO); }
        }
        self + -rhs
    }
}

impl ops::Mul for Scalar {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        const EXPECT_ERR: &str = "term should not be empty";
        match (self, rhs) {
            (Scalar::Product(lhs), Scalar::Product(rhs)) => {
                let mut lhs: VecDeque<Exponential> = VecDeque::from(lhs);
                let mut rhs: VecDeque<Exponential> = VecDeque::from(rhs);
                let mut new_factors: Vec<Exponential> = Vec::with_capacity(lhs.len() + rhs.len());
                while let (Some(lhs_next), Some(rhs_next)) = (lhs.front(), rhs.front()) {
                    let order: Ordering = (lhs_next.b).cmp(&rhs_next.b);
                    new_factors.push(
                        if order == Ordering::Less {lhs.pop_front().expect(EXPECT_ERR)}
                        else if order == Ordering::Greater {rhs.pop_front().expect(EXPECT_ERR)}
                        else {
                            let exponent = lhs_next.e + rhs_next.e;
                            lhs.pop_front();
                            Exponential{b: rhs.pop_front().expect(EXPECT_ERR).b, e: exponent}
                        }
                    );
                }
                new_factors.append(&mut Vec::from(lhs));
                new_factors.append(&mut Vec::from(rhs));
                Scalar::Product(new_factors)
            }
            (lhs, rhs) => {
                if lhs == ZERO || rhs == ZERO { return Scalar::from(ZERO); }
                if lhs == ONE { return rhs; }
                if rhs == ONE { return lhs; }
                let (old_factors, new_factor): (Vec<Exponential>, Scalar) =
                    if let Scalar::Product(lhs) = lhs { (lhs, rhs) }
                    else if let Scalar::Product(rhs) = rhs { (rhs, lhs) }
                    else { (vec![Exponential::from(lhs)], rhs) };
                Scalar::Product(old_factors) * Scalar::Product(vec![Exponential::from(new_factor)])
            }
        }
    }
}

impl ops::Div for Scalar {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        todo!()
    }
}

impl ops::Neg for Scalar {
    type Output = Self;
    fn neg(self) -> Self {
        Scalar::from(-ONE) * self
    }
}

