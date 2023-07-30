use std::cmp;
use std::fmt;
use std::ops;
use std::collections::VecDeque;
use std::collections::BTreeSet;

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

fn iterative_mul<T: ops::Mul<Output = T> + Copy>(identity: T, base: T, exp: u32) -> T {
    let mut product = identity;
    let mut base_to_i = identity;
    for i in 0..u32::BITS {
        if ((exp >> i) & 1) == 1 {
            product = product * base_to_i;
        }
        base_to_i = base_to_i * base;
    }
    product
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
        let gcd_nd: u32 = gcd(n.unsigned_abs(), d);
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

impl ops::BitXor<i32> for Rational {
    type Output = Self;
    fn bitxor(self, exp: i32) -> Self {
        let mut base = self;
        if exp < 0 {
            base = ONE / base;
        }
        iterative_mul(ONE, base, exp.unsigned_abs())
    }
}



#[derive(Clone, Eq, Ord, Debug)]
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

impl PartialEq<Self> for Exponential {
    fn eq(&self, other: &Self) -> bool {
        (self.b == other.b) && (self.e == other.e)
    }
}

impl PartialOrd for Exponential {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        let (Exponential{b: lb, e: le}, Exponential{b: rb, e: re}) = (self, other);
        let b_order = lb.partial_cmp(rb);
        if b_order == Some(cmp::Ordering::Equal) {le.partial_cmp(re)}
        else {b_order}
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

impl From<Exponential> for Scalar {
    fn from(exponential: Exponential) -> Self {
        if exponential.e == ONE { exponential.b }
        else { Scalar::Product(vec![exponential]) }
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
                    match (lhs_next, rhs_next) {
                        // Each sum will only have up to one rational term
                        (&Scalar::Rational(lhs_next), &Scalar::Rational(rhs_next)) => {
                            lhs.pop_front();
                            rhs.pop_front();
                            let rat_sum = lhs_next + rhs_next;
                            if rat_sum != ZERO {
                                new_terms.push(Scalar::Rational(rat_sum));
                            }
                        }
                        (lhs_next, rhs_next) => {
                            new_terms.push(
                                if *lhs_next < *rhs_next {lhs.pop_front()}
                                else {rhs.pop_front()}
                                    .expect("term should not be empty")
                            );
                        }
                    }
                }
                new_terms.append(&mut Vec::from(lhs));
                new_terms.append(&mut Vec::from(rhs));
                //println!("Final sum: {:?}", new_terms);
                Scalar::Sum(new_terms)
            }
            (lhs, rhs) => {
                if lhs == ZERO { return rhs; }
                if rhs == ZERO { return lhs; }
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
        if self == rhs { Scalar::from(ZERO) }
        else { self + -rhs }
    }
}

impl ops::Mul for Scalar {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        const EXPECT_ERR: &str = "term should not be empty";

        match (self, rhs) {
            (Scalar::Rational(lhs), Scalar::Rational(rhs)) => Scalar::Rational(lhs * rhs),
            (Scalar::Product(lhs), Scalar::Product(rhs)) => {
                let mut lhs: VecDeque<Exponential> = VecDeque::from(lhs);
                let mut rhs: VecDeque<Exponential> = VecDeque::from(rhs);
                let mut new_factors: Vec<Exponential> = Vec::with_capacity(lhs.len() + rhs.len());
                while let (Some(lhs_next), Some(rhs_next)) = (lhs.front(), rhs.front()) {
                    let order = (lhs_next.b).cmp(&rhs_next.b);
                    match order {
                        cmp::Ordering::Equal => {
                            let exp = lhs_next.e + rhs_next.e;
                            let base = rhs.pop_front().expect(EXPECT_ERR).b;
                            lhs.pop_front();
                            if exp != ZERO {
                                new_factors.push(Exponential {b: base, e: exp });
                            }
                        }
                        cmp::Ordering::Less => new_factors.push(lhs.pop_front().expect(EXPECT_ERR)),
                        cmp::Ordering::Greater => new_factors.push(rhs.pop_front().expect(EXPECT_ERR))
                    }
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

impl ops::BitXor<Rational> for Scalar {
    type Output = Self;
    fn bitxor(self, exp: Rational) -> Self::Output {
        match self {
            Scalar::Rational(x) => {
                if exp.d == 1 { Scalar::Rational(x ^ exp.n) }
                else { Scalar::Product(vec![Exponential{b: Scalar::Rational(x), e: exp}]) }
            }
            Scalar::Product(mut factors) => {
                for factor in &mut factors {
                    (*factor).e = (*factor).e * exp;
                }
                Scalar::Product(factors)
            }
            x => Scalar::Product(vec![Exponential{b: x, e: exp}])
        }
    }
}

impl ops::Div for Scalar {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * (rhs ^ -ONE)
    }
}

impl ops::Neg for Scalar {
    type Output = Self;
    fn neg(self) -> Self {
        Scalar::Rational(-ONE) * self
    }
}

#[derive(Copy, Clone, PartialEq)]
struct ExprCost {
    c: usize,
    v: usize
}

impl ExprCost {
    fn zero() -> ExprCost {
        ExprCost { c: 0, v: 0}
    }

    fn new(ops_count: usize, is_constant: bool) -> ExprCost {
        if is_constant { ExprCost { c: ops_count, v: 0 }}
        else { ExprCost { v: ops_count, c: 0 } }
    }
}

impl ops::Add for ExprCost {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        ExprCost { c: self.c + rhs.c, v: self.v + rhs.v}
    }
}

impl PartialOrd for ExprCost {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        let v_order = self.v.cmp(&other.v);
        if v_order == cmp::Ordering::Equal { Some(self.c.cmp(&other.c)) }
        else { Some(v_order) }
    }
}

/* to delete

fn round_up_2s(n: usize) -> usize {
    for i in (0..usize::BITS).rev() {
        let mask: usize = 1 << i;
        if n & mask != 0 {
            return if n & !mask == 0 { 1 << i } else { 1 << (i + 1) }
        }
    }
    0
}

 */

fn merge_scalars(arguments: Vec<Scalar>, merge: fn(Scalar, Scalar) -> Scalar, identity: &Scalar)
                 -> Scalar {
    if arguments.len() == 0 {
        identity.clone()
    }
    else {
        let left_len = arguments.len() >> 1;
        let mut left: Vec<Scalar> = Vec::with_capacity(left_len);
        let mut right: Vec<Scalar> = Vec::with_capacity(arguments.len() - left_len);
        for arg in arguments {
            if left.len() < left_len { &mut left } else { &mut right }
                .push(arg);
        }
        merge(merge_scalars(left, merge, identity), merge_scalars(right, merge, identity))
    }
}

impl Scalar {
    fn replace(self, to_replace: &Scalar, replace_with: &Scalar) -> Scalar {
        match self {
            Scalar::Sum(terms) => {
                let mut new_terms: Vec<Scalar> = Vec::with_capacity(terms.len());
                for term in terms {
                    new_terms.push(term.replace(to_replace, replace_with));
                }
                merge_scalars(new_terms, |x, y| x + y, &Scalar::from(ZERO))
            }
            Scalar::Product(factors) => {
                let mut new_factors: Vec<Scalar> = Vec::with_capacity(factors.len());
                for Exponential { b: factor_base, e: factor_exp } in factors {
                    new_factors.push(Scalar::from(Exponential {
                        b: factor_base.replace(to_replace, replace_with),
                        e: factor_exp
                    }));
                }
                merge_scalars(new_factors, |x, y| x * y, &Scalar::from(ONE))
            }
            x => {
                if x == *to_replace { replace_with.clone() }
                else { x }
            }
        }
    }

    fn is_constant(&self) -> bool {
        match self {
            Scalar::Rational(_) => true,
            Scalar::Variable(x) => *x == x.to_uppercase(),
            Scalar::Sum(terms) => {
                for term in terms {
                    if !term.is_constant() {
                        return false;
                    }
                }
                true
            }
            Scalar::Product(factors) => {
                for factor in factors {
                    if !factor.b.is_constant() {
                        return false;
                    }
                }
                true
            }
        }
    }

    fn cost(&self) -> ExprCost {
        match self {
            Scalar::Sum(terms) => {
                let mut total_cost: ExprCost = ExprCost::zero();
                for term in terms {
                    total_cost = total_cost + term.cost();
                }
                total_cost + ExprCost::new(terms.len() - 1, self.is_constant())
            }
            Scalar::Product(factors) => {
                let mut total_cost: ExprCost = ExprCost::zero();
                for factor in factors {
                    total_cost = total_cost + factor.b.cost();
                }
                total_cost + ExprCost::new(factors.len() - 1, self.is_constant())
            }
            _ => ExprCost { c: 0, v: 0}
        }
    }

    pub fn simplified(&self) -> Scalar {
        let swap = Scalar::Variable(String::from(char::from_u32(0x1F431).unwrap()));

        const RULES: [fn(&Scalar) -> Option<Scalar>; 1] = [
            { |x| None }
        ];

        struct Simplifier {
            simplest: (Scalar, ExprCost),
            territory: BTreeSet<Scalar>,
            queue: Vec<Scalar>
        }

        impl Simplifier {
            fn discover(&mut self, expr: Scalar) {
                if !self.territory.contains(&expr) {
                    let expr_cost = expr.cost();
                    if expr_cost < self.simplest.1 {
                        self.simplest = (expr.clone(), expr_cost)
                    }
                    self.territory.insert(expr.clone());
                    self.queue.push(expr);
                }
            }

            fn derive (&mut self, expr: Scalar, expr_placeholder: &Scalar, full_template: Scalar) {
                for rule in &RULES {
                    if let Some(expr_trans) = (*rule)(&expr) {
                        self.discover(full_template.clone().replace(expr_placeholder, &expr_trans));
                    }
                }

            }
        }

        let mut simplifier = Simplifier {
            simplest: (self.clone(), self.cost()),
            territory: BTreeSet::new(),
            queue: Vec::new()
        };

        simplifier.discover(self.clone());
        while let Some(next_expr) = simplifier.queue.pop() {
            simplifier.derive(next_expr, &swap, swap.clone());
        }
        simplifier.simplest.0
    }
}