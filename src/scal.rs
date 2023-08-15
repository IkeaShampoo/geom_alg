use super::algebra_tools::*;

use std::cmp::Ordering;
use std::{fmt, mem, ops};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

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

#[derive(Copy, Clone, PartialEq, Eq, Ord, Debug)]
pub struct Rational { n: i32, d: u32 }
pub const ZERO: Rational = Rational { n: 0, d: 1 };
pub const ONE: Rational = Rational { n: 1, d: 1 };

impl Rational {
    pub fn new(num: i32, den: i32) -> Rational {
        let (n, d): (i32, u32) =
            if den < 0 {(num * -1, (den * -1) as u32)}
            else { (num, den as u32) };
        let gcd_nd: u32 = gcd(n.unsigned_abs(), d);
        return Rational { n: n / (gcd_nd as i32), d: d / gcd_nd };
    }
    pub fn min(a: Rational, b: Rational) -> Rational {
        if a < b { a }
        else { b }
    }
    pub fn max(a: Rational, b: Rational) -> Rational {
        if a > b { a }
        else { b }
    }
    pub fn abs(self) -> Rational {
        Rational { n: self.n.abs(), d: self.d }
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

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let diff_n: i32 = (self.clone() - other.clone()).n;
        if diff_n < 0 { return Some(Ordering::Less); }
        if diff_n > 0 { return Some(Ordering::Greater); }
        Some(Ordering::Equal)
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
        let mut base = self;
        if exp < 0 {
            base = ONE / base;
        }
        exponentiate(ONE, base, exp.unsigned_abs())
    }
}



#[derive(Clone, Eq, Ord, Debug)]
pub struct Exponential {
    b: Scalar, e: Rational
}

impl From<Scalar> for Exponential {
    fn from(scalar: Scalar) -> Exponential {
        Exponential { b: scalar, e: ONE }
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (Exponential { b: lb, e: le },
             Exponential { b: rb, e: re }) = (self, other);
        let b_order = lb.partial_cmp(rb);
        if b_order == Some(Ordering::Equal) { le.partial_cmp(re) }
        else { b_order }
    }
}



#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub enum Scalar {
    Variable(String),
    Rational(Rational),
    Sum(Vec<Scalar>),
    Product(Vec<Exponential>),
}
pub const S_ZERO: Scalar = Scalar::Rational(ZERO);
pub const S_ONE: Scalar = Scalar::Rational(ONE);

impl From<String> for Scalar {
    fn from(name: String) -> Self {
        Scalar::Variable(name)
    }
}
impl From<&str> for Scalar {
    fn from(name: &str) -> Self {
        Scalar::Variable(String::from(name))
    }
}
impl From<Rational> for Scalar {
    fn from(fraction: Rational) -> Self {
        Scalar::Rational(fraction)
    }
}
impl From<Exponential> for Scalar {
    fn from(exponential: Exponential) -> Self {
        match exponential.e {
            ZERO => S_ONE,
            ONE => exponential.b,
            _ => if exponential.b == ONE { S_ONE }
                 else { Scalar::Product(vec![exponential]) }
        }
    }
}
impl TryInto<Vec<Scalar>> for Scalar {
    type Error = &'static str;
    fn try_into(self) -> Result<Vec<Scalar>, Self::Error> {
        match self {
            Scalar::Sum(terms) => Ok(terms),
            _ => Err("This is not a Scalar::Sum")
        }
    }
}
impl TryInto<Vec<Exponential>> for Scalar {
    type Error = &'static str;
    fn try_into(self) -> Result<Vec<Exponential>, Self::Error> {
        match self {
            Scalar::Product(factors) => Ok(factors),
            _ => Err("This is not a Scalar::Product")
        }
    }
}

impl PartialEq<Rational> for Scalar {
    fn eq(&self, other: &Rational) -> bool {
        *self == Scalar::from(*other)
    }
}

fn make_stupid_workaround_scalar(x: &mut Scalar) -> Scalar {
    let mut stupid_workaround_swap_thing = S_ONE;
    mem::swap(x, &mut stupid_workaround_swap_thing);
    stupid_workaround_swap_thing
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
                                if *lhs_next < *rhs_next { lhs.pop_front() }
                                else { rhs.pop_front() }
                                    .expect("term should not be empty")
                            );
                        }
                    }
                }
                new_terms.append(&mut Vec::from(lhs));
                new_terms.append(&mut Vec::from(rhs));
                //println!("Final sum: {:?}", new_terms);
                Scalar::Sum(new_terms).correct_form()
            }
            (lhs, rhs) => {
                if lhs == ZERO { return rhs; }
                if rhs == ZERO { return lhs; }
                let (old_terms, new_term): (Vec<Scalar>, Scalar) =
                    if let Scalar::Sum(lhs) = lhs { (lhs, rhs) }
                    else if let Scalar::Sum(rhs) = rhs { (rhs, lhs) }
                    else { (vec![lhs], rhs) };
                Scalar::Sum(old_terms) + Scalar::Sum(vec![new_term])
            }
        }
    }
}
impl ops::AddAssign for Scalar {
    fn add_assign(&mut self, rhs: Self) {
        *self = make_stupid_workaround_scalar(self) + rhs;
    }
}

impl ops::Mul for Scalar {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        const EXPECT_ERR: &str = "Factor does not exist";

        match (self, rhs) {
            (Scalar::Rational(lhs), Scalar::Rational(rhs)) => Scalar::Rational(lhs * rhs),
            (Scalar::Product(lhs), Scalar::Product(rhs)) => {
                let mut lhs: VecDeque<Exponential> = VecDeque::from(lhs);
                let mut rhs: VecDeque<Exponential> = VecDeque::from(rhs);
                let mut new_factors: Vec<Exponential> = Vec::with_capacity(lhs.len() + rhs.len());
                while let (Some(lhs_next), Some(rhs_next)) = (lhs.front(), rhs.front()) {
                    let order = (lhs_next.b).cmp(&rhs_next.b);
                    match order {
                        Ordering::Equal => {
                            let exp = lhs_next.e + rhs_next.e;
                            let base = rhs.pop_front().expect(EXPECT_ERR).b;
                            lhs.pop_front();
                            if exp != ZERO {
                                new_factors.push(Exponential { b: base, e: exp });
                            }
                        }
                        Ordering::Less => new_factors.push(lhs.pop_front().expect(EXPECT_ERR)),
                        Ordering::Greater => new_factors.push(rhs.pop_front().expect(EXPECT_ERR))
                    }
                }
                new_factors.append(&mut Vec::from(lhs));
                new_factors.append(&mut Vec::from(rhs));
                Scalar::Product(new_factors).correct_form()
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
impl ops::MulAssign for Scalar {
    fn mul_assign(&mut self, rhs: Self) {
        *self = make_stupid_workaround_scalar(self) * rhs;
    }
}

impl ops::BitXor<Rational> for Scalar {
    type Output = Self;
    fn bitxor(self, exp: Rational) -> Self::Output {
        match exp {
            ZERO => Scalar::from(ONE),
            ONE => self,
            _ => match self {
                /*
                Scalar::Rational(x) => {
                    if exp.d == 1 { Scalar::Rational(x ^ exp.n) }
                    else { Scalar::Product(vec![Exponential { b: Scalar::Rational(x), e: exp }]) }
                }
                 */
                Scalar::Product(mut factors) => {
                    for factor in &mut factors {
                        (*factor).e = (*factor).e * exp;
                    }
                    Scalar::Product(factors)
                }
                x => Scalar::Product(vec![Exponential { b: x, e: exp }])
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
impl ops::SubAssign for Scalar {
    fn sub_assign(&mut self, rhs: Self) {
        *self = make_stupid_workaround_scalar(self) - rhs;
    }
}
impl ops::Neg for Scalar {
    type Output = Self;
    fn neg(self) -> Self {
        Scalar::Rational(-ONE) * self
    }
}
impl ops::Div for Scalar {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.mul_inv()
    }
}
impl ops::DivAssign for Scalar {
    fn div_assign(&mut self, rhs: Self) {
        *self = make_stupid_workaround_scalar(self) / rhs;
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let v_order = self.v.cmp(&other.v);
        if v_order == Ordering::Equal { Some(self.c.cmp(&other.c)) }
        else { Some(v_order) }
    }
}

const RULES: [fn(&Scalar) -> Vec<Scalar>; 3] = [
    // Factorization
    |x| match x {
        Scalar::Sum(terms) => {
            #[derive(Copy, Clone)]
            struct FactorIndex {
                t_idx: usize,
                f_idx: usize,
                exp: Rational
            }
            type FactorIndexer = BTreeMap<Scalar, Vec<FactorIndex>>;

            fn index_factor(factors_indices: &mut FactorIndexer,
                            factor_base: &Scalar, factor_exp: Rational,
                            term_index: usize, factor_index: usize) {
                let index = FactorIndex { t_idx: term_index, f_idx: factor_index, exp: factor_exp };
                match factors_indices.get_mut(factor_base) {
                    Some(factor_indices) => factor_indices.push(index),
                    None => { factors_indices.insert(factor_base.clone(), vec![index]); }
                }
            }

            let mut factors_indices: FactorIndexer = BTreeMap::new();
            for i in 0..terms.len() {
                match &terms[i] {
                    Scalar::Product(factors) =>
                        for j in 0..factors.len() {
                            let factor = &factors[j];
                            index_factor(&mut factors_indices, &factor.b, factor.e, i, j);
                        }
                    x => index_factor(&mut factors_indices, &x, ONE, i, 0)
                }
            }

            let mut factorizations: Vec<Scalar> = Vec::new();
            for (factor, indices) in factors_indices {
                for i in 0..(indices.len() - 1) {

                    fn remove_factor(term: Scalar, factor_index: usize, exp: Rational) -> Scalar {
                        match term {
                            Scalar::Product(mut factors) => {
                                if factors[factor_index].e == exp {
                                    factors.remove(factor_index);
                                }
                                else {
                                    factors[factor_index].e = factors[factor_index].e - exp;
                                }
                                Scalar::Product(factors).correct_form()
                            }
                            _ => S_ONE
                        }
                    }

                    let FactorIndex { t_idx: t_idx1, f_idx: f_idx1, exp: exp1} = indices[i];
                    for &FactorIndex { t_idx: t_idx2, f_idx: f_idx2, exp: exp2} in
                        &indices[(i + 1)..] {
                        if (exp1.n < 0) == (exp2.n < 0) {
                            let exp: Rational = if exp1.abs() < exp2.abs() { exp1 }
                                                else { exp2 };
                            let mut new_terms: Vec<Scalar> = terms.clone();
                            let term1 = new_terms.remove(t_idx1);
                            let term2 = new_terms.remove(
                                if t_idx2 < t_idx1 { t_idx2 }
                                else { t_idx2 - 1 }
                            );

                            factorizations.push(Scalar::Sum(new_terms).correct_form() +
                                                (factor.clone() ^ exp) *
                                                (remove_factor(term1, f_idx1, exp) +
                                                 remove_factor(term2, f_idx2, exp)));
                        }
                    }
                }
            }
            factorizations
        }
        _ => Vec::new()
    },

    // Distribution and combination of rational-based exponentials
    |x| match x {
        Scalar::Product(factors) => {
            let mut distributions: Vec<Scalar> = Vec::new();
            for index1 in 0..factors.len() {
                for index2 in 0..factors.len() {
                    // Combination of rational-based exponentials
                    if let (Exponential { b: Scalar::Rational(b1), e: e1 },
                            Exponential { b: Scalar::Rational(b2), e: e2 }) =
                           (&factors[index1], &factors[index2]) {
                        let e_ratio = *e2 / *e1;
                        if index1 != index2 && e_ratio.d == 1 {
                            let (b1, b2) = (*b1, *b2);
                            let e1 = *e1;
                            let mut other_factors = factors.clone();
                            other_factors.remove(index1);
                            other_factors.remove(if index2 < index1 { index2 } else {index2 - 1 });
                            distributions.push(Scalar::Product(other_factors).correct_form() *
                                               (Scalar::Rational(b1 * (b2 ^ e_ratio.n)) ^ e1));
                        }
                    }
                    // Distribution
                    else if let Scalar::Sum(_) = &(factors[index2].b) {
                        fn distribute(distributions: &mut Vec<Scalar>, other_factors: Scalar,
                                      terms: Vec<Scalar>, distributed: Scalar) {
                            for i in 0..terms.len() {
                                let mut other_terms = terms.clone();
                                let term = other_terms.remove(i);
                                distributions.push(other_factors.clone() * (
                                    distributed.clone() * term +
                                    distributed.clone() * Scalar::Sum(other_terms).correct_form()
                                ));
                            }
                        }

                        // distribute factor into another factor
                        if factors[index1].e == factors[index2].e && index1 != index2 {
                            let mut other_factors: Vec<Exponential> = factors.clone();
                            let distributed: Scalar = other_factors.remove(index1).b;
                            let terms: Vec<Scalar> = match other_factors.remove(
                                if index2 < index1 { index2 }
                                else { index2 - 1 }
                            ).b {
                                Scalar::Sum(terms) => terms,
                                not_a_sum => panic!("{not_a_sum} is not a sum")
                            };

                            distribute(&mut distributions,
                                       Scalar::Product(other_factors).correct_form(),
                                       terms, distributed);
                        }

                        // distribute factor into itself
                        if (if index1 == index2 { factors[index2].e - factors[index1].e }
                            else { factors[index2].e }) >= ONE {

                            fn remove_factors(factors: &mut Vec<Exponential>,
                                              index1: usize, exp1: Rational,
                                              mut index2: usize, exp2: Rational) {
                                let new_exp1 = factors[index1].e - exp1;
                                if new_exp1 == ZERO {
                                    factors.remove(index1);
                                    if index2 > index1 {
                                        index2 -= 1;
                                    }
                                }
                                else { factors[index1].e = new_exp1 }

                                let new_exp2 = factors[index2].e - exp2;
                                if new_exp2 == ZERO { factors.remove(index2); }
                                else { factors[index2].e = new_exp2 }
                            }

                            let mut new_factors: Vec<Exponential> = factors.clone();

                        }
                    }
                }
            }
            distributions
        }
        _ => Vec::new()
    },

    // (Not Started) Exponentiation of rational-based exponentials
    |x| match x {
        Scalar::Product(factors) => {
            Vec::new()
        }
        _ => Vec::new()
    }
];

struct Simplifier {
    simplest: (Scalar, ExprCost),
    territory: BTreeSet<Scalar>,
    queue: Vec<Scalar>
}

impl Simplifier {
    fn discover(&mut self, expr: Scalar) {
        //println!("{}", expr);    // TEST

        if !self.territory.contains(&expr) {
            let expr_cost = expr.cost();
            if expr_cost < self.simplest.1 {
                self.simplest = (expr.clone(), expr_cost)
            }
            self.territory.insert(expr.clone());
            self.queue.push(expr);
        }
    }

    fn derive(expr: &Scalar) -> Vec<Scalar> {
        let mut transformed: Vec<Scalar> = Vec::new();
        for rule in &RULES {
            transformed.append(&mut (*rule)(expr));
        }
        match expr {
            Scalar::Sum(terms) => {
                for i in 0..terms.len() {
                    for trans_term in Simplifier::derive(&terms[i]) {
                        let mut new_terms: Vec<Scalar> = terms.clone();
                        new_terms.remove(i);
                        transformed.push(Scalar::Sum(new_terms).correct_form() + trans_term);
                    }
                }
            }
            Scalar::Product(factors) => {
                for i in 0..factors.len() {
                    for trans_factor in Simplifier::derive(&factors[i].b) {
                        let mut new_factors: Vec<Exponential> = factors.clone();
                        new_factors.remove(i);
                        transformed.push(Scalar::Product(new_factors).correct_form() *
                                         (trans_factor ^ factors[i].e));
                    }
                }
            }
            _ => {}
        }
        transformed
    }
}



#[inline(always)]
fn add_all(terms: Vec<Scalar>) -> Scalar {
    merge_all(terms, |x, y| x + y, &Scalar::from(ZERO))
}
#[inline(always)]
fn mul_all(factors: Vec<Scalar>) -> Scalar {
    merge_all(factors, |x, y| x * y, &Scalar::from(ONE))
}
fn mul_exp(factors: Vec<Exponential>) -> Scalar {
    let mut new_factors: Vec<Scalar> = Vec::with_capacity(factors.len());
    for factor in factors {
        new_factors.push(Scalar::from(factor));
    }
    mul_all(new_factors)
}

impl Scalar {
    pub fn mul_inv(self) -> Self {
        self ^ -ONE
    }

    fn replace(self, to_replace: &Scalar, replace_with: &Scalar) -> Scalar {
        match self {
            Scalar::Sum(terms) => {
                let mut new_terms: Vec<Scalar> = Vec::with_capacity(terms.len());
                for term in terms {
                    new_terms.push(term.replace(to_replace, replace_with));
                }
                add_all(new_terms)
            }
            Scalar::Product(factors) => {
                let mut new_factors: Vec<Scalar> = Vec::with_capacity(factors.len());
                for Exponential { b: factor_base, e: factor_exp } in factors {
                    new_factors.push(factor_base.replace(to_replace, replace_with) ^ factor_exp);
                }
                mul_all(new_factors)
            }
            x => {
                if x == *to_replace { replace_with.clone() }
                else { x }
            }
        }
    }

    fn replace_all(self, replacements: &BTreeMap<String, &Scalar>) -> Scalar {
        match self {
            Scalar::Sum(terms) => {
                let mut new_terms: Vec<Scalar> = Vec::with_capacity(terms.len());
                for term in terms {
                    new_terms.push(term.replace_all(replacements));
                }
                add_all(new_terms)
            }
            Scalar::Product(factors) => {
                let mut new_factors: Vec<Scalar> = Vec::with_capacity(factors.len());
                for Exponential { b: factor_base, e: factor_exp } in factors {
                    new_factors.push(factor_base.replace_all(replacements) ^ factor_exp);
                }
                mul_all(new_factors)
            }
            Scalar::Variable(x) => match replacements.get(&x) {
                Some(replacement) => (*replacement).clone(),
                None => Scalar::Variable(x)
            }
            x => x
        }
    }

    // use on a Scalar that may have an incorrect form (like a sum with one term)
    // does not re-order unordered subexpressions
    fn correct_form(self) -> Self {
        match self {
            Scalar::Sum(mut terms) => match terms.len() {
                0 => S_ZERO,
                1 => match terms.pop() {
                    Some(only_term) => only_term,
                    None => S_ZERO
                }
                _ => Scalar::Sum(terms)
            }
            Scalar::Product(mut factors) => match factors.len() {
                0 => S_ONE,
                1 => match factors.pop() {
                    Some(only_factor) => Scalar::from(only_factor),
                    None => S_ONE
                }
                _ => Scalar::Product(factors)
            }
            x => x
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
        //let amogus = Scalar::Variable(String::from(char::from_u32(0xD9E).unwrap()));

        let mut simplifier = Simplifier {
            simplest: (self.clone(), self.cost()),
            territory: BTreeSet::new(),
            queue: Vec::new()
        };

        simplifier.discover(self.clone());
        while let Some(next_expr) = simplifier.queue.pop() {
            for derived in Simplifier::derive(&next_expr) {
                simplifier.discover(derived);
            }
        }
        let (simplest, _) = simplifier.simplest;
        simplest
    }
}