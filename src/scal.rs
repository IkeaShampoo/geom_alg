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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Rational { n: i32, d: u32 }

impl Rational {
    pub const ZERO: Rational = Rational { n: 0, d: 1 };
    pub const ONE: Rational = Rational { n: 1, d: 1 };

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



#[derive(Clone, PartialEq, Eq, Debug)]
struct Exponential {
    b: Scalar, e: Rational
}

impl From<Scalar> for Exponential {
    fn from(scalar: Scalar) -> Exponential {
        Exponential { b: scalar, e: Rational::ONE }
    }
}

impl Ord for Exponential {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.b.cmp(&other.b) {
            Ordering:: Equal => self.e.cmp(&other.e),
            order => order
        }
    }
}
impl PartialOrd for Exponential {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Exponential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.e == Rational::ONE { Scalar::fmt(&self.b, f) }
        else { f.write_fmt(format_args!("({}^{})", self.b, self.e)) }
    }
}



#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
enum S {
    Variable(String),
    Rational(Rational),
    Sum(Vec<Scalar>),
    Product(Vec<Exponential>),
}
impl From<Exponential> for S {
    fn from(exponential: Exponential) -> Self {
        match exponential.e {
            Rational::ZERO => S::ONE,
            Rational::ONE => match exponential.b { Scalar(x) => x },
            _ => if exponential.b == Rational::ONE { S::ONE }
                 else { S::Product(vec![exponential]) }
        }
    }
}
impl S {
    const ZERO: S = S::Rational(Rational::ZERO);
    const ONE: S = S::Rational(Rational::ONE);

    // use on a Scalar that may have an incorrect form (like a sum with one term)
    // does not re-order unordered subexpressions
    fn correct_form(self) -> Self {
        match self {
            S::Sum(mut terms) => match terms.len() {
                0 => S::ZERO,
                1 => match terms.pop() {
                    Some(Scalar(only_term)) => only_term,
                    None => S::ZERO
                }
                _ => S::Sum(terms)
            }
            S::Product(mut factors) => match factors.len() {
                0 => S::ONE,
                1 => match factors.pop() {
                    Some(only_factor) => S::from(only_factor),
                    None => S::ONE
                }
                _ => S::Product(factors)
            }
            x => x
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Scalar(S);

impl From<String> for Scalar {
    fn from(name: String) -> Self {
        Scalar(S::Variable(name))
    }
}
impl From<&str> for Scalar {
    fn from(name: &str) -> Self {
        Scalar(S::Variable(String::from(name)))
    }
}
impl From<Rational> for Scalar {
    fn from(fraction: Rational) -> Self {
        Scalar(S::Rational(fraction))
    }
}
impl TryInto<Vec<Scalar>> for Scalar {
    type Error = &'static str;
    fn try_into(self) -> Result<Vec<Scalar>, Self::Error> {
        match self {
            Scalar(S::Sum(terms)) => Ok(terms),
            _ => Err("This is not a Scalar::Sum")
        }
    }
}
impl TryInto<Vec<Exponential>> for Scalar {
    type Error = &'static str;
    fn try_into(self) -> Result<Vec<Exponential>, Self::Error> {
        match self {
            Scalar(S::Product(factors)) => Ok(factors),
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
    let mut stupid_workaround_swap_thing = Scalar::ONE;
    mem::swap(x, &mut stupid_workaround_swap_thing);
    stupid_workaround_swap_thing
}

impl ops::Add for Scalar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Scalar(S::Rational(lhs)), Scalar(S::Rational(rhs))) => Scalar(S::Rational(lhs + rhs)),
            (Scalar(S::Sum(lhs)), Scalar(S::Sum(rhs))) => {
                let mut lhs: VecDeque<Scalar> = VecDeque::from(lhs);
                let mut rhs: VecDeque<Scalar> = VecDeque::from(rhs);
                let mut new_terms: Vec<Scalar> = Vec::with_capacity(lhs.len() + rhs.len());
                while let (Some(lhs_next), Some(rhs_next)) = (lhs.front(), rhs.front()) {
                    match (lhs_next, rhs_next) {
                        // Each sum will only have up to one rational term
                        (&Scalar(S::Rational(lhs_next)), &Scalar(S::Rational(rhs_next))) => {
                            lhs.pop_front();
                            rhs.pop_front();
                            let rat_sum = lhs_next + rhs_next;
                            if rat_sum != Rational::ZERO {
                                new_terms.push(Scalar(S::Rational(rat_sum)));
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
                Scalar(S::Sum(new_terms).correct_form())
            }
            (lhs, rhs) => {
                if lhs == Rational::ZERO { return rhs; }
                if rhs == Rational::ZERO { return lhs; }
                let (old_terms, new_term): (Vec<Scalar>, Scalar) =
                    if let Scalar(S::Sum(lhs)) = lhs { (lhs, rhs) }
                    else if let Scalar(S::Sum(rhs)) = rhs { (rhs, lhs) }
                    else { (vec![lhs], rhs) };
                Scalar(S::Sum(old_terms)) + Scalar(S::Sum(vec![new_term]))
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
            (Scalar(S::Rational(lhs)), Scalar(S::Rational(rhs))) => Scalar(S::Rational(lhs * rhs)),
            (Scalar(S::Product(lhs)), Scalar(S::Product(rhs))) => {
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
                            if base != -Rational::ONE && exp != Rational::ZERO {
                                new_factors.push(Exponential { b: base, e: exp });
                            }
                        }
                        Ordering::Less => new_factors.push(lhs.pop_front().expect(EXPECT_ERR)),
                        Ordering::Greater => new_factors.push(rhs.pop_front().expect(EXPECT_ERR))
                    }
                }
                new_factors.append(&mut Vec::from(lhs));
                new_factors.append(&mut Vec::from(rhs));
                Scalar(S::Product(new_factors).correct_form())
            }
            (Scalar::ZERO, _) | (_, Scalar::ZERO) => return Scalar::ZERO,
            (Scalar::ONE, rhs) => rhs,
            (lhs, Scalar::ONE) => lhs,
            (lhs, rhs) => {
                let (old_factors, new_factor): (Vec<Exponential>, Scalar) =
                    if let Scalar(S::Product(lhs)) = lhs { (lhs, rhs) }
                    else if let Scalar(S::Product(rhs)) = rhs { (rhs, lhs) }
                    else { (vec![Exponential::from(lhs)], rhs) };
                Scalar(S::Product(old_factors)) * 
                    Scalar(S::Product(vec![Exponential::from(new_factor)]))
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
            Rational::ZERO => Scalar::ONE,
            Rational::ONE => self,
            _ => match self {
                /*
                Scalar::Rational(x) => {
                    if exp.d == 1 { Scalar::Rational(x ^ exp.n) }
                    else { Scalar::Product(vec![Exponential { b: Scalar::Rational(x), e: exp }]) }
                }
                 */
                Scalar(S::Product(mut factors)) => {
                    for factor in &mut factors {
                        (*factor).e = (*factor).e * exp;
                    }
                    Scalar(S::Product(factors))
                }
                x => Scalar(S::Product(vec![Exponential { b: x, e: exp }]))
            }
        }
    }
}

impl ops::Sub for Scalar {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        if self == rhs { Scalar::ZERO }
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
        Scalar(S::Rational(-Rational::ONE)) * self
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
            Scalar(S::Rational(s)) => Rational::fmt(s, f),
            Scalar(S::Variable(s)) => f.write_str(s.as_str()),
            Scalar(S::Sum(s)) => {
                f.write_str("(")?;
                for i in 0..s.len() {
                    if i > 0 { f.write_str(" + ")?; }
                    Scalar::fmt(&s[i],f)?;
                }
                f.write_str(")")
            },
            Scalar(S::Product(s)) => 
                if s.len() == 1 { Exponential::fmt(&s[0], f) }
                else {
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
impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}



const RULES: [fn(&Scalar) -> Vec<Scalar>; 3] = [
    // Factorization
    |x| match x {
        Scalar(S::Sum(terms)) => {
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
                    Scalar(S::Product(factors)) =>
                        for j in 0..factors.len() {
                            let factor = &factors[j];
                            index_factor(&mut factors_indices, &factor.b, factor.e, i, j);
                        }
                    x => index_factor(&mut factors_indices, &x, Rational::ONE, i, 0)
                }
            }

            let mut factorizations: Vec<Scalar> = Vec::new();
            for (factor, indices) in factors_indices {
                for i in 0..(indices.len() - 1) {

                    fn remove_factor(term: Scalar, factor_index: usize, exp: Rational) -> Scalar {
                        match term {
                            Scalar(S::Product(mut factors)) => {
                                if factors[factor_index].e == exp {
                                    factors.remove(factor_index);
                                }
                                else {
                                    factors[factor_index].e = factors[factor_index].e - exp;
                                }
                                Scalar(S::Product(factors).correct_form())
                            }
                            _ => Scalar::ONE
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

                            factorizations.push(Scalar(S::Sum(new_terms).correct_form()) +
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
        Scalar(S::Product(factors)) => {
            let mut distributions: Vec<Scalar> = Vec::new();
            for index1 in 0..factors.len() {
                for index2 in 0..factors.len() {
                    // Combination of rational-based exponentials
                    if let (Exponential { b: Scalar(S::Rational(b1)), e: e1 },
                            Exponential { b: Scalar(S::Rational(b2)), e: e2 }) =
                           (&factors[index1], &factors[index2]) {
                        let e_ratio = *e2 / *e1;
                        // e2 = e1 * e_ratio
                        if index1 != index2 && e_ratio.d == 1 {
                            let (b1, b2) = (*b1, *b2);
                            let e1 = *e1;
                            let mut other_factors = factors.clone();
                            other_factors.remove(index1);
                            other_factors.remove(if index2 < index1 { index2 } else {index2 - 1 });
                            distributions.push(Scalar(S::Product(other_factors).correct_form()) *
                                               (Scalar(S::Rational(b1 * (b2 ^ e_ratio.n))) ^ e1));
                        }
                    }
                    // Distribution
                    else if let Scalar(S::Sum(_)) = &(factors[index2].b) {
                        fn distribute(distributions: &mut Vec<Scalar>, other_factors: Scalar,
                                      terms: Vec<Scalar>, terms_exp: Rational,
                                      distributed: Scalar) {
                            for i in 0..terms.len() {
                                let mut other_terms = terms.clone();
                                let term = other_terms.remove(i);
                                distributions.push(other_factors.clone() * ((
                                    distributed.clone() * term +
                                    distributed.clone() * Scalar(S::Sum(other_terms).correct_form())
                                ) ^ terms_exp));
                            }
                        }

                        // distribute factor into another factor
                        if factors[index1].e == factors[index2].e && index1 != index2 {
                            let mut other_factors = factors.clone();
                            let distributed = Scalar(S::from(other_factors.remove(index1)));
                            let (terms, terms_exp) = match other_factors.remove(
                                if index2 < index1 { index2 }
                                else { index2 - 1 }
                            ) {
                                Exponential { b: Scalar(S::Sum(terms)), e: exp} => (terms, exp),
                                not_a_sum => panic!("{not_a_sum} is not a sum")
                            };

                            distribute(&mut distributions,
                                       Scalar(S::Product(other_factors).correct_form()),
                                       terms, terms_exp, distributed);
                        }

                        // distribute factor into itself
                        if (if index1 == index2 { factors[index2].e - factors[index1].e }
                            else { factors[index2].e }) >= Rational::ONE {

                            fn remove_factors(factors: &mut Vec<Exponential>,
                                              index1: usize, exp1: Rational,
                                              mut index2: usize, exp2: Rational) {
                                let new_exp1 = factors[index1].e - exp1;
                                if new_exp1 == Rational::ZERO {
                                    factors.remove(index1);
                                    if index2 > index1 {
                                        index2 -= 1;
                                    }
                                }
                                else { factors[index1].e = new_exp1 }

                                let new_exp2 = factors[index2].e - exp2;
                                if new_exp2 == Rational::ZERO { factors.remove(index2); }
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
        Scalar(S::Product(factors)) => {
            Vec::new()
        }
        _ => Vec::new()
    }
];



#[derive(Copy, Clone, PartialEq)]
struct ExprCost {
    c: usize,       // Constant cost
    v: usize        // Variable cost
}
impl ExprCost {
    const ZERO: ExprCost = ExprCost { c: 0, v: 0};

    fn new(ops_count: usize, is_constant: bool) -> ExprCost {
        if is_constant { ExprCost { c: ops_count, v: 0 }}
        else { ExprCost { v: ops_count, c: 0 } }
    }
}
impl PartialOrd for ExprCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.v.cmp(&other.v) {
            Ordering::Equal => Some(self.c.cmp(&other.c)),
            x => Some(x)
        }
    }
}
impl ops::Add for ExprCost {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        ExprCost { c: self.c + rhs.c, v: self.v + rhs.v}
    }
}

#[derive(Copy, Clone, PartialEq)]
struct ExprComplexity {
    num_terms: usize,
    num_factors: usize
}
impl ExprComplexity {
    const ADD_IDENTITY: ExprComplexity = ExprComplexity { num_terms: 0, num_factors: 0 };
    const MUL_IDENTITY: ExprComplexity = ExprComplexity { num_terms: 1, num_factors: 0 };
}
impl PartialOrd for ExprComplexity {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match self.num_terms.cmp(&rhs.num_terms) {
            Ordering::Equal => self.num_factors.partial_cmp(&rhs.num_factors),
            x => Some(x)
        }
    }
}
impl ops::Add for ExprComplexity {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        ExprComplexity { 
            num_terms: self.num_terms + rhs.num_terms, 
            num_factors: self.num_factors + rhs.num_factors
        }
    }
}
impl ops::Mul for ExprComplexity {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        ExprComplexity {
            num_terms: self.num_terms * rhs.num_terms,
            num_factors: self.num_terms * rhs.num_factors + rhs.num_terms * self.num_factors
        }
    }
}
impl fmt::Display for ExprComplexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{}T, {}F", self.num_terms, self.num_factors))
    }
}



struct Simplifier {
    simplest: (Scalar, ExprCost),
    territory: BTreeSet<Scalar>,
    queue: Vec<Scalar>,
    count: usize,
    complexity: ExprComplexity
}

impl Simplifier {
    fn simplify(expr: &Scalar) -> Scalar {
        let mut simplifier = Simplifier {
            simplest: (expr.clone(), expr.cost()),
            territory: BTreeSet::new(),
            queue: Vec::new(),
            count: 0,
            complexity: expr.complexity()
        };

        simplifier.discover(expr.clone());
        while let Some(next_expr) = simplifier.queue.pop() {
            //eprintln!("Queue size: {}", simplifier.queue.len());
            for derived in Simplifier::derive(&next_expr) {
                simplifier.discover(derived);
            }
        }
        //eprintln!("{}", simplifier.count);
        let (simplest, _) = simplifier.simplest;
        simplest
    }
    fn discover(&mut self, expr: Scalar) {
        //eprintln!("{expr}");
        if !self.territory.contains(&expr) {
            //if self.count - ((self.count >> 12) << 12) == 0 { eprintln!("{expr}"); }
            let expr_cost = expr.cost();
            let expr_complexity = expr.complexity();
            //eprintln!("Complexity: {expr_complexity}");
            let &(_, simplest_cost) = &self.simplest;
            
            if expr_complexity < self.complexity {
                //eprintln!("New complexity: {}", expr_complexity);
                //eprintln!("Count: {}", self.count);
                self.simplest = (expr.clone(), expr_cost);
                self.territory = BTreeSet::new();
                self.queue = Vec::new();
                //self.count = 0;
                self.complexity = expr_complexity;
            }
            else if expr_complexity == self.complexity && expr_cost < simplest_cost {
                //eprintln!("{expr}");
                self.simplest = (expr.clone(), expr_cost)
            }
            self.territory.insert(expr.clone());
            self.queue.push(expr);
            self.count += 1;
        }
    }

    fn derive(expr: &Scalar) -> Vec<Scalar> {
        //println!("{expr}");
        let mut transformed: Vec<Scalar> = Vec::new();
        for rule in &RULES {
            transformed.append(&mut (*rule)(expr));
        }
        match expr {
            Scalar(S::Sum(terms)) => {
                for (i, term) in terms.iter().enumerate() {
                    for trans_term in Simplifier::derive(term) {
                        let mut new_terms: Vec<Scalar> = terms.clone();
                        new_terms.remove(i);
                        transformed.push(Scalar(S::Sum(new_terms).correct_form()) + trans_term);
                    }
                }
            }
            Scalar(S::Product(factors)) => {
                for (i, factor) in factors.iter().enumerate() {
                    for trans_factor in Simplifier::derive(&factor.b) {
                        let mut new_factors: Vec<Exponential> = factors.clone();
                        new_factors.remove(i);
                        transformed.push(Scalar(S::Product(new_factors).correct_form()) *
                                         (trans_factor ^ factor.e));
                    }
                }
            }
            _ => {}
        }
        transformed
    }
}



#[inline(always)]
fn add_all(size: usize, terms: impl Iterator<Item = Scalar>) -> Scalar {
    merge_all(size, terms, |x, y| x + y, &Scalar::ZERO)
}
#[inline(always)]
fn mul_all(size: usize, factors: impl Iterator<Item = Scalar>) -> Scalar {
    merge_all(size, factors, |x, y| x * y, &Scalar::ONE)
}
#[inline(always)]
fn mul_exp(factors: Vec<Exponential>) -> Scalar {
    mul_all(factors.len(), factors.into_iter().map(|f| Scalar(S::from(f))))
}


pub struct FragmentedScalar {
    expr: Scalar,
    temp_vars: usize,
    subexprs: BTreeMap<Scalar, Scalar>
}


impl Scalar {
    pub const ZERO: Scalar = Scalar(S::Rational(Rational::ZERO));
    pub const ONE: Scalar = Scalar(S::Rational(Rational::ONE));

    pub fn mul_inv(self) -> Self {
        self ^ -Rational::ONE
    }

    fn replace(self, to_replace: &Scalar, replace_with: &Scalar) -> Scalar {
        if self.eq(to_replace) { replace_with.clone() }
        else { match self {
            Scalar(S::Sum(terms)) => add_all(terms.len(), terms.into_iter()
                .map(|term| term.replace(to_replace, replace_with))),
            Scalar(S::Product(factors)) => mul_all(factors.len(), factors.into_iter()
                .map(|Exponential { b: factor_base, e: factor_exp }|
                        factor_base.replace(to_replace, replace_with) ^ factor_exp)),
            x => x
        }}
    }

    fn replace_all(self, replacements: &BTreeMap<Scalar, Scalar>) -> Scalar {
        match replacements.get(&self) {
            Some(replacement) => replacement.clone(),
            None => match self {
                Scalar(S::Sum(terms)) => add_all(terms.len(), terms.into_iter()
                    .map(|term| term.replace_all(replacements))),
                Scalar(S::Product(factors)) => mul_all(factors.len(), factors.into_iter()
                    .map(|Exponential { b: factor_base, e: factor_exp }|
                         factor_base.replace_all(replacements) ^ factor_exp)),
                x => x
            }
        }
    }

    fn map_variables(&self, other: &Scalar) -> Option<BTreeMap<String, &Scalar>> {
        fn map_variables_rec(this: &Scalar, other: &Scalar, 
                             map: &mut BTreeMap<String, &Scalar>) -> bool {
            todo!()
        }
        todo!()
    }

    fn is_constant(&self) -> bool {
        match self {
            Scalar(S::Rational(_)) => true,
            Scalar(S::Variable(x)) => *x == x.to_uppercase(),
            Scalar(S::Sum(terms)) => {
                for term in terms {
                    if !term.is_constant() {
                        return false;
                    }
                }
                true
            }
            Scalar(S::Product(factors)) => {
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
            Scalar(S::Sum(terms)) => merge_seq(terms.iter().map(|term| term.cost()),
                    |a, b| a + b, &ExprCost::ZERO) + 
                ExprCost::new(terms.len() - 1, self.is_constant()),
            Scalar(S::Product(factors)) => merge_seq(factors.iter().map(|factor| factor.b.cost()),
                    |a, b| a + b, &ExprCost::ZERO) + 
                ExprCost::new(factors.len() - 1, self.is_constant()),
            _ => ExprCost::ZERO
        }
    }

    // Provides an upper bound on the complexity of any expression that could be derived from 
    //  this expression without the addition of new term/factor pairs that cancel each other out
    fn complexity(&self) -> ExprComplexity {
        match self {
            Scalar(S::Sum(terms)) => merge_seq(terms.iter().map(|term| term.complexity()), 
                |a, b| a + b, &ExprComplexity::ADD_IDENTITY),
            Scalar(S::Product(factors)) => merge_seq(factors.iter()
                    .map(|factor| exponentiate(factor.b.complexity(), 
                        factor.e.round_from_zero().unsigned_abs(), ExprComplexity::MUL_IDENTITY)),
                |a, b| a * b, &ExprComplexity::MUL_IDENTITY),
            _ => ExprComplexity { num_terms: 1, num_factors: 1 }
        }
    }

    pub fn simplified(&self) -> Scalar {
        Simplifier::simplify(self)
    }

    pub fn zero_simplified(&self) -> Scalar {
        let terms: BTreeMap<Scalar, Rational> = BTreeMap::new();
        self.simplified()
    }

    pub fn fragment(self) -> FragmentedScalar {
        const SUM_PREFIX: &str = "\u{D9E}s";
        const PRODUCT_PREFIX: &str = "\u{D9E}p";

        fn get_the_chain<'a, T>(idx_str: &str, chains: &'a Vec<Vec<T>>) -> Option<&'a Vec<T>> {
            match idx_str.parse::<usize>() {
                Ok(i) => chains.get(i),
                _ => None
            }
            /* Old return for Ok(i) case
            chains.push(Vec::new());
            Some(chains.swap_remove(i))
             */
        }

        /*
        fn get_chain(expr: Scalar, sums: &mut Vec<Vec<Scalar>>, 
                     products: &mut Vec<Vec<Exponential>>) -> Option<Scalar> {
            match expr {
                Scalar::Variable(x) => match &x[..SUM_PREFIX.len()] {
                    SUM_PREFIX => get_the_chain(&x[SUM_PREFIX.len()..], sums)
                        .map_or(None, |terms| Some(Scalar::Sum(terms))),
                    PRODUCT_PREFIX => get_the_chain(&x[SUM_PREFIX.len()..], products)
                        .map_or(None, |factors| Some(Scalar::Product(factors))),
                    _ => None
                }
                _ => None
            }
        }
         */
        
        fn collect_chains(expr: Scalar, sums: &mut Vec<Vec<Scalar>>, 
                          products: &mut Vec<Vec<Exponential>>) -> Scalar {
            match expr {
                Scalar(S::Sum(terms)) => {
                    let new_terms: Vec<Scalar> = terms.into_iter()
                        .map(|term| collect_chains(term, sums, products)).collect();
                    sums.push(new_terms);
                    Scalar(S::Variable(String::from(SUM_PREFIX) + 
                        (sums.len() - 1).to_string().as_str()))
                }
                Scalar(S::Product(factors)) => {
                    let new_factors = factors.into_iter().map(|factor| Exponential {
                        b: collect_chains(factor.b, sums, products), 
                        e: factor.e }).collect();
                    products.push(new_factors);
                    Scalar(S::Variable(String::from(PRODUCT_PREFIX) + 
                        (products.len() - 1).to_string().as_str()))
                }
                x => x
            }
        }

        let mut sums: Vec<Vec<Scalar>> = Vec::new();
        let mut products: Vec<Vec<Exponential>> = Vec::new();
        collect_chains(self, &mut sums, &mut products);

        let sums_ref = &mut sums;
        let products_ref = &mut products;

        let get_sum = |expr| match expr {
            Scalar(S::Variable(x)) => match &x[..SUM_PREFIX.len()] {
                SUM_PREFIX => get_the_chain(&x[SUM_PREFIX.len()..], sums_ref),
                _ => None
            }
            _ => None
        };

        let get_product = |expr| match expr {
            Scalar(S::Variable(x)) => match &x[..SUM_PREFIX.len()] {
                PRODUCT_PREFIX => get_the_chain(&x[SUM_PREFIX.len()..], products_ref),
                _ => None
            }
            _ => None
        };

        for i in 0..sums.len() {
            for j in 0..sums.len() {
                // if terms contains sum, remove sum from terms and replace it with i
                // if sum contains terms, remove terms from sum and replace it with *something*
                let mut lhs = sums[i].iter().enumerate().peekable();
                let mut rhs = sums[j].iter().enumerate().peekable();
                let mut intersection: Vec<(usize, usize)> = 
                    Vec::with_capacity(usize::max(lhs.len(), rhs.len()));
                while let (Some((lidx, lhs_next)), Some((ridx, rhs_next))) = 
                          (lhs.peek(), rhs.peek()) {
                    match lhs_next.cmp(rhs_next) {
                        Ordering::Equal => {
                            intersection.push((*lidx, *ridx));
                            lhs.next();
                            rhs.next();
                        }
                        _ => {}
                    }
                }
                if intersection.len() > 1 {
                    
                }
            }
        }

        todo!()

        /*
        Log all argument sets for sums/products
        Find each sum's set's intersection with each other sum's set (repeat for products)
        Create fragment expression for each intersection, replace all occurances of the
            intersection (even those within other fragments) with the fragment's name.
            If the smallest intersections are fragmented first, then existing fragments don't
                have to be searched for the occurances of new fragments. 
         */
    }
}