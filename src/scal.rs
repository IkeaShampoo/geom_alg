use std::cmp;
use std::fmt;
use std::ops;
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
const ZERO: Rational = Rational { n: 0, d: 1 };
const ONE: Rational = Rational { n: 1, d: 1 };

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
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        let (Exponential { b: lb, e: le },
             Exponential { b: rb, e: re }) = (self, other);
        let b_order = lb.partial_cmp(rb);
        if b_order == Some(cmp::Ordering::Equal) { le.partial_cmp(re) }
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
        if exponential.e == ONE { exponential.b }
        else if exponential.e == ZERO { Scalar::from(ONE) }
        else { Scalar::Product(vec![exponential]) }
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
                Scalar::Sum(new_terms)
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
                                new_factors.push(Exponential { b: base, e: exp });
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
        if exp == ZERO { Scalar::from(ONE) }
        else if exp == ONE { self }
        else {
            match self {
                Scalar::Rational(x) => {
                    if exp.d == 1 { Scalar::Rational(x ^ exp.n) }
                    else { Scalar::Product(vec![Exponential { b: Scalar::Rational(x), e: exp }]) }
                }
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

const RULES: [fn(&Scalar) -> Vec<Scalar>; 2] = [
    |x| match x {                                          // Factorization
        Scalar::Sum(terms) => {
            type FactorIndexer = BTreeMap<Scalar, Vec<(usize, Rational)>>;
            fn index_factor(factors_indices: &mut FactorIndexer, term_index: usize,
                            factor_base: &Scalar, factor_exp: Rational) {
                match factors_indices.get_mut(factor_base) {
                    Some(factor_indices) => factor_indices.push((term_index, factor_exp)),
                    None => {
                        factors_indices.insert(factor_base.clone(), vec![(term_index,
                                                                          factor_exp)]);
                    }
                }
            }

            let mut factors_indices: FactorIndexer = BTreeMap::new();
            let mut term_index: usize = 0;
            for term in terms {
                match term {
                    Scalar::Product(factors) => {
                        for factor in factors {
                            index_factor(&mut factors_indices, term_index, &factor.b, factor.e);
                        }
                    }
                    x => index_factor(&mut factors_indices, term_index, &x, ONE)
                }
                term_index += 1;
            }

            let mut factorizations: Vec<Scalar> = Vec::new();
            for (factor, indices) in factors_indices {
                for i in 0..(indices.len() - 1) {

                    // ASSUMES THAT term CONTAINS factor
                    fn remove_factor(term: Scalar, factor: &Scalar, exp: Rational) -> Scalar {
                        match term {
                            Scalar::Product(mut factors) => {
                                for j in 0..factors.len() {
                                    if factors[j].b == *factor {
                                        if factors[j].e == exp {
                                            factors.remove(j);
                                        }
                                        else {
                                            factors[j].e = factors[j].e - exp;
                                        }
                                        break;
                                    }
                                }
                                match factors.len() {
                                    0 => Scalar::Rational(ONE),
                                    1 => match factors.pop() {
                                        Some(one_factor) => one_factor.b ^ one_factor.e,
                                        None => Scalar::Rational(ONE)
                                    }
                                    _ => Scalar::Product(factors)
                                }
                            }
                            _ => Scalar::Rational(ONE)
                        }
                    }

                    let (index1, exp1) = indices[i];
                    for &(index2, exp2) in &indices[(i + 1)..] {
                        if (exp1.n < 0) == (exp2.n < 0) {
                            let exp: Rational = if exp1.abs() < exp2.abs() { exp1 }
                                                else { exp2 };
                            let mut new_terms: Vec<Scalar> = terms.clone();
                            let init_last_index = new_terms.len() - 1;
                            let term1 = new_terms.swap_remove(index1);
                            let term2 = new_terms.swap_remove(
                                if index2 == init_last_index { index1 }
                                else { index2 }
                            );
                            new_terms.push((factor.clone() ^ exp) *
                                           (remove_factor(term1, &factor, exp) +
                                            remove_factor(term2, &factor, exp)));

                            factorizations.push(add_all(new_terms));
                        }
                    }
                }
            }
            factorizations
        }
        _ => Vec::new()
    },

    |x| match x {                                          // Partial Distribution
        Scalar::Product(factors) => {
            /*
            for each factor (factor1)
                for each other factor, if its base is a sum
                    for each term of sum
                        distribute factor1 into the term and into the rest of the sum

             f * (a + b)^e = f * (a + b) * (a + b)^(e - 1)
             (f^e) * (a + b)^e

             */
            let mut distributions: Vec<Scalar> = Vec::new();
            for index1 in 0..factors.len() {
                for index2 in 0..factors.len() {
                    if let Scalar::Sum(_) = &(factors[index2].b) {

                        fn remove_factor(factors: &mut Vec<Exponential>,
                                         index: usize, exp: Rational) {
                            let new_exp = factors[index].e - exp;
                            if new_exp == ZERO {
                                factors[index] = Exponential { b: Scalar::Rational(ONE), e: ONE };
                            }
                            else {
                                factors[index].e = new_exp;
                            }
                        }

                        if factors[index1].e == factors[index2].e && index1 != index2 {
                            let mut other_factors: Vec<Exponential> = factors.clone();
                            let distributed: Scalar = other_factors.remove(index1).b;
                            let terms: Vec<Scalar> = match other_factors.remove(
                                if index2 < index1 { index2 }
                                else { index2 - 1 }
                            ).b {
                                Scalar::Sum(terms) => terms,
                                _ => panic!("Uh oh")
                            };
                            let other_factors = match other_factors.len() {
                                0 => Scalar::Rational(ONE),
                                1 => {
                                    let x = other_factors.pop().expect("Factor is missing");
                                    x.b ^ x.e
                                },
                                _ => Scalar::Product(other_factors)
                            };

                            for i in 0..terms.len() {
                                let mut other_terms = terms.clone();
                                let term = other_terms.remove(i);
                                distributions.push(other_factors.clone() * (
                                    distributed.clone() * term +
                                    distributed.clone() * Scalar::Sum(other_terms)
                                ));
                            }
                        }
                        if (if index1 == index2 { factors[index2].e - factors[index1].e }
                            else { factors[index2].e })
                            >= ONE {
                            let mut new_factors: Vec<Exponential> = factors.clone();

                        }
                    }
                }
            }
            distributions
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
        println!("{}", expr);    // TEST

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
            for trans_expr in (*rule)(&expr) {
                self.discover(full_template.clone().replace(expr_placeholder, &trans_expr));
            }
        }
    }
}




fn merge_rec<T: Clone>(arguments: &mut Vec<T>, size: usize,
                merge: fn(T, T) -> T, identity: &T) -> T {
    if size > 2 {
        let left_size = size / 2;
        merge(merge_rec(arguments, left_size, merge, identity),
              merge_rec(arguments, size - left_size, merge, identity))
    }
    else {
        match (arguments.pop(), arguments.pop()) {
            (Some(a), Some(b)) => merge(a, b),
            (Some(x), None) | (None, Some(x)) => x,
            (None, None) => identity.clone()
        }
    }
}

fn merge_scalars(arguments: Vec<Scalar>, merge: fn(Scalar, Scalar) -> Scalar, identity: &Scalar)
                 -> Scalar {
    let init_size = arguments.len();
    merge_rec(&mut { arguments }, init_size, merge, identity)
}

fn add_all(terms: Vec<Scalar>) -> Scalar {
    merge_scalars(terms, |x, y| x + y, &Scalar::from(ZERO))
}

fn mul_all(factors: Vec<Scalar>) -> Scalar {
    merge_scalars(factors, |x, y| x * y, &Scalar::from(ONE))
}

fn mul_exp(factors: Vec<Exponential>) -> Scalar {
    let mut new_factors: Vec<Scalar> = Vec::with_capacity(factors.len());
    for factor in factors {
        new_factors.push(Scalar::from(factor));
    }
    mul_all(new_factors)
}

impl Scalar {
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
        let swap = Scalar::Variable(String::from(char::from_u32(0xD9E).unwrap()));

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