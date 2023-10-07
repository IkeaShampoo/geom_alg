use super::{algebra_tools::*, rational::*};

use std::cmp::Ordering;
use std::{fmt, mem, ops};
use std::collections::BTreeMap;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Exponential {
    b: Scalar, e: Rational
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
pub struct Sum(Vec<Scalar>);
impl Sum {
    pub fn terms(self) -> Vec<Scalar> {
        match self { Sum(terms) => terms }
    }
    pub fn terms_ref(&self) -> &Vec<Scalar> {
        match self { Sum(terms) => terms }
    }
}

impl Sum {
    fn add(self, rhs: Self) -> Self {
        let mut lhs = self.terms().into_iter().peekable();
        let mut rhs = rhs.terms().into_iter().peekable();
        let mut new_terms: Vec<Scalar> = Vec::with_capacity(lhs.len() + rhs.len());
        while let (Some(lhs_next), Some(rhs_next)) = (lhs.peek(), rhs.peek()) {
            match (lhs_next, rhs_next) {
                // Each sum will only have up to one rational term
                (&Scalar(S::Rational(lhs_next)), &Scalar(S::Rational(rhs_next))) => {
                    lhs.next();
                    rhs.next();
                    let rat_sum = lhs_next + rhs_next;
                    if rat_sum != Rational::ZERO {
                        new_terms.push(Scalar::from(rat_sum));
                    }
                }
                (lhs_next, rhs_next) => new_terms.push(
                    if *lhs_next < *rhs_next { lhs.next() }
                    else { rhs.next() } .unwrap())
            }
        }
        new_terms.extend(lhs);
        new_terms.extend(rhs);
        Sum(new_terms)
    }
}

impl From<Scalar> for Sum {
    fn from(x: Scalar) -> Self {
        match x {
            Scalar::ZERO => Sum(Vec::new()),
            Scalar(S::Sum(sum)) => sum,
            x => Sum(vec![x])
        }
    }
}



#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Product(Vec<Exponential>);
impl Product {
    fn factors(self) -> Vec<Exponential> {
        match self { Product(factors) => factors }
    }
    fn factors_ref(&self) -> &Vec<Exponential> {
        match self { Product(factors) => factors }
    }
}

impl Product {
    fn mul(self, rhs: Self) -> Self {
        let mut lhs = self.factors().into_iter().peekable();
        let mut rhs = rhs.factors().into_iter().peekable();
        let mut new_factors: Vec<Exponential> = Vec::with_capacity(lhs.len() + rhs.len());
        while let (Some(lhs_next), Some(rhs_next)) = (lhs.peek(), rhs.peek()) {
            let order = (lhs_next.b).cmp(&rhs_next.b);
            match order {
                Ordering::Equal => {
                    let exp = lhs_next.e + rhs_next.e;
                    let base = rhs.next().unwrap().b;
                    lhs.next();
                    if base != -Rational::ONE && exp != Rational::ZERO {
                        new_factors.push(Exponential { b: base, e: exp });
                    }
                }
                Ordering::Less => new_factors.push(lhs.next().unwrap()),
                Ordering::Greater => new_factors.push(rhs.next().unwrap())
            }
        }
        new_factors.extend(lhs);
        new_factors.extend(rhs);
        Product(new_factors)
    }
}

impl From<Scalar> for Product {
    fn from(x: Scalar) -> Self {
        match x {
            Scalar::ONE => Product(Vec::new()),
            Scalar(S::Product(product)) => product,
            x => Product(vec![Exponential { b: x, e: Rational::ONE }])
        }
    }
}



#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum ScalarInner {
    Rational(Rational),
    Variable(String),
    Sum(Sum),
    Product(Product),
}
type S = ScalarInner;

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
impl From<Exponential> for Scalar {
    fn from(exponential: Exponential) -> Self {
        match exponential.e {
            Rational::ZERO => Scalar::ONE,
            Rational::ONE => exponential.b,
            _ => if exponential.b == Rational::ONE { Scalar::ONE }
                 else { Scalar(S::Product(Product(vec![exponential]))) }
        }
    }
}
impl From<Sum> for Scalar {
    fn from(Sum(mut terms): Sum) -> Self {
        match terms.len() {
            0 => Scalar::ZERO,
            1 => match terms.pop() {
                Some(only_term) => only_term,
                None => Scalar::ZERO
            }
            _ => Scalar(S::Sum(Sum(terms)))
        }
    }
}
impl From<Product> for Scalar {
    fn from(Product(mut factors): Product) -> Self {
        match factors.len() {
            0 => Scalar::ZERO,
            1 => match factors.pop() {
                Some(only_factor) => Scalar::from(only_factor),
                None => Scalar::ZERO
            }
            _ => Scalar(S::Product(Product(factors)))
        }
    }
}
impl From<ScalarInner> for Scalar {
    fn from(inner: S) -> Self {
        match inner {
            S::Sum(sum) => Scalar::from(sum),
            S::Product(prod) => Scalar::from(prod),
            x => Scalar(x)
        }
    }
}

impl Into<ScalarInner> for Scalar {
    fn into(self) -> S {
        match self { Scalar(x) => x }
    }
}
impl AsRef<ScalarInner> for Scalar {
    fn as_ref(&self) -> &ScalarInner {
        match self { Scalar(x) => x }
    }
}

impl AddIdentity for Scalar {
    const ZERO: Scalar = Scalar(S::Rational(Rational::ZERO));
}
impl MulIdentity for Scalar {
    const ONE: Scalar = Scalar(S::Rational(Rational::ONE));
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
            (Scalar::ZERO, rhs) => rhs,
            (lhs, Scalar::ZERO) => lhs,
            (lhs, rhs) => Scalar::from(Sum::add(Sum::from(lhs), Sum::from(rhs)))
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
        match (self, rhs) {
            (Scalar(S::Rational(lhs)), Scalar(S::Rational(rhs))) => Scalar(S::Rational(lhs * rhs)),
            (Scalar::ZERO, _) | (_, Scalar::ZERO) => return Scalar::ZERO,
            (Scalar::ONE, rhs) => rhs,
            (lhs, Scalar::ONE) => lhs,
            (lhs, rhs) => Scalar::from(Product::mul(Product::from(lhs), Product::from(rhs)))
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
                Scalar::ONE => Scalar::ONE,
                Scalar(S::Product(Product(mut factors))) => {
                    for factor in &mut factors {
                    // can't change term order or multiplicatively invert terms
                        (*factor).e = (*factor).e * exp;
                    }
                    Scalar(S::Product(Product(factors)))
                }
                x => Scalar(S::Product(Product(vec![Exponential { b: x, e: exp }])))
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
        match self.as_ref() {
            S::Rational(s) => Rational::fmt(s, f),
            S::Variable(s) => f.write_str(s.as_str()),
            S::Sum(Sum(s)) => {
                f.write_str("(")?;
                for i in 0..s.len() {
                    if i > 0 { f.write_str(" + ")?; }
                    Scalar::fmt(&s[i],f)?;
                }
                f.write_str(")")
            },
            S::Product(Product(s)) => 
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



pub struct FragmentedScalar {
    expr: Scalar,
    temp_vars: usize,
    subexprs: BTreeMap<Scalar, Scalar>
}

impl Scalar {
    pub fn mul_inv(self) -> Self {
        self ^ -Rational::ONE
    }

    fn replace(self, to_replace: &Scalar, replace_with: &Scalar) -> Scalar {
        match self {
            to_replace => replace_with.clone(),
            Scalar(S::Sum(s)) => add_all(s.terms().into_iter()
                .map(|term| term.replace(to_replace, replace_with))),
            Scalar(S::Product(p)) => mul_all(p.factors().into_iter()
                .map(|Exponential { b: factor_base, e: factor_exp }|
                    factor_base.replace(to_replace, replace_with) ^ factor_exp)),
            x => x
        }
    }

    fn replace_all(self, replacements: &BTreeMap<Scalar, Scalar>) -> Scalar {
        match replacements.get(&self) {
            Some(replacement) => replacement.clone(),
            None => match self {
                Scalar(S::Sum(s)) => add_all(s.terms().into_iter()
                    .map(|term| term.replace_all(replacements))),
                Scalar(S::Product(p)) => mul_all(p.factors().into_iter()
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
            Scalar(S::Sum(Sum(terms))) => {
                for term in terms {
                    if !term.is_constant() {
                        return false;
                    }
                }
                true
            }
            Scalar(S::Product(Product(factors))) => {
                for factor in factors {
                    if !factor.b.is_constant() {
                        return false;
                    }
                }
                true
            }
        }
    }

    // doesn't work with non-polynomial-like expressions
    pub fn is_zero(&self) -> bool {
        fn distribute(a: Scalar, b: Scalar) -> Scalar {
            let (a, b): (Sum, Sum) = (a.into(), b.into());
            add_all(a.terms_ref().iter().map(|t1| 
                add_all(b.terms_ref().iter().map(|t2| t1.clone() * t2.clone()))))
        }
        fn expand(b: Scalar, e: u32) -> Scalar {
            merge_self(b.clone(), e, distribute, &b)
        }
        fn distribute_all(expr: &Scalar) -> Scalar {
            match expr.as_ref() {
                S::Sum(s) => add_all(s.terms_ref().iter().map(|term| distribute_all(term))),
                S::Product(p) => merge_all(p.factors_ref().iter().map(
                    |Exponential{b: terms, e: exp}|
                    if exp.numerator() > 0 && exp.denominator() == 1 {
                        expand(distribute_all(&terms), exp.numerator().unsigned_abs())
                    } 
                    else { terms.clone() ^ *exp }), distribute, &Scalar::ONE),
                x => expr.clone()
            }
        }
        distribute_all(self) == Scalar::ZERO
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
                Scalar(S::Sum(sum)) => {
                    let new_terms: Vec<Scalar> = sum.terms().into_iter()
                        .map(|term| collect_chains(term, sums, products)).collect();
                    sums.push(new_terms);
                    Scalar::from(String::from(SUM_PREFIX) + 
                        (sums.len() - 1).to_string().as_str())
                }
                Scalar(S::Product(prod)) => {
                    let new_factors = prod.factors().into_iter().map(|factor| Exponential {
                        b: collect_chains(factor.b, sums, products), 
                        e: factor.e }).collect();
                    products.push(new_factors);
                    Scalar::from(String::from(PRODUCT_PREFIX) + 
                        (products.len() - 1).to_string().as_str())
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
            for j in (i + 1)..sums.len() {
                // if terms contains sum, remove sum from terms and replace it with i
                // if sum contains terms, remove terms from sum and replace it with *something*
                let mut lhs = sums[i].iter().enumerate().peekable();
                let mut rhs = sums[j].iter().enumerate().peekable();
                let mut intersection_idcs: Vec<(usize, usize)> = 
                    Vec::with_capacity(usize::max(lhs.len(), rhs.len()));
                while let (Some((lidx, lhs_next)), Some((ridx, rhs_next))) = 
                          (lhs.peek(), rhs.peek()) {
                    match lhs_next.cmp(rhs_next) {
                        Ordering::Equal => {
                            intersection_idcs.push((*lidx, *ridx));
                            lhs.next();
                            rhs.next();
                        }
                        _ => {}
                    }
                }
                if intersection_idcs.len() > 1 {
                    let intersection: Vec<Scalar> = intersection_idcs.iter().map(|(lidx, ridx)| {
                        sums[i].remove(*lidx);
                        sums[j].remove(*ridx)
                    }).collect();
                    sums.push(intersection);
                    let intersection_name = Scalar::from(String::from(SUM_PREFIX) + 
                        (sums.len() - 1).to_string().as_str());
                    let idx1 = sums[i].binary_search(&intersection_name).map_or_else(|x| x, |x| x);
                    sums[i].insert(idx1, intersection_name.clone());
                    let idx2 = sums[j].binary_search(&intersection_name).map_or_else(|x| x, |x| x);
                    sums[j].insert(idx2, intersection_name);
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