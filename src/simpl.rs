use super::{scal::*, rational::*, algebra_tools::*};
use std::cmp::Ordering;
use std::{fmt, ops};

use std::collections::{BTreeMap, BTreeSet};

const RULES: [fn(&Scalar) -> Vec<Scalar>; 3] = [
    // Factorization
    |x| match x.as_ref() {
        ScalarInner::Sum(sum) => {
            let terms = sum.ref_terms();

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
                match &terms[i].as_ref() {
                    ScalarInner::Product(prod) => {
                        let factors = prod.ref_factors();
                        for j in 0..factors.len() {
                            let factor = &factors[j];
                            index_factor(&mut factors_indices, factor.ref_base(), 
                                *factor.ref_exponent(), i, j);
                        }
                    }
                    x => index_factor(&mut factors_indices, &terms[i], Rational::ONE, i, 0)
                }
            }

            let mut factorizations: Vec<Scalar> = Vec::new();
            for (factor, indices) in factors_indices {
                for i in 0..(indices.len() - 1) {

                    fn remove_factor(term: Scalar, factor_index: usize, exp: Rational) -> Scalar {
                        match term.as_ref() {
                            ScalarInner::Product(prod) => {
                                let to_remove = Scalar::from(prod.ref_factors()[factor_index]
                                    .clone()) ^ exp;
                                term / to_remove
                            },
                            _ => Scalar::ONE
                        }
                    }

                    let FactorIndex { t_idx: t_idx1, f_idx: f_idx1, exp: exp1} = indices[i];
                    for &FactorIndex { t_idx: t_idx2, f_idx: f_idx2, exp: exp2} in
                        &indices[(i + 1)..] {
                        if (exp1.numerator() < 0) == (exp2.numerator() < 0) {
                            let exp: Rational = if exp1.abs() < exp2.abs() { exp1 }
                                                else { exp2 };
                            let mut new_terms = sum.clone();
                            let term1 = new_terms.remove_term(t_idx1).unwrap();
                            let term2 = new_terms.remove_term(
                                if t_idx2 < t_idx1 { t_idx2 }
                                else { t_idx2 - 1 }).unwrap();

                            factorizations.push(Scalar::from(new_terms) +
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
    |x| match x.as_ref() {
        ScalarInner::Product(prod) => {
            let factors = prod.ref_factors();

            let mut distributions: Vec<Scalar> = Vec::new();
            for index1 in 0..factors.len() {
                for index2 in 0..factors.len() {
                    // Distribution
                    if let ScalarInner::Sum(_) = factors[index2].ref_base().as_ref() {
                        fn distribute(distributions: &mut Vec<Scalar>, other_factors: Scalar,
                                      terms: Sum, terms_exp: Rational, distributed: Scalar) {
                            for i in 0..terms.ref_terms().len() {
                                let mut other_terms = terms.clone();
                                let term = other_terms.remove_term(i).unwrap();
                                distributions.push(other_factors.clone() * ((
                                    distributed.clone() * term +
                                    distributed.clone() * Scalar::from(other_terms)
                                ) ^ terms_exp));
                            }
                        }

                        // distribute factor into another factor
                        if *factors[index1].ref_exponent() == *factors[index2].ref_exponent() 
                                && index1 != index2 {
                            let mut other_factors = prod.clone();
                            let distributed = other_factors.remove_factor(index1)
                                .unwrap().into_base();
                            let (terms_exp, terms) = match other_factors.remove_factor(
                                if index2 < index1 { index2 }
                                else { index2 - 1 }
                            ).unwrap() { factor => (
                                *factor.ref_exponent(), 
                                Sum::from(factor.into_base())) };

                            distribute(&mut distributions, Scalar::from(other_factors),
                                       terms, terms_exp, distributed);
                        }
                        
                        /*
                        // distribute factor into itself
                        if (if index1 == index2 { 
                                factors[index2].e - factors[index1].e
                            }
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
                         */
                    }
                }
            }
            distributions
        }
        _ => Vec::new()
    },

    // (Not Started) Exponentiation of rational-based exponentials
    |x| match x.as_ref() {
        ScalarInner::Product(prod) => {
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
impl AddIdentity for ExprComplexity {
    const ZERO: ExprComplexity = ExprComplexity { num_terms: 0, num_factors: 0 };
}
impl MulIdentity for ExprComplexity {
    const ONE: ExprComplexity = ExprComplexity { num_terms: 1, num_factors: 0 };
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



impl Scalar {
    fn cost(&self) -> ExprCost {
        match self.as_ref() {
            ScalarInner::Sum(s) => merge_seq(s.ref_terms().iter().map(|term| term.cost()),
                    |a, b| a + b, ExprCost::ZERO) + 
                ExprCost::new(s.ref_terms().len() - 1, self.is_constant()),
            ScalarInner::Product(p) => merge_seq(p.ref_factors().iter()
                        .map(|factor| factor.ref_base().cost()),
                    |a, b| a + b, ExprCost::ZERO) + 
                ExprCost::new(p.ref_factors().len() - 1, self.is_constant()),
            _ => ExprCost::ZERO
        }
    }

    // Provides an upper bound on the complexity of any expression that could be derived from 
    //  this expression without the addition of new term/factor pairs that cancel each other out
    fn complexity(&self) -> ExprComplexity {
        match self.as_ref() {
            ScalarInner::Sum(s) => merge_seq(s.ref_terms().iter().map(|term| term.complexity()), 
                |a, b| a + b, ExprComplexity::ZERO),
            ScalarInner::Product(p) => merge_seq(p.ref_factors().iter()
                    .map(|factor| exponentiate(factor.ref_base().complexity(), 
                        factor.ref_exponent().round_from_zero().unsigned_abs())),
                |a, b| a * b, ExprComplexity::ONE),
            _ => ExprComplexity { num_terms: 1, num_factors: 1 }
        }
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
    pub fn simplify(expr: &Scalar) -> Scalar {
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
        let mut transformed: Vec<Scalar> = RULES.iter()
            .map(|rule| rule(expr).into_iter())
            .flatten().collect();
        match expr.as_ref() {
            ScalarInner::Sum(terms) => {
                for (i, term) in terms.ref_terms().iter().enumerate() {
                    for trans_term in Simplifier::derive(term) {
                        let mut new_terms = terms.clone();
                        new_terms.remove_term(i);
                        transformed.push(Scalar::from(new_terms) + trans_term);
                    }
                }
            }
            ScalarInner::Product(factors) => {
                for (i, factor) in factors.ref_factors().iter().enumerate() {
                    for trans_factor in Simplifier::derive(factor.ref_base()) {
                        let mut new_factors = factors.clone();
                        new_factors.remove_factor(i);
                        transformed.push(Scalar::from(new_factors) * 
                            (trans_factor ^ *factor.ref_exponent()));
                    }
                }
            }
            _ => {}
        }
        transformed
    }
}