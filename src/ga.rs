use super::algebra_tools::*;
use super::scal::*;

use std::cmp::Ordering;
use std::ops;
use std::collections::VecDeque;

/// Canonical basis vector
#[derive(Clone, Eq, Ord)]
pub struct CBVec {
    pub id: i32,
    pub square: Scalar
}

/// Product of canonical basis vectors and scalars
#[derive(Clone, Eq, Ord)]
pub struct KBlade {
    /// Normalized form
    n: Vec<CBVec>,
    /// Coefficient/magnitude
    c: Scalar
}

/// Sum of k-blades
#[derive(Clone)]
pub struct MVec {
    blades: Vec<KBlade>
}

/// Sum of k-blades, all with grades equal to one k
#[derive(Clone)]
pub struct KVec {
    v: MVec,
    k: u32
}



impl From<CBVec> for KBlade {
    fn from(v: CBVec) -> Self {
        KBlade { n: vec![v], c: S_ONE }
    }
}
impl From<Scalar> for KBlade {
    fn from(x: Scalar) -> Self {
        KBlade { n: Vec::new(), c: x }
    }
}
impl From<KBlade> for MVec {
    fn from(b: KBlade) -> Self {
        MVec { blades: vec![b] }
    }
}



impl PartialEq<Self> for CBVec {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}
impl PartialOrd<Self> for CBVec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl PartialEq<Self> for KBlade {
    fn eq(&self, other: &Self) -> bool {
        self.n.eq(&other.n)
    }
}
impl PartialOrd<Self> for KBlade {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.n.partial_cmp(&other.n)
    }
}



impl ops::Mul for KBlade {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut sign_changes: usize = 0;
        let mut coefficient = self.c * rhs.c;
        let mut lhs: VecDeque<CBVec> = VecDeque::from(self.n);
        let mut rhs: VecDeque<CBVec> = VecDeque::from(rhs.n);
        let mut new_normal: Vec<CBVec> = Vec::with_capacity(lhs.len() + rhs.len());
        while let (Some(lhs_next), Some(rhs_next)) = (lhs.front(), rhs.front()) {
            match lhs_next.cmp(rhs_next) {
                Ordering::Less => new_normal.push(lhs.pop_front().unwrap()),
                Ordering::Equal => {
                    sign_changes += lhs.len() - 1;
                    coefficient = coefficient * lhs.pop_front().unwrap().square;
                    new_normal.push(rhs.pop_front().unwrap());
                }
                Ordering::Greater => {
                    sign_changes += lhs.len();
                    new_normal.push(rhs.pop_front().unwrap());
                }
            }
        }
        new_normal.append(&mut Vec::from(lhs));
        new_normal.append(&mut Vec::from(rhs));
        if sign_changes >> 1 == 1 {
            coefficient = coefficient * -S_ONE;
        }
        KBlade { n: new_normal, c: coefficient }
    }
}

impl ops::Add for MVec {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs: VecDeque<KBlade> = VecDeque::from(self.blades);
        let mut rhs: VecDeque<KBlade> = VecDeque::from(rhs.blades);
        let mut new_blades: Vec<KBlade> = Vec::with_capacity(lhs.len() + rhs.len());
        while let (Some(lhs_next), Some(rhs_next)) = (lhs.front(), rhs.front()) {
            match lhs_next.cmp(&rhs_next) {
                Ordering::Less => new_blades.push(lhs.pop_front().unwrap()),
                Ordering::Equal => {
                    let (lhs_next, rhs_next) = (lhs.pop_front().unwrap(), rhs.pop_front().unwrap());
                    let new_blade = KBlade { n: lhs_next.n, c: lhs_next.c + rhs_next.c };
                    if new_blade.c != S_ZERO {
                        new_blades.push(new_blade);
                    }
                }
                Ordering::Greater => new_blades.push(rhs.pop_front().unwrap())
            }
        }
        new_blades.append(&mut Vec::from(lhs));
        new_blades.append(&mut Vec::from(rhs));
        MVec { blades: new_blades }
    }
}

impl ops::Mul for MVec {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (lhs, rhs) = (self, rhs);
        let mut unsorted_blades: Vec<MVec> = Vec::with_capacity(lhs.blades.len() *
                                                                rhs.blades.len());
        for lhs_blade in &lhs.blades { for rhs_blade in &rhs.blades {
            unsorted_blades.push(MVec::from(lhs_blade.clone() * rhs_blade.clone()));
        }}
        merge_all(unsorted_blades, |a, b| a + b, &MVec::from(KBlade::from(S_ZERO)))
    }
}



impl KBlade {
    pub fn mul_inv(self) -> Self {
        let square = (self.clone() * self.clone()).c;
        KBlade { n: self.n, c: self.c / square }
    }
}