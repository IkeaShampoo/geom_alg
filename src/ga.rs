use super::algebra_tools::*;
use super::scal::*;

use std::cmp::{max, min, Ordering};
use std::collections::VecDeque;
use std::{fmt, mem, ops};

/// Canonical basis vector
#[derive(Clone, Eq)]
pub struct CBVec {
    pub id: i32,
    pub square: Scalar
}

/// Product of canonical basis vectors and scalars
#[derive(Clone, Eq)]
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
    k: usize
}



impl From<CBVec> for KBlade {
    fn from(v: CBVec) -> Self {
        KBlade { n: vec![v], c: Scalar::ONE }
    }
}
impl From<Scalar> for KBlade {
    fn from(x: Scalar) -> Self {
        KBlade { n: Vec::new(), c: x }
    }
}
impl From<KBlade> for MVec {
    fn from(b: KBlade) -> Self {
        MVec { blades: match b.c {
            Scalar::ZERO => Vec::new(),
            _ => vec![b]
        }}
    }
}
impl From<KBlade> for KVec {
    fn from(b: KBlade) -> Self {
        KVec { k: b.grade(), v: MVec::from(b) }
    }
}
impl From<KVec> for MVec {
    fn from(v: KVec) -> Self {
        v.v
    }
}



impl Ord for CBVec {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
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

impl Ord for KBlade {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.n.len().cmp(&other.n.len()) {
            Ordering::Equal => self.n.cmp(&other.n),
            x => x
        }
    }
}
impl PartialEq for KBlade {
    fn eq(&self, other: &Self) -> bool {
        self.n.eq(&other.n)
    }
}
impl PartialOrd for KBlade {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}



impl ops::Add for MVec {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = self.blades.into_iter().peekable();
        let mut rhs = rhs.blades.into_iter().peekable();
        let mut new_blades: Vec<KBlade> = Vec::with_capacity(lhs.len() + rhs.len());
        while let (Some(lhs_next), Some(rhs_next)) = (lhs.peek(), rhs.peek()) {
            match lhs_next.cmp(&rhs_next) {
                Ordering::Less => new_blades.push(lhs.next().unwrap()),
                Ordering::Equal => {
                    let (lhs_next, rhs_next) = (lhs.next().unwrap(), rhs.next().unwrap());
                    let coefficient = lhs_next.c + rhs_next.c;
                    if !coefficient.is_zero() {
                        new_blades.push(KBlade { n: lhs_next.n, c: coefficient });
                    }
                }
                Ordering::Greater => new_blades.push(rhs.next().unwrap())
            }
        }
        new_blades.extend(lhs);
        new_blades.extend(rhs);
        MVec { blades: new_blades }
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
                    rhs.pop_front();
                }
                Ordering::Greater => {
                    sign_changes += lhs.len();
                    new_normal.push(rhs.pop_front().unwrap());
                }
            }
        }
        new_normal.append(&mut Vec::from(lhs));
        new_normal.append(&mut Vec::from(rhs));
        if sign_changes & 1 == 1 {
            coefficient = coefficient * -Scalar::ONE
        }
        KBlade { n: new_normal, c: coefficient }
    }
}

impl ops::Mul<Scalar> for KVec {
    type Output = Self;
    fn mul(self, rhs: Scalar) -> Self::Output {
        KVec { v: self.v * rhs, k: self.k }
    }
}
impl ops::Div<Scalar> for KVec {
    type Output = Self;
    fn div(self, rhs: Scalar) -> Self::Output {
        KVec { v: self.v / rhs, k: self.k }
    }
}

impl ops::Mul<Scalar> for MVec {
    type Output = Self;
    fn mul(self, rhs: Scalar) -> Self::Output {
        match rhs {
            Scalar::ZERO => MVec { blades: Vec::new() },
            s => {
                let mut v = self;
                for blade in &mut v.blades {
                    blade.c *= s.clone();
                }
                v
            }
        }
    }
}
impl ops::Div<Scalar> for MVec {
    type Output = Self;
    fn div(self, rhs: Scalar) -> Self::Output {
        self * rhs.mul_inv()
    }
}

impl ops::Mul for MVec {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let (lhs, rhs) = (self, rhs);
        let mut unsorted_blades: Vec<MVec> = Vec::with_capacity(lhs.blades.len() *
                                                                rhs.blades.len());
        for lhs_blade in &lhs.blades { for rhs_blade in &rhs.blades {
            unsorted_blades.push(MVec::from(lhs_blade.clone() * rhs_blade.clone()));
        }}
        //println!("{:#?}", unsorted_blades);
        merge_vec(unsorted_blades, |a, b| a + b, &MVec::from(KBlade::ZERO))
    }
}

impl ops::Neg for KBlade {
    type Output = Self;
    fn neg(self) -> Self::Output {
        KBlade { n: self.n, c: self.c * -Scalar::ONE }
    }
}
impl ops::Neg for MVec {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -Scalar::ONE
    }
}
impl ops::Sub for MVec {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl ops::Div for KBlade {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.mul_inv()
    }
}
impl ops::Div<KVec> for MVec {
    type Output = Self;
    fn div(self, rhs: KVec) -> Self::Output {
        self * rhs.mul_inv().v
    }
}



#[inline(always)]
pub fn inner(a: KVec, b: KVec) -> KVec {
    let grade = max(a.k, b.k) - min(a.k, b.k);
    (a.v * b.v).to_grade(grade)
}
#[inline(always)]
pub fn outer(a: KVec, b: KVec) -> KVec {
    let grade = a.k + b.k;
    (a.v * b.v).to_grade(grade)
}
#[inline(always)]
pub fn join(a: KVec, b: KVec) -> KVec {
    outer(a, b)
}
pub fn meet(a: KVec, b: KVec, space: KBlade) -> KVec {
    let e = KVec::from(space.clone());
    inner(outer(inner(a, e.clone()),
                inner(b, e)),
          KVec::from(space.mul_inv()))
}



impl KBlade {
    pub const ZERO: Self = KBlade { n: Vec::new(), c: Scalar::ZERO };
    pub const ONE: Self = KBlade { n: Vec::new(), c: Scalar::ONE };

    pub fn grade(&self) -> usize {
        self.n.len()
    }
    pub fn basis(&self) -> &Vec<CBVec> {
        &self.n
    }
    pub fn coefficient(&self) -> &Scalar {
        &self.c
    }
    pub fn coefficient_mut(&mut self) -> &mut Scalar {
        &mut self.c
    }
    pub fn mul_inv(self) -> Self {
        let square = (self.clone() * self.clone()).c;
        KBlade { n: self.n, c: self.c / square }
    }
}

impl KVec {
    pub fn blades(&self) -> &Vec<KBlade> {
        &self.v.blades
    }
    pub fn num_blades(&self) -> usize {
        self.v.blades.len()
    }
    pub fn coefficient_at(&self, index: usize) -> &Scalar {
        &self.v.blades[index].c
    }
    pub fn coefficient_at_mut(&mut self, index: usize) -> &mut Scalar {
        &mut self.v.blades[index].c
    }
    pub fn mul_inv(mut self) -> Self {
        let square = match inner(self.clone(), self.clone()).v.blades.pop() {
            Some(scalar) => scalar.c,
            None => Scalar::ZERO
        };
        KVec { v: self.v * square, k: self.k }
    }

    pub fn rename(self, name: &String) -> Self {
        KVec { v: self.v.rename(name), k: self.k }
    }
    pub fn with_name(name: &String, basis: &KBlade, grade: usize) -> Self {
        let dimensions = basis.grade();
        if grade == 0 {
            KVec { v: MVec::from(KBlade::from(Scalar::from(name.clone()))), k: 0 }
        }
        else if grade > dimensions {
            KVec { v: MVec::ZERO, k: grade }
        }
        else {
            let mut blades: Vec<KBlade> = Vec::with_capacity(choose(dimensions, grade));
            let mut element_indices: Vec<usize> = (0..grade).collect();
            let last_element: usize = grade - 1;

            while element_indices[last_element] < dimensions {
                let blade_id = blades.len() + 1;
                blades.push(KBlade {
                    n: element_indices.iter().map(|i| basis.n[*i].clone()).collect(),
                    c: Scalar::from(name.clone() + blade_id.to_string().as_str()) });

                // increment least significant index
                element_indices[last_element] += 1;
                let mut last_incremented_index = last_element;
                for i in (1..grade).rev() {
                    // (dimensions - 1) - indices[i] < (grade - 1) - i
                    // number of elements available < number of element slots on the right to fill
                    if dimensions + i < grade + element_indices[i] {
                        // increment element slot to left and later reset all slots to the right
                        element_indices[i - 1] += 1;
                        last_incremented_index = i - 1;
                    }
                }
                // reset all elements to the right so their indices are increasing
                for i in (last_incremented_index + 1)..grade {
                    element_indices[i] = element_indices[i - 1] + 1;
                }
            }

            KVec { v: MVec { blades }, k: grade }
        }
    }
    pub fn reverse_mul_order(self) -> Self {
        KVec { v: self.v.reverse_mul_order(), k: self.k }
    }
}

impl MVec {
    pub const ZERO: Self = MVec { blades: Vec::new() };

    pub fn blades(&self) -> &Vec<KBlade> {
        &self.blades
    }
    pub fn num_blades(&self) -> usize {
        self.blades.len()
    }
    pub fn coefficient_at(&self, index: usize) -> &Scalar {
        &self.blades[index].c
    }
    pub fn coefficient_at_mut(&mut self, index: usize) -> &mut Scalar {
        &mut self.blades[index].c
    }
    pub fn to_grade(self, grade: usize) -> KVec {
        let mut new_blades: Vec<KBlade> = Vec::with_capacity(self.blades.len());
        for blade in self.blades {
            if blade.grade() == grade {
                new_blades.push(blade);
            }
        }
        KVec { v: MVec { blades: new_blades }, k: grade }
    }

    pub fn rename(self, name: &String) -> Self {
        let mut renamed = self;
        for (i, blade) in renamed.blades.iter_mut().enumerate() {
            blade.c = Scalar::from(name.clone() + (i + 1).to_string().as_str());
        }
        renamed
    }
    pub fn with_name(name: &String, basis: &KBlade, max_grade: usize) -> Self {
        let template = MVec { blades: basis.n.iter().map(|e| KBlade::from(e.clone())).collect() };
        merge_all((0..max_grade).map(|i| template.clone().rename(&(i.to_string() + "_"))),
                  |a, b| a * b, &MVec { blades: Vec::new() }).rename(name)
    }
    pub fn reverse_mul_order(mut self) -> Self {
        for blade in &mut self.blades {
            if {
                // sign changes =
                // sum_{i = 0}^{n - 1} {n - (i + 1)} =
                // n*n - sum_{i = 1}^{n} {i} =
                // n*n - n(n + 1) / 2 =
                // n(n-1)/2
                let k = blade.grade();
                if k == 0 { false }
                else { (k * (k - 1)) & 2 != 0 }
            } {
                blade.c *= -Scalar::ONE;
            }
        }
        self
    }
}

impl fmt::Display for CBVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("e{}", self.id))
    }
}
impl fmt::Debug for CBVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl fmt::Display for KBlade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.c != Scalar::ONE {
            self.c.fmt(f)?;
        }
        for basis_vec in &self.n {
            basis_vec.fmt(f)?;
        }
        Ok(())
    }
}
impl fmt::Debug for KBlade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl fmt::Display for MVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.blades.len() {
            if i > 0 { f.write_str(" + ")?; }
            self.blades[i].fmt(f)?;
        }
        Ok(())
    }
}
impl fmt::Debug for MVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl fmt::Display for KVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.v.fmt(f)
    }
}
impl fmt::Debug for KVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
