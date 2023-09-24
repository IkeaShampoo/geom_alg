use std::{ops, iter::Step, cmp::Ordering};

/*
pub trait Int = Sized + Copy + Ord + Step +
    ops::Add + ops::Sub + ops::Mul + ops::Div + ops::Rem +
    ops::AddAssign + ops::SubAssign + ops::MulAssign + ops::DivAssign + ops::RemAssign +
    ops::Shl + ops::Shr + ops::ShlAssign + ops::ShrAssign +
    ops::BitOr + ops::BitAnd + ops::BitXor +
    ops::BitOrAssign + ops::BitAndAssign + ops::BitXorAssign;
 */

 pub trait AddIdentity: Sized + Clone + ops::Add<Output = Self> {
    const ZERO: Self;
 }
 pub trait MulIdentity: Sized + Clone + ops::Mul<Output = Self> {
    const ONE: Self;
 }
 pub trait Ring = Clone + AddIdentity + ops::Sub<Output = Self> + ops::Neg + MulIdentity + 
    ops::AddAssign + ops::SubAssign + ops::MulAssign;

pub fn factorial(first: usize, last: usize) -> usize {
    let mut result = 1;
    for i in first..=last {
        result *= i;
    }
    result
}
pub fn choose(n: usize, k: usize) -> usize {
    factorial(n - k + 1, n) / factorial(1, k)
}

pub fn exponentiate<T: Clone + MulIdentity> (base: T, exp: u32) -> T {
    let mut product = T::ONE;
    let mut base_raised = base;
    let mut shifted_exp = exp;
    loop { // for i in 0..u32::BITS
        if (shifted_exp & 1) == 1 { // ith binary digit of exp is 1
            product = product.clone() * base_raised.clone();
        }
        shifted_exp >>= 1;
        if shifted_exp == 0 {
            return product;
        }
        base_raised = base_raised.clone() * base_raised; // base ^ (2 ^ i)
    }
}

pub fn root_u64(radicand: u64, radical: u32) -> Option<u64> {
    let mut lower = 0;
    let mut upper = radicand;
    while lower <= upper {
        let mid = (upper + lower) >> 1;
        let mid_raised = exponentiate(mid, radical);
        match radicand.cmp(&mid_raised) {
            Ordering::Less => upper = mid - 1,
            Ordering::Equal => return Some(mid),
            Ordering::Greater => lower = mid + 1
        }
    }
    None
}

pub fn root_i64(radicand: i64, radical: u32) -> Option<i64> {
    if (radicand < 0) && (radical & 1 == 1) {
        None
    }
    else {
        root_u64(radicand.unsigned_abs(), radical)
            .map_or(None, |root| Some(root as i64 * radicand.signum()))
    }
}

// logarithmically-complex implementation of merge
pub fn merge_all_rec<T: Clone>(size: usize, arguments: &mut dyn Iterator<Item = T>,
                               merge_func: fn(T, T) -> T, identity: &T) -> T {
    if size > 2 {
        let left_size = size / 2;
        merge_func(merge_all_rec(left_size, arguments, merge_func, identity), 
                   merge_all_rec(size - left_size, arguments, merge_func, identity))
    }
    else {
        match (arguments.next(), arguments.next()) {
            (Some(lhs), Some(rhs)) => merge_func(lhs, rhs),
            (Some(x), None) | (None, Some(x)) => x,
            (None, None) => identity.clone()
        }
    }
}

// Linear time complexity implementation
#[inline(always)]
pub fn merge_seq<T>(arguments: impl Iterator<Item = T>, merge_func: fn(T, T) -> T, 
                    identity: T) -> T{
    let mut result = identity;
    for arg in arguments {
        result = merge_func(result, arg);
    }
    result
}

/// Logarithmic time complexity implementations
/// Merges all arguments into one, preserving their initial ordering.
/// May differ from merge_func(arg1, merge_func(arg2, ...)) if merge_func isn't associative.

#[inline(always)]
pub fn merge_all<T: Clone>(size: usize, arguments: impl Iterator<Item = T>, 
                           merge_func: fn(T, T) -> T, identity: &T) -> T {
    merge_all_rec(size, &mut {arguments}, merge_func, identity)
}
#[inline(always)]
pub fn merge_vec<T: Clone>(arguments: Vec<T>, merge_func: fn(T, T) -> T, identity: &T) -> T {
    merge_all_rec(arguments.len(), &mut arguments.into_iter(), merge_func, identity)
    //merge_vec_simple(arguments, merge_func, identity)
}

#[inline(always)]
pub fn add_all<T: AddIdentity>(size: usize, terms: impl Iterator<Item = T>) -> T {
    merge_all(size, terms, |x, y| x + y, &T::ZERO)
}
#[inline(always)]
pub fn mul_all<T: MulIdentity>(size: usize, factors: impl Iterator<Item = T>) -> T {
    merge_all(size, factors, |x, y| x * y, &T::ONE)
}



impl AddIdentity for u32 {
    const ZERO: Self = 0;
}
impl MulIdentity for u32 {
    const ONE: Self = 1;
}
impl AddIdentity for u64 {
    const ZERO: Self = 0;
}
impl MulIdentity for u64 {
    const ONE: Self = 1;
}
impl AddIdentity for usize {
    const ZERO: Self = 0;
}
impl MulIdentity for usize {
    const ONE: Self = 1;
}