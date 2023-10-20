use std::{ops, cmp::Ordering};

////////////////////////////////////////////////////////////////////////////////////////////////////
//////////                            Mathematical Operations                             //////////
////////////////////////////////////////////////////////////////////////////////////////////////////

 pub trait AddIdentity: Sized + Clone + ops::Add<Output = Self> {
    const ZERO: Self;
 }
 pub trait MulIdentity: Sized + Clone + ops::Mul<Output = Self> {
    const ONE: Self;
 }

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

pub fn merge_self<T: Clone>(base: T, merges: u32, merge_func: fn(T, T) -> T, identity: &T) -> T {
    let mut product = identity.clone();
    let mut base_raised = base;
    let mut merges_remaining = merges;
    loop { // for i in 0..u32::BITS
        if (merges_remaining & 1) == 1 { // ith binary digit of merges_remaining is 1
            product = merge_func(product.clone(), base_raised.clone());
        }
        merges_remaining >>= 1;
        if merges_remaining == 0 {
            return product;
        }
        base_raised = merge_func(base_raised.clone(), base_raised); // base ^ (2 ^ i)
    }
}

/// Time complexity: O(log(exp))
pub fn exponentiate<T: MulIdentity>(base: T, exp: u32) -> T {
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

/// Returns the given root of the radicand (exact if Ok, truncated if Err)
/// Time complexity: O(log(radicand)log(index))
pub fn root_u64(radicand: u64, index: u32) -> Result<u64, u64> {
    let mut lower = 0;
    let mut upper = radicand;
    while lower <= upper {
        let mid = (upper + lower) >> 1;
        match radicand.cmp(&exponentiate(mid, index)) {
            Ordering::Less => upper = mid - 1,
            Ordering::Equal => return Ok(mid),
            Ordering::Greater => lower = mid + 1
        }
    }
    Err(upper)
}
pub fn sqrt_u64(radicand: u64) -> Result<u64, u64> {
    let mut lower = 0;
    let mut upper = radicand;
    while lower <= upper {
        let mid = (upper + lower) >> 1;
        match radicand.cmp(&(mid * mid)) {
            Ordering::Less => upper = mid - 1,
            Ordering::Equal => return Ok(mid),
            Ordering::Greater => lower = mid + 1
        }
    }
    Err(upper)
}
pub fn root_i64(radicand: i64, index: u32) -> Option<Result<i64, i64>> {
    if (radicand < 0) && (index & 1 == 0) {
        None
    }
    else {
        root_u64(radicand.unsigned_abs(), index).map_or_else(
            |root_trunc| Some(Err(root_trunc as i64 * radicand.signum())), 
            |root| Some(Ok(root as i64 * radicand.signum())))
    }
}

/// Time complexity: O(n^1/2)
pub fn prime_factor(n: u32) -> Option<u32> {
    if n % 2 == 0 {
        Some(2)
    }
    else if n % 3 == 0 {
        Some(3)
    }
    else {
        let max_factor = sqrt_u64(n as u64).map_or_else(|x| x + 1, |x| x) as u32;
        let mut i: u32 = 5;
        while i < max_factor {
            if n % i == 0 {
                return Some(i);
            }
            if n % (i + 2) == 0 {
                return Some(i + 2);
            }
            i += 6;
        }
        None
    }
}

pub fn simplify_root(mut radicand: u64, mut index: u32) -> (u64, u32) {
    let mut unsimplifiable_factors = 1;
    // worst case, iterates log(index) times
    loop {
        match prime_factor(index) {
            Some(index_factor) => {
                index /= index_factor;
                match root_u64(radicand, index_factor) {
                    Ok(root) => radicand = root,
                    Err(_) => unsimplifiable_factors *= index_factor
                }
            }
            None => break
        }
    }
    (radicand, index * unsimplifiable_factors)
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//////////                       Math-Associated Utility Functions                        //////////
////////////////////////////////////////////////////////////////////////////////////////////////////

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
pub fn merge_all<T: Clone>(arguments: impl ExactSizeIterator<Item = T>, 
                           merge_func: fn(T, T) -> T, identity: &T) -> T {
    merge_all_rec(arguments.len(), &mut {arguments}, merge_func, identity)
}
#[inline(always)]
pub fn merge_vec<T: Clone>(arguments: Vec<T>, merge_func: fn(T, T) -> T, identity: &T) -> T {
    merge_all_rec(arguments.len(), &mut arguments.into_iter(), merge_func, identity)
    //merge_vec_simple(arguments, merge_func, identity)
}

#[inline(always)]
pub fn add_all<T: AddIdentity>(terms: impl ExactSizeIterator<Item = T>) -> T {
    merge_all(terms, |x, y| x + y, &T::ZERO)
}
#[inline(always)]
pub fn mul_all<T: MulIdentity>(factors: impl ExactSizeIterator<Item = T>) -> T {
    merge_all(factors, |x, y| x * y, &T::ONE)
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
impl AddIdentity for u128 {
    const ZERO: Self = 0;
}
impl MulIdentity for u128 {
    const ONE: Self = 1;
}
impl AddIdentity for usize {
    const ZERO: Self = 0;
}
impl MulIdentity for usize {
    const ONE: Self = 1;
}
impl AddIdentity for i32 {
    const ZERO: Self = 0;
}
impl MulIdentity for i32 {
    const ONE: Self = 1;
}
impl AddIdentity for i64 {
    const ZERO: Self = 0;
}
impl MulIdentity for i64 {
    const ONE: Self = 1;
}
impl AddIdentity for i128 {
    const ZERO: Self = 0;
}
impl MulIdentity for i128 {
    const ONE: Self = 1;
}
impl AddIdentity for isize {
    const ZERO: Self = 0;
}
impl MulIdentity for isize {
    const ONE: Self = 1;
}