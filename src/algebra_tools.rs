use std::ops;

pub fn factorial(first: usize, last: usize) -> usize {
    let mut result: usize = 1;
    for i in first..=last {
        result *= i;
    }
    result
}
pub fn choose(n: usize, k: usize) -> usize {
    factorial(n - k + 1, n) / factorial(1, k)
}

pub fn exponentiate<T: ops::Mul<Output = T> + Copy>(identity: T, base: T, exp: u32) -> T {
    let mut product = identity;
    let mut base_raised = base;
    let mut shifted_exp = exp;
    while shifted_exp != 0 { // for i in 0..u32::BITS
        if (shifted_exp & 1) == 1 { // ith binary digit of exp is 1
            product = product * base_raised;
        }
        base_raised = base_raised * base_raised; // base ^ (2 ^ i)
        shifted_exp >>= 1;
    }
    product
}

// linearly-complex implementation of merge
fn merge_linear<T: Clone>(arguments: Vec<T>, merge_func: fn(T, T) -> T, identity: &T) -> T {
    let mut result = identity.clone();
    for arg in arguments {
        result = merge_func(result, arg);
    }
    result
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
    //merge_linear(arguments, merge_func, identity)
}