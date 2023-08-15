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
    let mut base_to_i = identity;
    for i in 0..u32::BITS {
        if ((exp >> i) & 1) == 1 {
            product = product * base_to_i;
        }
        base_to_i = base_to_i * base;
    }
    product
}

fn merge_rec<T: Clone>(arguments: &mut Vec<T>, size: usize,
                       merge_func: fn(T, T) -> T, identity: &T) -> T {
    if size > 2 {
        let right_size = size / 2;
        let rhs = merge_rec(arguments, right_size, merge_func, identity);
        let lhs = merge_rec(arguments, size - right_size, merge_func, identity);
        merge_func(lhs, rhs)
    }
    else {
        match (arguments.pop(), arguments.pop()) {
            (Some(rhs), Some(lhs)) => merge_func(lhs, rhs),
            (Some(x), None) | (None, Some(x)) => x,
            (None, None) => identity.clone()
        }
    }
}

/// Merges all arguments into one, preserving their initial ordering.
/// May differ from merge_func(arg1, merge_func(arg2, ...)) if merge_func isn't associative.
pub fn merge_all<T: Clone>(arguments: Vec<T>, merge_func: fn(T, T) -> T, identity: &T) -> T {
    let init_size = arguments.len();
    merge_rec(&mut {arguments}, init_size, merge_func, identity)
}