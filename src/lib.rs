mod scal;
mod ga;
mod algebra_tools;
mod benchmarking;

#[cfg(test)]
mod tests {
    use super::*;
    use scal::*;
    use ga::*;
    use algebra_tools::*;
    use benchmarking::Stopwatch;

    #[test]
    fn basic_addition_test() {
        let watch = Stopwatch::new_running();
        let x = Scalar::from(Rational::from(5));
        let a = Scalar::Variable(String::from("a"));
        let xa = x.clone() + a.clone();
        let ax = a.clone() + x.clone();

        //println!("{} is ordered {:?} than {}", x, x.cmp(&a), a);
        assert_eq!(ax, xa);
        assert_eq!(ax.to_string(), xa.to_string());
        assert_eq!(ax.to_string(), String::from("(a + 5)"));
        println!("basic_addition_test runtime: {watch} nanoseconds");
    }

    #[test]
    fn basic_multiplication_test() {
        let watch = Stopwatch::new_running();
        let x = Scalar::from(Rational::from(5));
        let a = Scalar::Variable(String::from("a"));
        let xa = x.clone() * a.clone();
        let ax = a.clone() * x.clone();

        assert_eq!(ax, xa);
        assert_eq!(ax.to_string(), xa.to_string());
        assert_eq!(ax.to_string(), String::from("(a * 5)"));
        println!("basic_multiplication_test runtime: {watch} nanoseconds");
    }

    #[test]
    fn factorization_test() {
        // PROBLEM CHILD:
        //(r1 * ((r2 * x3) + (r3 * x2 * -1) + (r4 * x1)))
        //+ (r2 * -1 * ((r1 * x3) + (r3 * x1 * -1) + (r4 * x2 * -1)))
        //+ (r3 * (-1^2) * ((r1 * x2) + (r2 * x1 * -1) + (r4 * x3)))
        //+ (r4 * -1 * ((r1 * x1) + (r2 * x2) + (r3 * x3)))
        let watch = Stopwatch::new_running();
        let a = Scalar::from("a");
        let b = Scalar::from("b");
        let c = Scalar::from("c");
        let d = Scalar::from("d");
        let ab_ac_ad = a.clone() * b.clone() + a.clone() * c.clone() + a.clone() * d.clone();
        let a_bcd = a.clone() * (b.clone() + c.clone() + d.clone());
        assert_eq!(ab_ac_ad.simplified().to_string(), a_bcd.to_string());
        println!("factorization_test runtime: {watch} nanoseconds");
    }

    #[test]
    fn rotor_test() {
        let watch = Stopwatch::new_running();
        let dimensions: usize = 3;
        let basis = merge_all(
            (1..=dimensions)
                .map(|i| KBlade::from(CBVec { id: i as i32, square: S_ONE }))
                .collect(),
            |a, b| a * b, &KBlade::from(S_ONE));
        println!("{basis}");
        let r = MVec::with_name(&String::from("r"), &basis, 2);
        let r_inv = r.clone().reverse_mul_order();
        println!("r: {r}");
        println!("r reversed: {r_inv}");
        let x = MVec::from(KVec::with_name(&String::from("x"), &basis, 1));
        println!("x: {x}");
        println!("x + 0: {}", x.clone() + MVec::from(KBlade::from(S_ZERO)));
        let rx = r * x;
        println!("rx, {} blades: {rx}", rx.num_blades());
        let mut x_rot = rx * r_inv;
        let last_blade_idx = x_rot.num_blades() - 1;
        //*x_rot.coefficient_at_mut(last_blade_idx) = x_rot.coefficient_at(last_blade_idx).simplified();
        println!("x rotated, {} blades: {x_rot}", x_rot.num_blades());
        println!("rotor_test runtime: {watch} nanoseconds");
    }
}
