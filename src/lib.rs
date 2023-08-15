mod scal;
mod ga;
mod algebra_tools;

#[cfg(test)]
mod tests {
    use super::*;
    use scal::*;
    use ga::*;
    use algebra_tools::*;

    #[test]
    fn basic_addition_test() {
        let x = Scalar::from(Rational::from(5));
        let a = Scalar::Variable(String::from("a"));
        let xa = x.clone() + a.clone();
        let ax = a.clone() + x.clone();

        //println!("{} is ordered {:?} than {}", x, x.cmp(&a), a);
        assert_eq!(ax, xa);
        assert_eq!(ax.to_string(), xa.to_string());
        assert_eq!(ax.to_string(), String::from("(a + 5)"));
    }

    #[test]
    fn basic_multiplication_test() {
        let x = Scalar::from(Rational::from(5));
        let a = Scalar::Variable(String::from("a"));
        let xa = x.clone() * a.clone();
        let ax = a.clone() * x.clone();

        assert_eq!(ax, xa);
        assert_eq!(ax.to_string(), xa.to_string());
        assert_eq!(ax.to_string(), String::from("(a * 5)"));
    }

    #[test]
    fn ez_factorization_test() {
        let a = Scalar::from("a");
        let b = Scalar::from("b");
        let c = Scalar::from("c");
        let d = Scalar::from("d");
        let ab_ac_ad = a.clone() * b.clone() + a.clone() * c.clone() + a.clone() * d.clone();
        let a_bcd = a.clone() * (b.clone() + c.clone() + d.clone());
        assert_eq!(ab_ac_ad.simplified().to_string(), a_bcd.to_string());
    }

    #[test]
    fn rotor_test() {
        let dimensions: usize = 3;
        let basis = merge_all(
            (1..=dimensions)
                .map(|i| KBlade::from(CBVec { id: i as i32, square: S_ONE.clone() }))
                .collect(),
            |a, b| a * b, &KBlade::from(S_ONE.clone()));
        println!("{basis}");
        let r = MVec::with_name(&String::from("r"), &basis, 2);
        println!("{}", r.blades()[3] == r.blades()[4]);
        let r_inv = r.clone().reverse_mul_order();
        println!("r: {r} \nr reversed: {r_inv}");
    }
}
