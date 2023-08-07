mod scal;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_addition_test() {
        let x = scal::Scalar::from(scal::Rational::from(5));
        let a = scal::Scalar::Variable(String::from("a"));
        let xa = x.clone() + a.clone();
        let ax = a.clone() + x.clone();

        //println!("{} is ordered {:?} than {}", x, x.cmp(&a), a);
        assert_eq!(ax, xa);
        assert_eq!(ax.to_string(), xa.to_string());
        assert_eq!(ax.to_string(), String::from("(a + 5)"));
    }

    #[test]
    fn basic_multiplication_test() {
        let x = scal::Scalar::from(scal::Rational::from(5));
        let a = scal::Scalar::Variable(String::from("a"));
        let xa = x.clone() * a.clone();
        let ax = a.clone() * x.clone();

        assert_eq!(ax, xa);
        assert_eq!(ax.to_string(), xa.to_string());
        assert_eq!(ax.to_string(), String::from("(a * 5)"));
    }

    #[test]
    fn ez_factorization_test() {
        let a = scal::Scalar::from("a");
        let b = scal::Scalar::from("b");
        let c = scal::Scalar::from("c");
        let ab_ac = a.clone() * b.clone() + a.clone() * c.clone();
        let a_bc = a.clone() * (b.clone() + c.clone());
        assert_eq!(ab_ac.simplified().to_string(), a_bc.to_string());
    }
}
