mod scal;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition_test() {
        let xr = scal::Rational::from(5);
        let x = scal::Scalar::from(xr);
        println!("{}", xr > scal::Rational::from(3));
        let a = scal::Scalar::Variable(String::from("a"));
        let xa = x.clone() + a.clone();
        let ax = a.clone() + x.clone();
        println!("{}", xa);
        println!("{}", ax);
        assert!(ax == xa);
    }
}
