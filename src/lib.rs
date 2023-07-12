mod scal;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        let x = scal::Rational::from(5);
        println!("{}", x > scal::Rational::from(3));
        let a = scal::Scalar::Variable(String::from("a"));
        assert_eq!(result, 4);
    }
}
