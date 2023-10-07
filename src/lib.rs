//#![feature(trait_alias)]
//#![feature(step_trait)]

pub mod scal;
pub mod ga;
pub mod simpl;
mod algebra_tools;
mod benchmarking;
mod rational;

/* TODO
 *  Rewrite Exponential::ord() to order exponentiated rationals firstly by whether they are rational
 *      and secondly by the magnitude of their exponents
 *  Rewrite Sum::add() to combine differently-rational-scaled copies of the same expression
 *  Rewrite Product::mul() to combine rationals exponentiated to powers of the same magnitude
 *  Give Scalar::is_zero() ability to put exponentiated rationals in a canonical form
 *  Move Scalar::fragment() to another module
 *  Make a second version of Simplifier that doesn't cap complexity
 *  Improve on the distribution rule for simplification
 *  Place limits on simplification runtime
 *  Give options to limit simplification rules
 */

#[cfg(test)]
mod tests {
    use super::*;
    use scal::*;
    use rational::*;
    use ga::*;
    use algebra_tools::*;
    use benchmarking::Stopwatch;

    #[test]
    fn basic_addition_test() {
        let watch = Stopwatch::new_running();
        let x = Scalar::from(Rational::from(5));
        let a = Scalar::from("a");
        let xa = x.clone() + a.clone();
        let ax = a.clone() + x.clone();
        let xabc = xa.clone() + (Scalar::from("b") * Scalar::from("c"));

        println!("{} is ordered {:?} than {}", x, x.cmp(&a), a);
        println!("{xabc}");
        assert_eq!(ax, xa);
        assert_eq!(ax.to_string(), xa.to_string());
        assert_eq!(ax.to_string(), String::from("(5 + a)"));
        println!("basic_addition_test runtime: {watch} nanoseconds");
    }

    #[test]
    fn basic_multiplication_test() {
        let watch = Stopwatch::new_running();
        let x = Scalar::from(Rational::from(5));
        let a = Scalar::from("a");
        let xa = x.clone() * a.clone();
        let ax = a.clone() * x.clone();

        assert_eq!(ax, xa);
        assert_eq!(ax.to_string(), xa.to_string());
        assert_eq!(ax.to_string(), String::from("(5 * a)"));
        println!("basic_multiplication_test runtime: {watch} nanoseconds");
    }

    #[test]
    fn ez_factorization_test() {
        let watch = Stopwatch::new_running();
        let a = Scalar::from("a");
        let b = Scalar::from("b");
        let c = Scalar::from("c");
        let d = Scalar::from("d");
        let ab_ac_ad = a.clone() * b.clone() + a.clone() * c.clone() + a.clone() * d.clone();
        let a_bcd = a.clone() * (b.clone() + c.clone() + d.clone());
        //assert_eq!(ab_ac_ad.simplified().to_string(), a_bcd.to_string());
        println!("ez_factorization_test runtime: {watch} nanoseconds");
    }

    #[test]
    fn factorization_test() {
        // PROBLEM CHILD:
        //(r1 * ((r2 * x3) + (r3 * x2 * -1) + (r4 * x1)))
        //+ (r2 * -1 * ((r1 * x3) + (r3 * x1 * -1) + (r4 * x2 * -1)))
        //+ (r3 * (-1^2) * ((r1 * x2) + (r2 * x1 * -1) + (r4 * x3)))
        //+ (r4 * -1 * ((r1 * x1) + (r2 * x2) + (r3 * x3)))
        let watch = Stopwatch::new_running();
        let (a, b, c, d) = (Scalar::from("a"), Scalar::from("b"), 
                            Scalar::from("c"), Scalar::from("d"));
        let (x, y, z) = (Scalar::from("e"), Scalar::from("f"), Scalar::from("g"));
        let prod1 = a.clone() * ((b.clone() * z.clone()) + 
                                 (c.clone() * y.clone() * -Scalar::ONE) + 
                                 (d.clone() * x.clone()));
        //println!("\nPROD1: {}", prod1.simplified());
        let prod2 = b.clone() * ((a.clone() * z.clone()) + 
                                 (c.clone() * x.clone() * -Scalar::ONE) + 
                                 (d.clone() * y.clone() * -Scalar::ONE)) * -Scalar::ONE;
        //println!("\nPROD2: {}", prod2.simplified());
        let prod3 = c.clone() * ((a.clone() * y.clone() * ((-Scalar::ONE) ^ Rational::from(2))) + 
                                 (b.clone() * x.clone() * -Scalar::ONE) + 
                                 (d.clone() * z.clone()));
        //println!("\nPROD3: {}", prod3.simplified());
        let prod4 = d.clone() * ((a * x) + (b * y) + (c * z)) * -Scalar::ONE;
        //println!("\nPROD4: {}", prod4.simplified());

        //let all = (((prod1 + prod2).simplified() + prod3).simplified() + prod4).simplified();
        //println!("\nALL: {all}", );
        //println!("\nALL: {}", ((prod1 + prod2).simplified() + (prod3 + prod4).simplified()).simplified());
        //assert_eq!(all.to_string(), "0");

        println!("factorization_test runtime: {watch} nanoseconds");
    }

    #[test]
    fn rotor_test() {
        let watch = Stopwatch::new_running();
        let dimensions: usize = 3;
        let basis = merge_all_rec(dimensions, &mut (1..=dimensions)
                .map(|i| KBlade::from(CBVec { id: i as i32, square: Scalar::ONE })),
            |a, b| a * b, &KBlade::ONE);
        println!("{basis}");
        let r = MVec::with_name(&String::from("r"), &basis, 2);
        let r_inv = r.clone().reverse_mul_order();
        println!("r: {r}");
        println!("r reversed: {r_inv}");
        let x = MVec::from(KVec::with_name(&String::from("x"), &basis, 1));
        println!("x: {x}");
        println!("x + 0: {}", x.clone() + MVec::ZERO);
        let rx = r * x;
        println!("rx, {} blades: {rx}", rx.num_blades());
        let mut x_rot = rx * r_inv;
        let last_blade_idx = x_rot.num_blades() - 1;
        println!("x rotated, {} blades: {x_rot}", x_rot.num_blades());
        println!("rotor_test runtime: {watch} nanoseconds");
    }

    #[test]
    fn root_test() {
        let base = Rational::new(-27, 64);
        let exp = Rational::new(2, 3);
        if let Some(root) = base ^ exp {
            println!("{root}");
        }

        let (r, i) = simplify_root(27, 6);
        println!("{r} ^ (1/{i})");
    }
}
