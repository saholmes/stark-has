use ark_pallas::Fr as F;
use ark_ff::{Field, One, Zero};
use ark_poly::{
    EvaluationDomain,
    GeneralEvaluationDomain,
    DenseUVPolynomial,
    Polynomial,
};
use ark_poly::polynomial::univariate::DensePolynomial;

pub fn deep_ali_merge_evals(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    omega: F,
    z: F,
) -> (Vec<F>, F, F) {

    let n = a_eval.len();
    assert!(n.is_power_of_two());

    let domain =
        GeneralEvaluationDomain::<F>::new(n)
            .expect("power-of-two domain");

    // Compute ω^j
    let mut omega_pows = Vec::with_capacity(n);
    let mut x = F::one();
    for _ in 0..n {
        omega_pows.push(x);
        x *= omega;
    }

    // Φ̃(ω^j)
    let mut phi_eval = vec![F::zero(); n];
    for i in 0..n {
        phi_eval[i] =
            a_eval[i] * s_eval[i]
            + e_eval[i]
            - t_eval[i];
    }

    // -------------------------
    // Compute Φ(z)
    // -------------------------

    let n_inv = F::from(n as u64).inverse().unwrap();

    let mut bary_sum = F::zero();
    for j in 0..n {
        let inv = (z - omega_pows[j]).inverse().unwrap();
        bary_sum += phi_eval[j] * omega_pows[j] * inv;
    }

    let phi_z = n_inv * bary_sum;

    // Z_H(z)
    let zh_z = z.pow([n as u64]) - F::one();

    let c_star = phi_z / zh_z;

    // -------------------------
    // Compute f0
    // -------------------------

    let mut f0_eval = Vec::with_capacity(n);
    for j in 0..n {
        let denom = omega_pows[j] - z;
        f0_eval.push(phi_eval[j] * denom.inverse().unwrap());
    }

    // -------------------------
    // Enforce rate 1/32
    // -------------------------

    let mut coeffs = domain.ifft(&f0_eval);

    let d0 = n / 32;
    coeffs.truncate(d0);

    let poly = DensePolynomial::from_coefficients_vec(coeffs);

    let f0_low_rate = domain.fft(poly.coeffs());

    (f0_low_rate, z, c_star)
}
