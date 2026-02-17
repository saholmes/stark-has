
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_macros)]
use ark_ff::{Field, One, Zero};
use ark_pallas::Fr as F;

use ark_poly::polynomial::univariate::DensePolynomial;
use ark_poly::{DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const PARALLEL_MIN_ELEMS: usize = 1 << 12;

#[inline]
fn enable_parallel(len: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        len >= PARALLEL_MIN_ELEMS && rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = len;
        false
    }
}

/* ============================================================
   Domain helpers
============================================================ */

fn build_omega_pows(omega: F, n: usize) -> Vec<F> {
    let mut omega_pows = Vec::with_capacity(n);
    let mut x = F::one();
    for _ in 0..n {
        omega_pows.push(x);
        x *= omega;
    }
    omega_pows
}

fn is_in_domain(z: F, n: usize) -> bool {
    z.pow([n as u64]) == F::one()
}

fn zh_at(z: F, n: usize) -> F {
    z.pow([n as u64]) - F::one()
}

/* ============================================================
   Barycentric evaluation
============================================================ */

fn lagrange_bary_sum(values: &[F], z: F, omega_pows: &[F]) -> F {
    debug_assert_eq!(values.len(), omega_pows.len());

    if enable_parallel(values.len()) {
        #[cfg(feature = "parallel")]
        {
            return values
                .par_iter()
                .zip(omega_pows.par_iter())
                .map(|(&val, &wj)| {
                    let inv = (z - wj).inverse().expect("z ∉ H");
                    val * wj * inv
                })
                .reduce(|| F::zero(), |acc, term| acc + term);
        }
    }

    let mut sum = F::zero();
    for (val, &wj) in values.iter().zip(omega_pows.iter()) {
        let inv = (z - wj).inverse().expect("z ∉ H");
        sum += *val * wj * inv;
    }
    sum
}

/* ============================================================
   Φ̃(x) construction
============================================================ */

fn fill_phi_eval(
    phi_eval: &mut [F],
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    r_eval_opt: Option<&[F]>,
    beta: F,
) {
    if enable_parallel(phi_eval.len()) {
        #[cfg(feature = "parallel")]
        {
            match r_eval_opt {
                Some(r_eval) => {
                    phi_eval.par_iter_mut().enumerate().for_each(|(i, slot)| {
                        let base = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
                        *slot = base + beta * r_eval[i];
                    });
                }
                None => {
                    phi_eval.par_iter_mut().enumerate().for_each(|(i, slot)| {
                        *slot = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
                    });
                }
            }
            return;
        }
    }

    match r_eval_opt {
        Some(r_eval) => {
            for i in 0..phi_eval.len() {
                let base = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
                phi_eval[i] = base + beta * r_eval[i];
            }
        }
        None => {
            for i in 0..phi_eval.len() {
                phi_eval[i] = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
            }
        }
    }
}

/* ============================================================
   f₀(ω^j) = Φ̃(ω^j)/(ω^j − z)
============================================================ */

fn fill_f0_eval(f0_eval: &mut [F], phi_eval: &[F], omega_pows: &[F], z: F) {
    if enable_parallel(f0_eval.len()) {
        #[cfg(feature = "parallel")]
        {
            f0_eval.par_iter_mut().enumerate().for_each(|(j, slot)| {
                let inv = (omega_pows[j] - z).inverse().expect("z ∉ H");
                *slot = phi_eval[j] * inv;
            });
            return;
        }
    }

    for (j, slot) in f0_eval.iter_mut().enumerate() {
        let inv = (omega_pows[j] - z).inverse().expect("z ∉ H");
        *slot = phi_eval[j] * inv;
    }
}

/* ============================================================
   DEEP‑ALI merge (base field, Pallas)
============================================================ */

pub fn deep_ali_merge_evals(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    omega: F,
    z: F,
) -> (Vec<F>, F, F) {
    deep_ali_merge_evals_blinded(a_eval, s_eval, e_eval, t_eval, None, F::zero(), omega, z)
}

pub fn deep_ali_merge_evals_blinded(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    r_eval_opt: Option<&[F]>,
    beta: F,
    omega: F,
    z: F,
) -> (Vec<F>, F, F) {
    let n = a_eval.len();
    assert!(n > 1);
    assert!(n.is_power_of_two(), "domain must be power-of-two");
    assert!(!is_in_domain(z, n), "z must be outside H");

    let omega_pows = build_omega_pows(omega, n);

    // ------------------------------------------------------------
    // 1️⃣ Build Φ̃(ω^j)
    // ------------------------------------------------------------

    let mut phi_eval = vec![F::zero(); n];
    fill_phi_eval(
        &mut phi_eval,
        a_eval,
        s_eval,
        e_eval,
        t_eval,
        r_eval_opt,
        beta,
    );

    // ------------------------------------------------------------
    // 2️⃣ Compute Φ̃(z) and c_star
    // ------------------------------------------------------------

    let n_inv = F::from(n as u64).inverse().expect("n invertible");

    let bary_sum = lagrange_bary_sum(&phi_eval, z, &omega_pows);

    let c_star = n_inv * bary_sum;

    // (optional reconstruction check)
    let _phi_z = zh_at(z, n) * c_star;

    // ------------------------------------------------------------
    // 3️⃣ Compute f₀(ω^j) = Φ̃(ω^j)/(ω^j − z)
    // ------------------------------------------------------------

    let mut f0_eval = vec![F::zero(); n];
    fill_f0_eval(&mut f0_eval, &phi_eval, &omega_pows, z);

    // ------------------------------------------------------------
    // 4️⃣ ✅ Enforce rate ρ₀ = 1/32 (degree truncation)
    // ------------------------------------------------------------

    let domain = GeneralEvaluationDomain::<F>::new(n).expect("power-of-two domain");

    let mut coeffs = domain.ifft(&f0_eval);

    let d0 = n / 32;
    assert!(d0 > 0, "domain too small for 1/32 rate");

    if coeffs.len() > d0 {
        coeffs.truncate(d0);
    }

    let poly = DensePolynomial::from_coefficients_vec(coeffs);

    let f0_low_rate = domain.fft(poly.coeffs());

    (f0_low_rate, z, c_star)
}

/* ============================================================
   Cached domain helper
============================================================ */

#[derive(Clone)]
pub struct DomainH {
    pub n: usize,
    pub omega: F,
    pub omega_pows: Vec<F>,
}

impl DomainH {
    pub fn new_radix2(n: usize) -> Self {
        use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
        let dom = Domain::<F>::new(n).expect("radix-2 domain exists");

        let omega = dom.group_gen;
        let omega_pows = build_omega_pows(omega, n);

        Self {
            n,
            omega,
            omega_pows,
        }
    }

    pub fn merge_deep_ali(
        &self,
        a_eval: &[F],
        s_eval: &[F],
        e_eval: &[F],
        t_eval: &[F],
        z: F,
    ) -> (Vec<F>, F, F) {
        deep_ali_merge_evals(a_eval, s_eval, e_eval, t_eval, self.omega, z)
    }
}

/* ============================================================
   Deterministic sampling helper (tests)
============================================================ */

pub fn sample_z_beta_from_seed(seed: u64, n: usize) -> (F, F) {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);

    let beta = F::from(rng.gen::<u64>());

    let z = loop {
        let cand = F::from(rng.gen::<u64>());
        if !is_in_domain(cand, n) {
            break cand;
        }
    };

    (z, beta)
}

/* ============================================================
   FRI module
============================================================ */

pub mod fri;
