
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_macros)]
//use ark_pallas::Fr as F;
use ark_goldilocks::Goldilocks as F;
use ark_serialize::CanonicalSerialize;
use rand::{rngs::StdRng, Rng, SeedableRng};

use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::{
    EvaluationDomain, GeneralEvaluationDomain,
};

// ✅ NEW: cubic DEEP tower
use crate::deep_tower::Fp3;

// ✅ REAL MERKLE API ONLY
use merkle::{
    MerkleChannelCfg,
    MerkleTreeChannel,
    MerkleOpening,
};

// ✅ TRANSCRIPT
use transcript::{default_params as transcript_params, Transcript};

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

#[cfg(feature = "fri_bench_log")]
#[allow(unused_macros)]
macro_rules! logln {
    ($($tt:tt)*) => { eprintln!($($tt)*); }
}
#[cfg(not(feature = "fri_bench_log"))]
macro_rules! logln {
    ($($tt:tt)*) => {};
}

mod ds {
    pub const FRI_SEED: &[u8] = b"FRI/seed";
    pub const FRI_INDEX: &[u8] = b"FRI/index";
    pub const FRI_Z_L: &[u8] = b"FRI/z/l";
    pub const FRI_Z_L_1: &[u8] = b"FRI/z/l/1";
    pub const FRI_Z_L_2: &[u8] = b"FRI/z/l/2";
    pub const FRI_LEAF: &[u8] = b"FRI/leaf";
}

fn tr_hash_fields_tagged(tag: &[u8], fields: &[F]) -> F {
    let mut tr = Transcript::new(b"FRI/FS", transcript_params());
    tr.absorb_bytes(tag);
    for &x in fields {
        tr.absorb_field(x);
    }
    tr.challenge(b"out")
}

#[derive(Clone, Copy, Debug)]
pub struct FriDomain {
    pub omega: F,
    pub size: usize,
}

impl FriDomain {
    pub fn new_radix2(size: usize) -> Self {
        let dom = Domain::<F>::new(size).expect("radix-2 domain exists");
        Self { omega: dom.group_gen, size }
    }
}

fn build_z_pows(z_l: F, m: usize) -> Vec<F> {
    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m {
        z_pows.push(acc);
        acc *= z_l;
    }
    z_pows
}

// -----------------------------------------------------------------------------
// ✅ NEW: Goldilocks-safe DEEP quotient (Fp³)
// -----------------------------------------------------------------------------

fn compute_q_layer_fp3(
    f_l: &[F],
    z: Fp3,
    omega: F,
) -> Vec<Fp3> {
    let n = f_l.len();
    let mut q = Vec::with_capacity(n);

    let f0 = Fp3::from_base(f_l[0]);
    let mut x = F::one();

    for i in 0..n {
        let num   = Fp3::from_base(f_l[i]) - f0;
        let denom = Fp3::from_base(x) - z;
        q.push(num * denom.inv()); // ✅ Fp³ identity
        x *= omega;
    }
    q
}

// -----------------------------------------------------------------------------
// ✅ Legacy base-field DEEP (for non-Goldilocks fields)
// -----------------------------------------------------------------------------

fn compute_q_layer_base(
    f_l: &[F],
    z: F,
    omega: F,
) -> Vec<F> {
    let n = f_l.len();
    let mut q = Vec::with_capacity(n);

    let mut x = F::one();
    for i in 0..n {
        q.push((f_l[i] - f_l[0]) * (x - z).inverse().unwrap());
        x *= omega;
    }

    q
}



fn dot_with_z_pows(chunk: &[F], z_pows: &[F]) -> F {
    debug_assert_eq!(chunk.len(), z_pows.len());
    let mut s = F::zero();
    for (val, zp) in chunk.iter().zip(z_pows.iter()) {
        s += *val * *zp;
    }
    s
}

fn fold_layer_sequential(f_l: &[F], z_pows: &[F], m: usize) -> Vec<F> {
    f_l
        .chunks(m)
        .map(|chunk| dot_with_z_pows(chunk, z_pows))
        .collect()
}

#[cfg(feature = "parallel")]
fn fold_layer_parallel(f_l: &[F], z_pows: &[F], m: usize) -> Vec<F> {
    f_l
        .par_chunks(m)
        .map(|chunk| dot_with_z_pows(chunk, z_pows))
        .collect()
}

fn fill_repeated_targets(target: &mut [F], src: &[F], m: usize) {
    for (bucket, chunk) in src.iter().zip(target.chunks_mut(m)) {
        for item in chunk {
            *item = *bucket;
        }
    }
}

fn merkle_depth(leaves: usize, arity: usize) -> usize {
    assert!(arity >= 2, "Merkle arity must be ≥ 2");

    let mut depth = 1;
    let mut cur = leaves;

    while cur > arity {
        cur = (cur + arity - 1) / arity;
        depth += 1;
    }

    depth
}


#[cfg(feature = "parallel")]
fn fill_repeated_targets_parallel(target: &mut [F], src: &[F], m: usize) {
    target
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(idx, chunk)| {
            let bucket = src[idx];
            for item in chunk {
                *item = bucket;
            }
        });
}

pub fn fri_sample_z_ell(seed_z: u64, level: usize, domain_size: usize) -> F {
    let fused = tr_hash_fields_tagged(
        ds::FRI_Z_L,
        &[F::from(seed_z), F::from(level as u64), F::from(domain_size as u64)],
    );
    let mut seed_bytes = [0u8; 32];
    fused.serialize_uncompressed(&mut seed_bytes[..]).expect("serialize");
    let mut rng = StdRng::from_seed(seed_bytes);

    let exp_bigint = <F as PrimeField>::BigInt::from(domain_size as u64);

    let mut tries = 0usize;
    const MAX_TRIES: usize = 1_000;
    loop {
        let cand = F::from(rng.gen::<u64>());
        if !cand.is_zero() && cand.pow(exp_bigint.as_ref()) != F::one() {
            return cand;
        }
        tries += 1;
        if tries >= MAX_TRIES {
            let fallback = F::from(seed_z.wrapping_add(level as u64).wrapping_add(7));
            if fallback.pow(exp_bigint.as_ref()) != F::one() {
                return fallback;
            }
            return F::from(11u64);
        }
    }
}

pub fn compute_s_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    let n = f_l.len();
    assert!(n % m == 0);
    let n_next = n / m;

    let z_pows = build_z_pows(z_l, m);

    // First compute the folded values (same as fri_fold_layer_impl)
    let mut folded = vec![F::zero(); n_next];
    for b in 0..n_next {
        let mut acc = F::zero();
        for j in 0..m {
            acc += f_l[b + j * n_next] * z_pows[j];
        }
        folded[b] = acc;
    }

    // Then repeat each folded value m times to match original domain
    let mut s_per_i = vec![F::zero(); n];
    for b in 0..n_next {
        for j in 0..m {
            s_per_i[b + j * n_next] = folded[b];
        }
    }

    s_per_i
}

fn layer_sizes_from_schedule(n0: usize, schedule: &[usize]) -> Vec<usize> {
    let mut sizes = Vec::with_capacity(schedule.len() + 1);
    let mut n = n0;
    sizes.push(n);
    for &m in schedule {
        assert!(n % m == 0, "schedule not dividing domain size");
        n /= m;
        sizes.push(n);
    }
    sizes
}

fn hash_leaf(f: F, s: F, q: F) -> F {
    tr_hash_fields_tagged(ds::FRI_LEAF, &[f, s, q])
}

fn hash_node(children: &[F]) -> F {
    tr_hash_fields_tagged(b"FRI/MERKLE/NODE", children)
}

fn verify_merkle_opening_explicit(
    root: F,
    index: usize,
    arity: usize,
    path: &[Vec<F>], // one Vec<F> per level
    leaf_hash: F,
) -> bool {
    let mut cur = leaf_hash;
    let mut idx = index;

    for siblings in path {
        let mut children = siblings.clone();

        let pos = idx % arity;
        if pos >= children.len() {
            return false;
        }

        children.insert(pos, cur);
        cur = hash_node(&children);

        idx /= arity;
    }

    cur == root
}

fn verify_local_check_fold(
    i: usize,
    m: usize,
    n_layer: usize,
    child_leaf_i: CombinedLeaf,
    parent_f_b: F,
) -> bool {
    let n_next = n_layer / m;
    let b = i % n_next;
    child_leaf_i.s == parent_f_b
}

fn fs_seed_from_roots(roots: &[F]) -> F {
    tr_hash_fields_tagged(ds::FRI_SEED, roots)
}

fn index_from_seed(seed_f: F, n_pow2: usize) -> usize {
    assert!(n_pow2.is_power_of_two());
    let mask = n_pow2 - 1;
    let mut seed_bytes = [0u8; 32];
    seed_f.serialize_uncompressed(&mut seed_bytes[..]).unwrap();
    let mut rng = StdRng::from_seed(seed_bytes);
    (rng.gen::<u64>() as usize) & mask
}

fn index_seed(roots_seed: F, ell: usize, q: usize) -> F {
    tr_hash_fields_tagged(
        ds::FRI_INDEX,
        &[roots_seed, F::from(ell as u64), F::from(q as u64)],
    )
}

#[derive(Clone, Copy, Debug)]
pub struct CombinedLeaf {
    pub f: F,   // fℓ(x)
    pub s: F,   // folded value
    pub q: F,   // DEEP quotient value Qℓ(x)
}

pub struct FriLayerCommitment {
    pub n: usize,
    pub m: usize,
    pub root: F,
}

pub struct FriTranscript {
    pub schedule: Vec<usize>,
    pub layers: Vec<FriLayerCommitment>,
}

pub struct FriProverParams {
    pub schedule: Vec<usize>,
    pub seed_z: u64,
}

pub struct FriProverState {
    pub f_layers: Vec<Vec<F>>,
    pub s_layers: Vec<Vec<F>>,
    pub q_layers: Vec<Vec<Fp3>>,   // NEW
    pub transcript: FriTranscript,
    pub omega_layers: Vec<F>,
    pub z_layers: Vec<F>,
}

fn pick_arity_for_layer(n: usize, requested_m: usize) -> usize {
    if requested_m >= 128 && n % 128 == 0 { return 128; }
    if requested_m >= 64  && n % 64  == 0 { return 64; }
    if requested_m >= 32  && n % 32  == 0 { return 32; }
    if requested_m >= 16  && n % 16  == 0 { return 16; }
    if requested_m >= 8   && n % 8   == 0 { return 8; }
    if requested_m >= 4   && n % 4   == 0 { return 4; }
    if n % 2 == 0 { return 2; }
    1
}

pub fn deep_fri_prove(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &DeepFriParams,
) -> DeepFriProof {
    // ------------------------
    // Build prover state + transcript
    // ------------------------

    let prover_params = FriProverParams {
        schedule: params.schedule.clone(),
        seed_z: params.seed_z,
    };

    // ✅ FRI internally derives z_fp3 via Fiat–Shamir
    let st = fri_build_transcript(
        f0,
        domain0,
        &prover_params,
    );

    // ------------------------
    // Fiat–Shamir seed for queries
    // ------------------------

    let roots_seed = fs_seed_from_roots(
        &st.transcript
            .layers
            .iter()
            .map(|l| l.root)
            .collect::<Vec<_>>(),
    );

    // ------------------------
    // Generate query openings + Merkle proofs
    // ------------------------

    let (query_refs, roots, layer_proofs) =
        fri_prove_queries(&st, params.r, roots_seed);

    // ------------------------
    // Materialize query payloads
    // ------------------------

    let mut queries = Vec::with_capacity(params.r);

    for q in query_refs {
        let mut payloads = Vec::with_capacity(st.transcript.schedule.len());

        for (ell, rref) in q.per_layer_refs.iter().enumerate() {
            // x_i must match the prover's q_i construction
            let omega = st.omega_layers[ell];
            let x_i = omega.pow([rref.i as u64]);

            // ✅ Extract full Fp³ quotient
            let q_fp3 = st.q_layers[ell][rref.i];

            payloads.push(LayerOpenPayload {
                f_i: st.f_layers[ell][rref.i],
                f_0: st.f_layers[ell][0],
                s_i: st.s_layers[ell][rref.i],

                // ✅ CHANGED: bind all Fp³ coordinates
                q_a0: q_fp3.a0,
                q_a1: q_fp3.a1,
                q_a2: q_fp3.a2,

                x_i,
                f_parent_b: st.f_layers[ell + 1][rref.parent_index],
                s_parent_b: st.s_layers[ell + 1][rref.parent_index],
            });
        }

        queries.push(FriQueryPayload {
            per_layer_refs: q.per_layer_refs,
            per_layer_payloads: payloads,
            final_index: q.final_index,
            final_pair: q.final_pair,
        });
    }

    // ------------------------
    // Return proof
    // ------------------------

    DeepFriProof {
        roots,
        layer_proofs,
        queries,
        n0: domain0.size,
        omega0: domain0.omega,
    }
}

pub fn deep_fri_proof_size_bytes(proof: &DeepFriProof) -> usize {
    const FIELD_BYTES: usize = 8;   // Goldilocks = 64-bit field
    const INDEX_BYTES: usize = 8;   // fixed-width index serialization

    let mut bytes = 0usize;

    // ----------------------------------------
    // Merkle roots
    // ----------------------------------------
    bytes += proof.roots.len() * FIELD_BYTES;

    // ----------------------------------------
    // Query payloads
    // ----------------------------------------
    // Each payload contains:
    // f_i, f_0, s_i,
    // q_a0, q_a1, q_a2,
    // x_i,
    // f_parent_b, s_parent_b
    // = 9 field elements
    for q in &proof.queries {
        bytes += q.per_layer_payloads.len() * 9 * FIELD_BYTES;

        // final_pair (2 field elements)
        bytes += 2 * FIELD_BYTES;
    }

    // ----------------------------------------
    // Merkle openings
    // ----------------------------------------
    for layer in &proof.layer_proofs.layers {
        for opening in &layer.openings {

            // Leaf field element
            bytes += FIELD_BYTES;

            // Opening index
            bytes += INDEX_BYTES;

            // All siblings at every level
            for level in &opening.path {
                bytes += level.len() * FIELD_BYTES;
            }
        }
    }

    bytes
}



fn bind_statement_to_transcript(
    tr: &mut Transcript,
    schedule: &[usize],
    n0: usize,
    seed_z: u64,
) {
    // Domain separation for statement binding
    tr.absorb_bytes(b"DEEP-FRI-STATEMENT");

    // Initial domain size
    tr.absorb_field(F::from(n0 as u64));

    // Folding schedule
    tr.absorb_field(F::from(schedule.len() as u64));
    for &m in schedule {
        tr.absorb_field(F::from(m as u64));
    }

    // Seed used to derive z_ℓ
    tr.absorb_field(F::from(seed_z));
}

fn compute_q_layer(
    f_l: &[F],
    z_l: F,
    omega: F,
) -> Vec<F> {
    let n = f_l.len();
    let mut q = Vec::with_capacity(n);

    let mut x = F::one();
    for i in 0..n {
        let denom = x - z_l;
        // denom ≠ 0 because z_l ∉ Hℓ (already ensured)
        q.push((f_l[i] - f_l[0]) * denom.inverse().unwrap());
        x *= omega;
    }

    q
}

pub fn fri_fold_layer(
    evals: &[F],
    z_l: F,
    folding_factor: usize,
) -> Vec<F> {
    let domain_size = evals.len();
    let domain = GeneralEvaluationDomain::<F>::new(domain_size)
        .expect("Domain size must be a power of two.");
    let domain_generator = domain.group_gen();

    fri_fold_layer_impl(evals, z_l, domain_generator, folding_factor)
}

// -----------------------------------------------------------------------------
// ✅ Transcript + prover logic
// -----------------------------------------------------------------------------

pub fn fri_build_transcript(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &FriProverParams,
) -> FriProverState {
    let schedule = params.schedule.clone();
    let l = schedule.len();

    let mut f_layers = Vec::with_capacity(l + 1);
    let mut s_layers = Vec::with_capacity(l + 1);
    let mut q_layers = Vec::with_capacity(l);
    let mut z_layers_fp3 = Vec::with_capacity(l);
    let mut omega_layers = Vec::with_capacity(l);

    let mut cur_f = f0;
    let mut cur_size = domain0.size;
    f_layers.push(cur_f.clone());

    let mut tr = Transcript::new(b"FRI/FS", transcript_params());

    bind_statement_to_transcript(
        &mut tr,
        &schedule,
        domain0.size,
        params.seed_z,
    );

    // ------------------------------------------------------------
    // ✅ SINGLE DEEP CHALLENGE (Fiat–Shamir, prover == verifier)
    // ------------------------------------------------------------

    let z_fp3 = Fp3 {
        a0: tr.challenge(b"z_fp3/a0"),
        a1: tr.challenge(b"z_fp3/a1"),
        a2: tr.challenge(b"z_fp3/a2"),
    };

    // ------------------------------------------------------------
    // Build FRI layers
    // ------------------------------------------------------------

    for (ell, &m) in schedule.iter().enumerate() {
        // ✅ Same z_fp3 reused for all layers
        z_layers_fp3.push(z_fp3);

        let dom = Domain::<F>::new(cur_size).unwrap();
        let omega = dom.group_gen;
        omega_layers.push(omega);

        // ✅ DEEP quotient in Fp³
        let q = compute_q_layer_fp3(&cur_f, z_fp3, omega);
        q_layers.push(q);

        // ✅ Standard FRI folding using z.a0
        cur_f = fri_fold_layer(&cur_f, z_fp3.a0, m);
        cur_size /= m;
        f_layers.push(cur_f.clone());
    }

    // ------------------------------------------------------------
    // s-layers
    // ------------------------------------------------------------

    for ell in 0..l {
        s_layers.push(compute_s_layer(
            &f_layers[ell],
            z_fp3.a0,
            schedule[ell],
        ));
    }
    s_layers.push(vec![F::zero(); f_layers[l].len()]);

    // ------------------------------------------------------------
    // Merkle commitments
    // ------------------------------------------------------------

    let roots_seed = tr.challenge(ds::FRI_SEED);

    let mut trace_hash = [0u8; 32];
    roots_seed
        .serialize_uncompressed(&mut trace_hash[..])
        .unwrap();

    let mut layers = Vec::with_capacity(l + 1);
    for ell in 0..l {
        let n = f_layers[ell].len();
        let m_ell = schedule[ell];
        let arity = pick_arity_for_layer(n, m_ell).max(2);
        let depth = merkle_depth(n, arity);

        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
        let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

        for i in 0..n {
            let q = q_layers[ell][i];
            tree.push_leaf(&[
                f_layers[ell][i],
                s_layers[ell][i],
                q.a0,
                q.a1,
                q.a2,
            ]);
        }

        let root = tree.finalize();
        layers.push(FriLayerCommitment { n, m: m_ell, root });

        eprintln!("[PROVER] z_fp3 = {:?}", z_fp3);
    }

    FriProverState {
        f_layers,
        s_layers,
        q_layers,
        transcript: FriTranscript { schedule, layers },
        omega_layers,
        z_layers: vec![z_fp3.a0; l],
    }
}
#[derive(Clone)]
pub struct LayerQueryRef {
    pub i: usize,
    pub child_pos: usize,
    pub parent_index: usize,
    pub parent_pos: usize,
}

#[derive(Clone)]
pub struct FriQueryOpenings {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub final_index: usize,
    pub final_pair: (F, F),
}

#[derive(Clone)]
pub struct LayerOpenPayload {
    pub f_i: F,
    pub f_0: F,
    pub s_i: F,

    // ✅ Full Fp³ quotient
    pub q_a0: F,
    pub q_a1: F,
    pub q_a2: F,

    pub x_i: F,
    pub f_parent_b: F,
    pub s_parent_b: F,
}

#[derive(Clone)]
pub struct FriQueryPayload {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub per_layer_payloads: Vec<LayerOpenPayload>,
    pub final_index: usize,
    pub final_pair: (F, F),
}

#[derive(Clone)]
pub struct LayerProof {
    pub openings: Vec<MerkleOpening>, // one per query
}

pub struct FriLayerProofs {
    pub layers: Vec<LayerProof>,
}

pub fn fri_prove_queries(
    st: &FriProverState,
    r: usize,
    roots_seed: F,
) -> (Vec<FriQueryOpenings>, Vec<F>, FriLayerProofs) {
    let L = st.transcript.schedule.len();
    let mut all_refs = Vec::with_capacity(r);

    // ------------------------
    // Query index selection
    // ------------------------

    for q in 0..r {
        let mut per_layer_refs = Vec::with_capacity(L);

        // Sample initial index i_0
        let mut i = {
            let layer0 = &st.transcript.layers[0];
            let n_pow2 = layer0.n.next_power_of_two();
            let seed = index_seed(roots_seed, 0, q);
            index_from_seed(seed, n_pow2) % layer0.n
        };

        // Walk down the FRI layers (STRIDED)
        for ell in 0..L {
            let n = st.transcript.layers[ell].n;
            let m = st.transcript.schedule[ell];
            let n_next = n / m;

            per_layer_refs.push(LayerQueryRef {
                i,
                child_pos: i % m,          // informational only
                parent_index: i % n_next,  // ✅ STRIDED parent
                parent_pos: 0,
            });

            // ✅ Chain index correctly for next layer
            i = i % n_next;
        }

        // ✅ FINAL CONSTANCY: compare f_L[i] with f_L[0]
        let final_i = i;

        all_refs.push(FriQueryOpenings {
            per_layer_refs,
            final_index: final_i,
            final_pair: (
                st.f_layers[L][final_i], // f_L[i]
                st.f_layers[L][0],       // f_L[0]
            ),
        });
    }

    // ------------------------
    // Merkle openings
    // ------------------------

    let mut trace_hash = [0u8; 32];
    roots_seed
        .serialize_uncompressed(&mut trace_hash[..])
        .unwrap();

    let mut layer_proofs = Vec::with_capacity(L);

    for ell in 0..L {
        let layer = &st.transcript.layers[ell];

        let arity = pick_arity_for_layer(layer.n, layer.m).max(2);
        let depth = merkle_depth(layer.n, arity);

        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
        let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

        // ✅ Commit prover state exactly
        for i in 0..layer.n {
            let q = st.q_layers[ell][i];
            tree.push_leaf(&[
                st.f_layers[ell][i],
                st.s_layers[ell][i],
                q.a0,
                q.a1,
                q.a2,
            ]);
        }

        tree.finalize();

        let mut openings = Vec::with_capacity(r);
        for q in 0..r {
            let idx = all_refs[q].per_layer_refs[ell].i;
            openings.push(tree.open(idx));
        }

        layer_proofs.push(LayerProof { openings });
    }

    let roots = st.transcript.layers.iter().map(|l| l.root).collect();

    (all_refs, roots, FriLayerProofs { layers: layer_proofs })
}

#[derive(Clone)]
pub struct DeepFriParams {
    pub schedule: Vec<usize>,
    pub r: usize,
    pub seed_z: u64,
}

pub struct DeepFriProof {
    pub roots: Vec<F>,
    pub layer_proofs: FriLayerProofs,
    pub queries: Vec<FriQueryPayload>,
    pub n0: usize,
    pub omega0: F,
}

pub fn deep_fri_verify(params: &DeepFriParams, proof: &DeepFriProof) -> bool {
    let L = params.schedule.len();
    let sizes = layer_sizes_from_schedule(proof.n0, &params.schedule);

    // ----------------------------------------
    // Reconstruct Fiat–Shamir transcript
    // ----------------------------------------

    let mut tr = Transcript::new(b"FRI/FS", transcript_params());

    bind_statement_to_transcript(
        &mut tr,
        &params.schedule,
        proof.n0,
        params.seed_z,
    );

    // ✅ Reconstruct the SINGLE DEEP challenge z_fp3
    let z_fp3 = Fp3 {
        a0: tr.challenge(b"z_fp3/a0"),
        a1: tr.challenge(b"z_fp3/a1"),
        a2: tr.challenge(b"z_fp3/a2"),
    };

    let z_layers_fp3 = vec![z_fp3; L];

    let roots_seed = tr.challenge(ds::FRI_SEED);

    let mut trace_hash = [0u8; 32];
    roots_seed
        .serialize_uncompressed(&mut trace_hash[..])
        .unwrap();

    eprintln!("[VERIFY] z_fp3 = {:?}", z_fp3);

    // ----------------------------------------
    // Query verification
    // ----------------------------------------

    for q in 0..params.r {
        let qp = &proof.queries[q];

        for ell in 0..L {
            let opening = &proof.layer_proofs.layers[ell].openings[q];

            let arity = pick_arity_for_layer(sizes[ell], params.schedule[ell]).max(2);
            let depth = merkle_depth(sizes[ell], arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);

            // ------------------------
            // Merkle verification
            // ------------------------

            if !MerkleTreeChannel::verify_opening(
                &cfg,
                proof.roots[ell],
                opening,
                &trace_hash,
            ) {
                eprintln!(
                    "[FAIL][MERKLE] q={} ell={} opening_index={}",
                    q, ell, opening.index
                );
                return false;
            }

            let rref = &qp.per_layer_refs[ell];
            let pay = &qp.per_layer_payloads[ell];

            // ------------------------
            // Merkle index binding
            // ------------------------

            if opening.index != rref.i {
                eprintln!(
                    "[FAIL][INDEX BINDING] q={} ell={} opening.index={} rref.i={}",
                    q, ell, opening.index, rref.i
                );
                return false;
            }

            // ------------------------
            // ✅ DEEP quotient check (Fp³, SINGLE z)
            // ------------------------

            // Reconstruct full Fp³ quotient
            let q_fp3 = Fp3 {
                a0: pay.q_a0,
                a1: pay.q_a1,
                a2: pay.q_a2,
            };

            let num = Fp3::from_base(pay.f_i - pay.f_0);
            let denom = Fp3::from_base(pay.x_i) - z_layers_fp3[ell];

            if q_fp3 * denom != num {
                eprintln!(
                    "[FAIL][DEEP-FP3] q={} ell={}\n  f_i={:?}\n  f_0={:?}\n  q_fp3={:?}\n  x_i={:?}\n  z_fp3={:?}",
                    q,
                    ell,
                    pay.f_i,
                    pay.f_0,
                    q_fp3,
                    pay.x_i,
                    z_fp3,
                );
                return false;
            }

            // ------------------------
            // Fold consistency (STRIDED FRI)
            // ------------------------

            let n = sizes[ell];
            let m = params.schedule[ell];
            let n_next = n / m;

            let b = rref.i % n_next;

            let s_child = pay.s_i;
            let f_parent_b = pay.f_parent_b;

            if s_child != f_parent_b {
                eprintln!(
                    "[FAIL][FOLD] q={} ell={}\n  i={}\n  m={}\n  n_next={}\n  b={}\n  s_child={:?}\n  f_parent_b={:?}",
                    q,
                    ell,
                    rref.i,
                    m,
                    n_next,
                    b,
                    s_child,
                    f_parent_b,
                );
                return false;
            }
        }

        // ------------------------
        // Final-layer constancy
        // ------------------------

        if qp.final_pair.0 != qp.final_pair.1 {
            eprintln!(
                "[FAIL][FINAL CONSTANCY] q={} f={:?} s={:?}",
                q,
                qp.final_pair.0,
                qp.final_pair.1
            );
            return false;
        }
    }

    eprintln!("[VERIFY] SUCCESS");
    true
}

fn fri_fold_layer_impl(
    evals: &[F],
    z_l: F,
    omega: F,
    folding_factor: usize,
) -> Vec<F> {
    let n = evals.len();
    assert!(n % folding_factor == 0);

    let n_next = n / folding_factor;
    let mut out = vec![F::zero(); n_next];

    let z_pows = build_z_pows(z_l, folding_factor);

    if enable_parallel(n_next) {
        #[cfg(feature = "parallel")]
        {
            out.par_iter_mut().enumerate().for_each(|(b, out_b)| {
                let mut acc = F::zero();
                for j in 0..folding_factor {
                    acc += evals[b + j * n_next] * z_pows[j];
                }
                *out_b = acc;
            });
            return out;
        }
    }

    for b in 0..n_next {
        let mut acc = F::zero();
        for j in 0..folding_factor {
            acc += evals[b + j * n_next] * z_pows[j];
        }
        out[b] = acc;
    }

    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Field, FftField, One, Zero};
    use ark_goldilocks::Goldilocks;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use rand::Rng;
    use std::collections::HashSet;

    // FIX: Moved these imports inside the test module where they are used.
    use ark_ff::UniformRand;
    use ark_poly::polynomial::univariate::DensePolynomial;
    use rand::seq::SliceRandom;

    type TestField = Goldilocks;

    fn random_polynomial<F: Field>(degree: usize, rng: &mut impl Rng) -> Vec<F> {
        (0..=degree).map(|_| F::rand(rng)).collect()
    }
    
    fn perform_fold<F: Field + FftField>(
        evals: &[F],
        domain: GeneralEvaluationDomain<F>,
        alpha: F,
        folding_factor: usize,
    ) -> (Vec<F>, GeneralEvaluationDomain<F>) {
        assert!(evals.len() % folding_factor == 0);

        let n = evals.len();
        let next_n = n / folding_factor;

        let next_domain = GeneralEvaluationDomain::<F>::new(next_n)
            .expect("valid folded domain");

        let folding_domain = GeneralEvaluationDomain::<F>::new(folding_factor)
            .expect("valid folding domain");

        let generator = domain.group_gen();

        let folded = (0..next_n)
            .map(|i| {
                let coset_values: Vec<F> = (0..folding_factor)
                    .map(|j| evals[i + j * next_n])
                    .collect();

                let coset_generator = generator.pow([i as u64]);
                fold_one_coset(&coset_values, alpha, coset_generator, &folding_domain)
            })
            .collect();

        (folded, next_domain)
    }


    fn fold_one_coset<F: Field + FftField>(
        coset_values: &[F],
        alpha: F,
        coset_generator: F,
        folding_domain: &GeneralEvaluationDomain<F>,
    ) -> F {
        let p_coeffs = folding_domain.ifft(coset_values);
        let poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let evaluation_point = alpha * coset_generator.inverse().unwrap();
        poly.evaluate(&evaluation_point)
    }

    #[test]
    fn test_fri_local_consistency_check_soundness() {
        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        const NUM_TRIALS: usize = 1000000;

        let mut rng = rand::thread_rng();
        let mut detections = 0;

        let z_l = TestField::from(5u64);
        let f: Vec<TestField> = (0..DOMAIN_SIZE).map(|_| TestField::rand(&mut rng)).collect();
        let f_next_claimed: Vec<TestField> = vec![TestField::zero(); DOMAIN_SIZE / FOLDING_FACTOR];

        let domain = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();
        let generator = domain.group_gen();
        let folding_domain = GeneralEvaluationDomain::<TestField>::new(FOLDING_FACTOR).unwrap();

        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..f_next_claimed.len());

            let coset_values: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f[query_index + j * (DOMAIN_SIZE / FOLDING_FACTOR)])
                .collect();

            let coset_generator = generator.pow([query_index as u64]);
            let s_reconstructed = fold_one_coset(&coset_values, z_l, coset_generator, &folding_domain);

            let s_claimed = f_next_claimed[query_index];

            if s_reconstructed != s_claimed {
                detections += 1;
            }
        }
        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        
        println!("[Consistency Check] Detections: {}/{}, Measured Rate: {:.4}", detections, NUM_TRIALS, measured_rate);
        
        assert!((measured_rate - 1.0).abs() < 0.01, "Detection rate should be close to 100%");
    }

    #[test]
    fn test_fri_distance_amplification() {
        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        const NUM_TRIALS: usize = 100_000;
        const INITIAL_CORRUPTION_FRACTION: f64 = 0.05;

        let mut rng = rand::thread_rng();
        let z_l = TestField::from(5u64);

        let large_domain = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();

        let degree_bound = DOMAIN_SIZE / FOLDING_FACTOR;
        let p_coeffs = random_polynomial(degree_bound - 2, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let p_evals = large_domain.fft(p_poly.coeffs());

        // Corrupt the evaluations
        let mut f_evals = p_evals.clone();
        let num_corruptions = (DOMAIN_SIZE as f64 * INITIAL_CORRUPTION_FRACTION) as usize;
        let mut corrupted_indices = HashSet::new();

        while corrupted_indices.len() < num_corruptions {
            corrupted_indices.insert(rng.gen_range(0..DOMAIN_SIZE));
        }

        for &idx in &corrupted_indices {
            f_evals[idx] = TestField::rand(&mut rng);
        }

        // Fold honest and corrupted codewords using production logic
        let folded_honest = fri_fold_layer(&p_evals, z_l, FOLDING_FACTOR);
        let folded_corrupted = fri_fold_layer(&f_evals, z_l, FOLDING_FACTOR);

        let mut detections = 0;

        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..folded_honest.len());

            if folded_honest[query_index] != folded_corrupted[query_index] {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        let theoretical_rate =
            (FOLDING_FACTOR as f64 * INITIAL_CORRUPTION_FRACTION).min(1.0);

        println!("\n[Distance Amplification Test (Single Layer)]");
        println!(
            "  - Initial Corruption: {:.2}% ({} points)",
            INITIAL_CORRUPTION_FRACTION * 100.0,
            num_corruptions
        );
        println!("  - Detections: {}/{}", detections, NUM_TRIALS);
        println!("  - Measured Detection Rate: {:.4}", measured_rate);
        println!("  - Theoretical Detection Rate: {:.4}", theoretical_rate);

        let tolerance = 0.05;
        assert!(
            (measured_rate - theoretical_rate).abs() < tolerance,
            "Measured detection rate should be close to the theoretical rate."
        );
    }

    #[test]
    #[ignore]
    fn test_full_fri_protocol_soundness() {
        const FOLDING_SCHEDULE: [usize; 3] = [4, 4, 4];
        const INITIAL_DOMAIN_SIZE: usize = 4096;
        const NUM_TRIALS: usize = 1000000; 

        let mut rng = rand::thread_rng();
        let mut detections = 0;

        for _ in 0..NUM_TRIALS {
            let alpha = TestField::rand(&mut rng);

            let mut domains = Vec::new();
            let mut current_size = INITIAL_DOMAIN_SIZE;
            domains.push(GeneralEvaluationDomain::<TestField>::new(current_size).unwrap());
            for &folding_factor in &FOLDING_SCHEDULE {
                current_size /= folding_factor;
                domains.push(GeneralEvaluationDomain::<TestField>::new(current_size).unwrap());
            }

            let mut fraudulent_layers: Vec<Vec<TestField>> = Vec::new();
            let f0: Vec<TestField> = (0..INITIAL_DOMAIN_SIZE).map(|_| TestField::rand(&mut rng)).collect();
            fraudulent_layers.push(f0);

            let mut current_layer_evals = fraudulent_layers[0].clone();
            for &folding_factor in &FOLDING_SCHEDULE {
                let next_layer = fri_fold_layer(&current_layer_evals, alpha, folding_factor);
                current_layer_evals = next_layer;
                fraudulent_layers.push(current_layer_evals.clone());
            }

            let mut trial_detected = false;
            let mut query_index = rng.gen_range(0..domains[1].size());

            for l in 0..FOLDING_SCHEDULE.len() {
                let folding_factor = FOLDING_SCHEDULE[l];
                let current_domain = &domains[l];
                let next_domain = &domains[l+1];
                
                let coset_generator = current_domain.group_gen().pow([query_index as u64]);
                let folding_domain = GeneralEvaluationDomain::<TestField>::new(folding_factor).unwrap();

                let coset_values: Vec<TestField> = (0..folding_factor)
                    .map(|j| fraudulent_layers[l][query_index + j * next_domain.size()])
                    .collect();

                let s_reconstructed = fold_one_coset(&coset_values, alpha, coset_generator, &folding_domain);
                let s_claimed = fraudulent_layers[l+1][query_index];

                if s_reconstructed != s_claimed {
                    trial_detected = true;
                    break;
                }
                
                if l + 1 < FOLDING_SCHEDULE.len() {
                    query_index %= domains[l+2].size();
                }
            }

            if !trial_detected {
                let last_layer = fraudulent_layers.last().unwrap();
                let first_element = last_layer[0];
                for &element in last_layer.iter().skip(1) {
                    if element != first_element {
                        trial_detected = true;
                        break;
                    }
                }
            }

            if trial_detected {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;

        println!("\n[Full Protocol Soundness Test (ε_eff)]");
        println!("  - Protocol Schedule: {:?}", FOLDING_SCHEDULE);
        println!("  - Initial Domain Size: {}", INITIAL_DOMAIN_SIZE);
        println!("  - Detections: {}/{}", detections, NUM_TRIALS);
        println!("  - Measured Effective Detection Rate (ε_eff): {:.4}", measured_rate);

        assert!(measured_rate > 0.90, "Effective detection rate should be very high");
    }

    #[test]
    fn test_intermediate_layer_fraud_soundness() {
        const INITIAL_DOMAIN_SIZE: usize = 4096;
        const FOLDING_SCHEDULE: [usize; 3] = [4, 4, 4];
        const NUM_TRIALS: usize = 20000;
        const FRAUD_LAYER_INDEX: usize = 1;

        let mut rng = rand::thread_rng();
        let mut detections = 0;


        // --- Challenges ---
        let alphas: Vec<TestField> = (0..FOLDING_SCHEDULE.len())
            .map(|_| TestField::rand(&mut rng))
            .collect();

        // --- Honest proof ---
        let final_layer_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE.iter().product::<usize>();
        let degree_bound = final_layer_size - 1;

        let p_coeffs = random_polynomial(degree_bound, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);

        let domain0 =
            GeneralEvaluationDomain::<TestField>::new(INITIAL_DOMAIN_SIZE).unwrap();
        let honest_f0 = domain0.fft(p_poly.coeffs());

        let mut honest_layers = vec![honest_f0];
        let mut current = honest_layers[0].clone();

        for (l, &factor) in FOLDING_SCHEDULE.iter().enumerate() {
            let next = fri_fold_layer(&current, alphas[l], factor);
            honest_layers.push(next.clone());
            current = next;
        }

        // --- Create ONE fraudulent prover (fixed fraud location) ---
        let mut prover_layers = honest_layers.clone();

        let fraud_layer_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE[0..FRAUD_LAYER_INDEX].iter().product::<usize>();

        let fraud_index = rng.gen_range(0..fraud_layer_size);
        let honest = prover_layers[FRAUD_LAYER_INDEX][fraud_index];
        let mut corrupted = TestField::rand(&mut rng);
        while corrupted == honest {
            corrupted = TestField::rand(&mut rng);
        }
        prover_layers[FRAUD_LAYER_INDEX][fraud_index] = corrupted;

        // --- Verifier parameters ---
        let l = FRAUD_LAYER_INDEX - 1;
        let folding_factor = FOLDING_SCHEDULE[l];
        let current_domain_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE[0..l].iter().product::<usize>();
        let next_domain_size = current_domain_size / folding_factor;

        let current_domain =
            GeneralEvaluationDomain::<TestField>::new(current_domain_size).unwrap();
        

        // --- Monte Carlo verifier simulation ---
        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..next_domain_size);

            let x = current_domain.element(query_index);

            let coset_values: Vec<TestField> = (0..folding_factor)
                .map(|j| prover_layers[l][query_index + j * next_domain_size])
                .collect();

            let mut lhs = TestField::zero();
            let mut alpha_pow = TestField::one();

            for j in 0..folding_factor {
                lhs += coset_values[j] * alpha_pow;
                alpha_pow *= alphas[l];
            }

            let reconstructed = lhs;

            let claimed = prover_layers[l + 1][query_index];

            if reconstructed != claimed {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        let theoretical_rate = 1.0 / fraud_layer_size as f64;

        println!("\n[Intermediate Layer Fraud Soundness]");
        println!("  Fraud layer size: {}", fraud_layer_size);
        println!("  Fraud index: {}", fraud_index);
        println!("  Detections: {}/{}", detections, NUM_TRIALS);
        println!("  Measured detection rate: {:.6}", measured_rate);
        println!("  Theoretical rate: {:.6}", theoretical_rate);

        let tolerance = theoretical_rate * 5.0;
        assert!(
            (measured_rate - theoretical_rate).abs() < tolerance,
            "Measured detection rate deviates from theory"
        );
    }

    #[test]
    #[ignore]
    fn test_fri_effective_detection_rate() {
        println!("\n--- Running Rust Test: Effective Detection Rate (arkworks) ---");

        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        let degree = 31;

        let mut rng = rand::thread_rng();

        // --- Low-degree polynomial ---
        let p_coeffs = random_polynomial(degree, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let domain0 = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();

        let f0_good = domain0.fft(p_poly.coeffs());
        let mut f0_corrupt = f0_good.clone();

        // --- Initial corruption ---
        let rho_0 = 0.06;
        let num_corruptions = (DOMAIN_SIZE as f64 * rho_0) as usize;
        let indices: Vec<usize> = (0..DOMAIN_SIZE).collect();

        for &idx in indices.choose_multiple(&mut rng, num_corruptions) {
            let honest = f0_corrupt[idx];
            let mut corrupted = TestField::rand(&mut rng);
            while corrupted == honest {
                corrupted = TestField::rand(&mut rng);
            }
            f0_corrupt[idx] = corrupted;
        }

        // --- Fold twice ---
        let alpha1 = TestField::rand(&mut rng);
        let f1_corrupt = fri_fold_layer(&f0_corrupt, alpha1, FOLDING_FACTOR);

        let alpha2 = TestField::rand(&mut rng);
        let f2_corrupt = fri_fold_layer(&f1_corrupt, alpha2, FOLDING_FACTOR);

        // --- Verifier simulation ---
        let num_trials = 200_000;
        let mut detections = 0;

        let domain1_size = DOMAIN_SIZE / FOLDING_FACTOR;
        let domain2_size = domain1_size / FOLDING_FACTOR;

        for _ in 0..num_trials {
            // Sample final-layer index first (correct FRI distribution)
            let i2 = rng.gen_range(0..domain2_size);
            let k = rng.gen_range(0..FOLDING_FACTOR);
            let i1 = i2 + k * domain2_size;

            // --- Layer 1 consistency check ---
            let x1 = domain0.element(i1);

            let coset0: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f0_corrupt[i1 + j * domain1_size])
                .collect();

            let reconstructed_f1 =
                fold_one_coset(&coset0, alpha1, x1, &domain0);

            if reconstructed_f1 != f1_corrupt[i1] {
                detections += 1;
                continue;
            }

            // --- Layer 2 consistency check ---
            let x2 = x1.square();

            let coset1: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f1_corrupt[i2 + j * domain2_size])
                .collect();

            let reconstructed_f2 =
                fold_one_coset(&coset1, alpha2, x2, &domain0);

            if reconstructed_f2 != f2_corrupt[i2] {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / num_trials as f64;

        // --- Theory ---
        let rho_1 = 1.0 - (1.0 - rho_0).powi(FOLDING_FACTOR as i32);
        let rho_2 = 1.0 - (1.0 - rho_1).powi(FOLDING_FACTOR as i32);

        println!("rho_0 = {:.4}", rho_0);
        println!("rho_1 = {:.4}", rho_1);
        println!("rho_2 = {:.4}", rho_2);
        println!("Measured effective detection rate: {:.4}", measured_rate);
        println!("Theoretical effective detection rate: {:.4}", rho_2);

        let delta = 0.03;
        assert!(
            (measured_rate - rho_2).abs() < delta,
            "Measured rate {:.4} not close to theoretical {:.4}",
            measured_rate,
            rho_2
        );

        println!("✅ Effective detection rate matches theory");
    }

    #[test]
    fn debug_single_fold_distance_amplification() {
        // We now use our custom-defined GoldilocksField as the base field.
        //type F = GoldilocksField;

        // 1. Setup with known parameters
        let log_domain_size = 12; // 4096
        let initial_domain_size = 1 << log_domain_size;
        let folding_factor = 4;
        let initial_corruption_rate = 0.06;

        let mut rng = StdRng::seed_from_u64(0);

        // 2. Create a valid codeword C_0 and a corrupted version C'_0
    
        // a. Create a valid low-degree polynomial
        let degree = (initial_domain_size / folding_factor) - 1;
        let domain = GeneralEvaluationDomain::<F>::new(initial_domain_size)
            .expect("Failed to create domain");
        let poly_p0 = DensePolynomial::<F>::rand(degree, &mut rng);

        // b. Evaluate it to get the "true" codeword C_0
        let codeword_c0_evals = poly_p0.evaluate_over_domain(domain).evals;

        // c. Create a corrupted version C'_0 by modifying a percentage of points
        let mut corrupted_codeword_c_prime_0_evals = codeword_c0_evals.clone();
        let num_corruptions = (initial_domain_size as f64 * initial_corruption_rate).ceil() as usize;
        let mut corrupted_indices = HashSet::new();

        while corrupted_indices.len() < num_corruptions {
            let idx_to_corrupt = usize::rand(&mut rng) % initial_domain_size;
            if corrupted_indices.contains(&idx_to_corrupt) {
                continue;
            }

            let original_value = corrupted_codeword_c_prime_0_evals[idx_to_corrupt];
            let mut new_value = F::rand(&mut rng);
            // Ensure the new value is actually different
            while new_value == original_value {
                new_value = F::rand(&mut rng);
            }
            corrupted_codeword_c_prime_0_evals[idx_to_corrupt] = new_value;
            corrupted_indices.insert(idx_to_corrupt);
        }

        // 3. Simulate a single fold on both the true and corrupted codewords
        let alpha = F::rand(&mut rng); // The verifier's challenge

        let (folded_corrupted_evals, new_domain) = perform_fold(
            &corrupted_codeword_c_prime_0_evals,
            domain,
            alpha,
            folding_factor,
        );
    
        let (folded_true_evals, _) = perform_fold(
            &codeword_c0_evals,
            domain,
            alpha,
            folding_factor,
        );

        // 4. Manually and explicitly calculate the distance of the result
        // Count how many elements in our folded corrupted codeword differ from the "true" folded one.
        let differing_points = folded_corrupted_evals
            .iter()
            .zip(folded_true_evals.iter())
            .filter(|(a, b)| a != b)
            .count();

        let measured_rho_1 = differing_points as f64 / new_domain.size() as f64;

        // 5. Assert against the precise theoretical value
        let theoretical_rho_1 = 1.0_f64 - (1.0_f64 - initial_corruption_rate).powf(folding_factor as f64);
    
        println!("--- Debugging Single Fold (Goldilocks Field) ---");
        println!("Initial rho_0:       {}", initial_corruption_rate);
        println!("Measured rho_1:      {}", measured_rho_1);
        println!("Theoretical rho_1:   {}", theoretical_rho_1);

        // Use a tight tolerance for this direct check. A small deviation is expected
        // due to statistical effects of random corruption, but it should be very small.
        let tolerance = 0.01; 
        assert!(
            (measured_rho_1 - theoretical_rho_1).abs() < tolerance, 
            "Single fold amplification measured rate {} is not close to precise theoretical rate {}",
            measured_rho_1,
            theoretical_rho_1
        );
    }
}