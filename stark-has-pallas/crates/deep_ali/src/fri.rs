
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_macros)]
use ark_pallas::Fr as F;
//use ark_serialize::CanonicalSerialize;

use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain,
    Polynomial,
};

use merkle::{MerkleChannelCfg, MerkleOpening, MerkleTreeChannel};

use transcript::{default_params as transcript_params, Transcript};

mod ds {
    pub const FRI_SEED: &[u8] = b"FRI/seed";
    pub const FRI_INDEX: &[u8] = b"FRI/index";
    pub const FRI_Z_L: &[u8] = b"FRI/z/l";
}

/* ============================================================
   Domain
============================================================ */

#[derive(Clone, Copy, Debug)]
pub struct FriDomain {
    pub omega: F,
    pub size: usize,
}

impl FriDomain {
    pub fn new_radix2(size: usize) -> Self {
        let dom = Domain::<F>::new(size).expect("radix-2 domain exists");
        Self {
            omega: dom.group_gen,
            size,
        }
    }
}

/* ============================================================
   Algebra helpers
============================================================ */

fn compute_q_layer(f_l: &[F], z_l: F, omega: F) -> (Vec<F>, F) {
    let n = f_l.len();

    let domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
    let coeffs = domain.ifft(f_l);
    let poly = DensePolynomial::from_coefficients_vec(coeffs);

    let f_z = poly.evaluate(&z_l);

    let mut q = Vec::with_capacity(n);
    let mut x = F::one();

    for i in 0..n {
        let denom = x - z_l;
        q.push((f_l[i] - f_z) * denom.inverse().unwrap());
        x *= omega;
    }

    (q, f_z)
}

pub fn compute_s_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    let n = f_l.len();
    let n_next = n / m;

    let mut z_pow = F::one();
    let mut z_pows = Vec::with_capacity(m);
    for _ in 0..m {
        z_pows.push(z_pow);
        z_pow *= z_l;
    }

    let mut folded = vec![F::zero(); n_next];
    for b in 0..n_next {
        for j in 0..m {
            folded[b] += f_l[b + j * n_next] * z_pows[j];
        }
    }

    let mut s = vec![F::zero(); n];
    for b in 0..n_next {
        for j in 0..m {
            s[b + j * n_next] = folded[b];
        }
    }

    s
}

pub fn fri_fold_layer(evals: &[F], z_l: F, m: usize) -> Vec<F> {
    let n = evals.len();
    let n_next = n / m;

    let mut z_pow = F::one();
    let mut z_pows = Vec::with_capacity(m);
    for _ in 0..m {
        z_pows.push(z_pow);
        z_pow *= z_l;
    }

    let mut out = vec![F::zero(); n_next];

    for b in 0..n_next {
        let mut acc = F::zero();
        for j in 0..m {
            acc += evals[b + j * n_next] * z_pows[j];
        }
        out[b] = acc;
    }

    out
}

/* ============================================================
   Prover State
============================================================ */

pub struct FriProverState {
    pub f_layers: Vec<Vec<F>>,
    pub s_layers: Vec<Vec<F>>,
    pub q_layers: Vec<Vec<F>>,
    pub fz_layers: Vec<F>,
    pub omega_layers: Vec<F>,
    pub roots: Vec<F>,
}

fn pick_arity_for_layer(n: usize, m: usize) -> usize {
    [128, 64, 32, 16, 8, 4, 2]
        .into_iter()
        .find(|&a| m >= a && n % a == 0)
        .unwrap_or(2)
}

fn merkle_depth(leaves: usize, arity: usize) -> usize {
    let mut depth = 1;
    let mut cur = leaves;
    while cur > arity {
        cur = (cur + arity - 1) / arity;
        depth += 1;
    }
    depth
}

pub fn fri_build_layers(
    f0: Vec<F>,
    domain0: FriDomain,
    schedule: &[usize],
    z_layers: &[F],
) -> FriProverState {
    let L = schedule.len();

    let mut f_layers = vec![f0];
    let mut s_layers = Vec::with_capacity(L + 1);
    let mut q_layers = Vec::with_capacity(L);
    let mut fz_layers = Vec::with_capacity(L);
    let mut omega_layers = Vec::with_capacity(L);

    let mut cur_f = f_layers[0].clone();
    let mut cur_size = domain0.size;

    for (ell, &m) in schedule.iter().enumerate() {
        let z = z_layers[ell];

        let dom = Domain::<F>::new(cur_size).unwrap();
        let omega = dom.group_gen;
        omega_layers.push(omega);

        let (q, f_z) = compute_q_layer(&cur_f, z, omega);
        q_layers.push(q);
        fz_layers.push(f_z);

        s_layers.push(compute_s_layer(&cur_f, z, m));

        cur_f = fri_fold_layer(&cur_f, z, m);
        cur_size /= m;

        f_layers.push(cur_f.clone());
    }

    s_layers.push(vec![F::zero(); f_layers[L].len()]);

    let trace_hash = [0u8; 32];
    let mut roots = Vec::with_capacity(L + 1);

    for ell in 0..=L {
        let n = f_layers[ell].len();
        let m = if ell < L { schedule[ell] } else { 1 };

        let arity = pick_arity_for_layer(n, m);
        let depth = merkle_depth(n, arity);
        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);

        let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

        for i in 0..n {
            tree.push_leaf(
                f_layers[ell][i],
                s_layers[ell][i],
                if ell < L { q_layers[ell][i] } else { F::zero() },
            );
        }

        roots.push(tree.finalize());
    }

    FriProverState {
        f_layers,
        s_layers,
        q_layers,
        fz_layers,
        omega_layers,
        roots,
    }
}

/* ============================================================
   Query Structures
============================================================ */

#[derive(Clone)]
pub struct LayerQueryRef {
    pub i: usize,
    pub parent_index: usize,
}

#[derive(Clone)]
pub struct LayerOpenPayload {
    pub f_i: F,
    pub f_z: F,
    pub s_i: F,
    pub q_i: F,
    pub x_i: F,
    pub f_parent_b: F,
}

#[derive(Clone)]
pub struct FriQueryPayload {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub per_layer_payloads: Vec<LayerOpenPayload>,
    pub final_index: usize,
    pub final_pair: (F, F),
}

use std::collections::BTreeMap;

#[derive(Clone)]
pub struct CompressedLayerProof {
    // Unique sibling nodes
    pub nodes: Vec<F>,

    // For each query: flattened sibling references
    pub paths: Vec<Vec<usize>>,

    // Leaf indices for each query
    pub leaf_indices: Vec<usize>,
}

pub struct FriLayerProofs {
    pub layers: Vec<CompressedLayerProof>,
}

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

/* ============================================================
   Query Generation
============================================================ */

fn tr_hash_fields_tagged(tag: &[u8], fields: &[F]) -> F {
    let mut tr = Transcript::new(b"FRI/FS", transcript_params());
    tr.absorb_bytes(tag);
    for &x in fields {
        tr.absorb_field(x);
    }
    tr.challenge(b"out")
}

fn fri_prove_queries(
    st: &FriProverState,
    r: usize,
    roots_seed: F,
) -> (Vec<FriQueryPayload>, FriLayerProofs) {
    let L = st.fz_layers.len();
    let mut queries = Vec::with_capacity(r);

    // ----------------------------
    // Step 1: derive all query refs
    // ----------------------------
    let mut all_refs: Vec<Vec<LayerQueryRef>> = Vec::with_capacity(r);

    for q in 0..r {
        let seed = tr_hash_fields_tagged(
            ds::FRI_INDEX,
            &[roots_seed, F::from(0u64), F::from(q as u64)],
        );

        let bigint = seed.into_bigint();
        let mut i = bigint.as_ref()[0] as usize % st.f_layers[0].len();

        let mut per_layer_refs = Vec::with_capacity(L);

        for ell in 0..L {
            let n = st.f_layers[ell].len();
            let m = n / st.f_layers[ell + 1].len();
            let n_next = n / m;

            let parent_index = i % n_next;

            per_layer_refs.push(LayerQueryRef { i, parent_index });

            i = parent_index;
        }

        all_refs.push(per_layer_refs);
    }

    // ----------------------------
    // Step 2: build compressed proofs
    // ----------------------------
    let trace_hash = [0u8; 32];
    let mut layer_proofs = Vec::with_capacity(L);

    for ell in 0..L {
        let n = st.f_layers[ell].len();
        let m = n / st.f_layers[ell + 1].len();

        let arity = pick_arity_for_layer(n, m);
        let depth = merkle_depth(n, arity);
        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);

        // Build tree ONCE
        let mut tree = MerkleTreeChannel::new(cfg.clone(), trace_hash);

        for i in 0..n {
            tree.push_leaf(
                st.f_layers[ell][i],
                st.s_layers[ell][i],
                st.q_layers[ell][i],
            );
        }

        tree.finalize();

        // Dedup siblings
        let mut node_map: BTreeMap<F, usize> = BTreeMap::new();
        let mut nodes: Vec<F> = Vec::new();
        let mut paths: Vec<Vec<usize>> = Vec::with_capacity(r);
        let mut leaf_indices: Vec<usize> = Vec::with_capacity(r);

        for q in 0..r {
            let idx = all_refs[q][ell].i;
            leaf_indices.push(idx);

            let opening = tree.open(idx);

            let mut path_indices = Vec::new();

            for sibling_layer in opening.path.iter() {
                for sib in sibling_layer {
                    let entry = node_map.entry(*sib).or_insert_with(|| {
                        let pos = nodes.len();
                        nodes.push(*sib);
                        pos
                    });
                    path_indices.push(*entry);
                }
            }

            paths.push(path_indices);
        }

        layer_proofs.push(CompressedLayerProof {
            nodes,
            paths,
            leaf_indices,
        });
    }

    // ----------------------------
    // Step 3: materialize payloads
    // ----------------------------
    for q in 0..r {
        let mut payloads = Vec::with_capacity(L);

        for ell in 0..L {
            let rref = &all_refs[q][ell];
            let omega = st.omega_layers[ell];
            let x_i = omega.pow([rref.i as u64]);

            payloads.push(LayerOpenPayload {
                f_i: st.f_layers[ell][rref.i],
                f_z: st.fz_layers[ell],
                s_i: st.s_layers[ell][rref.i],
                q_i: st.q_layers[ell][rref.i],
                x_i,
                f_parent_b: st.f_layers[ell + 1][rref.parent_index],
            });
        }

        queries.push(FriQueryPayload {
            per_layer_refs: all_refs[q].clone(),
            per_layer_payloads: payloads,
            final_index: all_refs[q][L - 1].parent_index,
            final_pair: (
                st.f_layers[L][all_refs[q][L - 1].parent_index],
                st.f_layers[L][0],
            ),
        });
    }

    (
        queries,
        FriLayerProofs {
            layers: layer_proofs,
        },
    )
}

/* ============================================================
   Prover
============================================================ */

pub fn deep_fri_prove(f0: Vec<F>, domain0: FriDomain, params: &DeepFriParams) -> DeepFriProof {
    let L = params.schedule.len();

    let mut tr = Transcript::new(b"FRI/FS", transcript_params());

    tr.absorb_bytes(b"DEEP-FRI-STATEMENT");
    tr.absorb_field(F::from(domain0.size as u64));
    tr.absorb_field(F::from(L as u64));
    for &m in &params.schedule {
        tr.absorb_field(F::from(m as u64));
    }
    tr.absorb_field(F::from(params.seed_z));

    let mut z_layers = Vec::with_capacity(L);
    for _ in 0..L {
        z_layers.push(tr.challenge(ds::FRI_Z_L));
    }

    let st = fri_build_layers(f0, domain0, &params.schedule, &z_layers);

    for root in &st.roots {
        tr.absorb_field(*root);
    }

    let roots_seed = tr.challenge(ds::FRI_SEED);

    let (queries, layer_proofs) = fri_prove_queries(&st, params.r, roots_seed);

    DeepFriProof {
        roots: st.roots,
        layer_proofs,
        queries,
        n0: domain0.size,
        omega0: domain0.omega,
    }
}

/* ============================================================
   Verifier
============================================================ */

pub fn deep_fri_verify(params: &DeepFriParams, proof: &DeepFriProof) -> bool {
    let L = params.schedule.len();

    // ---------------------------------------
    // Recompute layer sizes
    // ---------------------------------------
    let mut sizes = Vec::with_capacity(L + 1);
    let mut n = proof.n0;
    sizes.push(n);

    for &m in &params.schedule {
        if n % m != 0 {
            return false;
        }
        n /= m;
        sizes.push(n);
    }

    // ---------------------------------------
    // Rebuild transcript
    // ---------------------------------------
    let mut tr = Transcript::new(b"FRI/FS", transcript_params());

    tr.absorb_bytes(b"DEEP-FRI-STATEMENT");
    tr.absorb_field(F::from(proof.n0 as u64));
    tr.absorb_field(F::from(L as u64));

    for &m in &params.schedule {
        tr.absorb_field(F::from(m as u64));
    }

    tr.absorb_field(F::from(params.seed_z));

    let mut z_layers = Vec::with_capacity(L);
    for _ in 0..L {
        z_layers.push(tr.challenge(ds::FRI_Z_L));
    }

    for root in &proof.roots {
        tr.absorb_field(*root);
    }

    let _roots_seed = tr.challenge(ds::FRI_SEED);

    // ---------------------------------------
    // Merkle salt (must match prover)
    // ---------------------------------------
    let trace_hash = [0u8; 32];

    // ---------------------------------------
    // Verify each query
    // ---------------------------------------
    for q in 0..params.r {
        let qp = &proof.queries[q];

        for ell in 0..L {
            let compressed = &proof.layer_proofs.layers[ell];

            let n_layer = sizes[ell];
            let m = params.schedule[ell];

            let arity = pick_arity_for_layer(n_layer, m);
            let depth = merkle_depth(n_layer, arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);

            // ---------------------------------------
            // Reconstruct Merkle path
            // ---------------------------------------
            let idx = compressed.leaf_indices[q];
            let path_indices = &compressed.paths[q];

            let mut cursor = 0;
            let mut path: Vec<Vec<F>> = Vec::with_capacity(depth);

            for _ in 0..depth {
                let mut siblings = Vec::with_capacity(arity - 1);

                for _ in 0..(arity - 1) {
                    if cursor >= path_indices.len() {
                        return false;
                    }
                    siblings.push(compressed.nodes[path_indices[cursor]]);
                    cursor += 1;
                }

                path.push(siblings);
            }

            // ---------------------------------------
            // Reconstruct leaf exactly as prover did
            // ---------------------------------------
            let pay = &qp.per_layer_payloads[ell];

            let leaf = MerkleTreeChannel::compute_leaf_static(
                &cfg,
                &trace_hash,
                idx,
                pay.f_i,
                pay.s_i,
                pay.q_i,
            );

            let opening = MerkleOpening {
                leaf,
                path,
                index: idx,
            };

            if !MerkleTreeChannel::verify_opening(&cfg, proof.roots[ell], &opening, &trace_hash) {
                return false;
            }

            // ---------------------------------------
            // Check index consistency
            // ---------------------------------------
            let rref = &qp.per_layer_refs[ell];
            if idx != rref.i {
                return false;
            }

            // ---------------------------------------
            // DEEP equation
            // ---------------------------------------
            let lhs = pay.q_i * (pay.x_i - z_layers[ell]);
            let rhs = pay.f_i - pay.f_z;

            if lhs != rhs {
                return false;
            }

            // ---------------------------------------
            // Fold consistency
            // ---------------------------------------
            if pay.s_i != pay.f_parent_b {
                return false;
            }
        }

        // ---------------------------------------
        // Final constant check
        // ---------------------------------------
        if qp.final_pair.0 != qp.final_pair.1 {
            return false;
        }
    }

    true
}

/* ============================================================
   Proof Size
============================================================ */

pub fn deep_fri_proof_size_bytes<Ff: PrimeField>(proof: &DeepFriProof) -> usize {
    let fb = Ff::BigInt::NUM_LIMBS * std::mem::size_of::<u64>();
    let usize_bytes = std::mem::size_of::<usize>();

    let mut bytes = 0;

    // Roots
    bytes += proof.roots.len() * fb;

    // Payload fields (6 per layer per query)
    for q in &proof.queries {
        bytes += q.per_layer_payloads.len() * 6 * fb;
    }

    // Compressed Merkle proofs
    for layer in &proof.layer_proofs.layers {
        // Unique sibling nodes (field elements)
        bytes += layer.nodes.len() * fb;

        // Leaf indices
        bytes += layer.leaf_indices.len() * usize_bytes;

        // Path references
        for path in &layer.paths {
            bytes += path.len() * usize_bytes;
        }
    }

    bytes
}
