
#[allow(dead_code)]
use ark_ff::{PrimeField, Zero};
use ark_pallas::Fr as F;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use poseidon::{params::generate_params_t17_x5, permute, PoseidonParams, T};

use sha3::{Digest, Sha3_256};

/// ============================================================
/// Helpers: field <-> bytes
/// ============================================================

#[inline]
fn field_to_bytes(x: &F) -> Vec<u8> {
    let mut out = Vec::new();
    x.serialize_compressed(&mut out)
        .expect("canonical serialize");
    out
}

#[inline]
#[allow(dead_code)]
fn bytes_to_field(bytes: &[u8]) -> F {
    F::from_le_bytes_mod_order(bytes)
}

/// Decode a row that was encoded as a concatenation of
/// `serialize_compressed(field)` values.
fn decode_row(bytes: &[u8]) -> Vec<F> {
    let mut out = Vec::new();
    let mut cursor = bytes;

    while !cursor.is_empty() {
        let mut reader = cursor;
        let x = F::deserialize_compressed(&mut reader).expect("canonical deserialize");
        let consumed = cursor.len() - reader.len();
        cursor = &cursor[consumed..];
        out.push(x);
    }

    out
}

/// ============================================================
/// Dual commitment object
/// ============================================================
///
/// (sha3_commit, poseidon_commit, trace_hash)
///
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualCommitment {
    pub sha_commit: [u8; 32],
    pub poseidon_root: F,
    pub trace_hash: [u8; 32],
}

/// ============================================================
/// Merkle commitment using Poseidon (t = 17, arity = 16)
/// ============================================================

pub struct MerkleCommitment {
    pub arity: usize,
    pub params: PoseidonParams,
}

impl MerkleCommitment {
    /// Construct with standard Poseidon parameters
    pub fn with_default_params() -> Self {
        let seed = b"POSEIDON-T17-X5-SEED";
        let params = generate_params_t17_x5(seed);
        Self { arity: 16, params }
    }

    // ============================================================
    // Encoding
    // ============================================================

    fn encode_trace_rows(trace: &[Vec<F>]) -> Vec<Vec<u8>> {
        trace
            .iter()
            .map(|row| {
                let mut out = Vec::new();
                for x in row {
                    out.extend_from_slice(&field_to_bytes(x));
                }
                out
            })
            .collect()
    }

    fn encode_trace_flat(trace: &[Vec<F>]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in trace {
            for x in row {
                out.extend_from_slice(&field_to_bytes(x));
            }
        }
        out
    }

    // ============================================================
    // SHA3 commitments
    // ============================================================

    fn sha3_trace(trace: &[Vec<F>]) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"TRACE_HASH_V1");
        h.update(&Self::encode_trace_flat(trace));
        h.finalize().into()
    }

    fn sha3_commit(trace: &[Vec<F>], trace_hash: &[u8; 32]) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"TRACE_BYTES_COMMIT_V1");
        h.update(trace_hash);
        h.update(&Self::encode_trace_flat(trace));
        h.finalize().into()
    }

    // ============================================================
    // Poseidon sponge with domain separation
    // ============================================================

    fn poseidon_hash_with_ds(inputs: &[F], params: &PoseidonParams, trace_hash: &[u8; 32]) -> F {
        let mut state = [F::zero(); T];

        // Domain separation: bind trace_hash into capacity
        state[T - 1] = F::from_le_bytes_mod_order(trace_hash);

        for chunk in inputs.chunks(T - 1) {
            for (i, &x) in chunk.iter().enumerate() {
                state[i] += x;
            }
            permute(&mut state, params);
        }

        state[0]
    }

    // ============================================================
    // Poseidon Merkle commitment
    // ============================================================

    pub fn commit(&self, trace: &[Vec<F>]) -> F {
        let trace_hash = Self::sha3_trace(trace);
        self.commit_with_hash(trace, &trace_hash)
    }

    fn commit_with_hash(&self, trace: &[Vec<F>], trace_hash: &[u8; 32]) -> F {
        let leaves_bytes = Self::encode_trace_rows(trace);

        // --- Leaf hashing (CRITICAL: correct packing) ---
        let mut level: Vec<F> = leaves_bytes
            .iter()
            .map(|bytes| {
                let fields = decode_row(bytes);
                Self::poseidon_hash_with_ds(&fields, &self.params, trace_hash)
            })
            .collect();

        // --- Merkle tree ---
        while level.len() > 1 {
            let mut next = Vec::new();
            for chunk in level.chunks(self.arity) {
                next.push(Self::poseidon_hash_with_ds(chunk, &self.params, trace_hash));
            }
            level = next;
        }

        level[0]
    }

    // ============================================================
    // Dual commitment API
    // ============================================================

    pub fn dual_commit(&self, trace: &[Vec<F>]) -> DualCommitment {
        let trace_hash = Self::sha3_trace(trace);
        let sha_commit = Self::sha3_commit(trace, &trace_hash);
        let poseidon_root = self.commit_with_hash(trace, &trace_hash);

        DualCommitment {
            sha_commit,
            poseidon_root,
            trace_hash,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_commit_roundtrip() {
        let mc = MerkleCommitment::with_default_params();

        let trace = vec![
            vec![F::from(1u64), F::from(2u64)],
            vec![F::from(3u64), F::from(4u64)],
        ];

        let r1 = mc.commit(&trace);
        let r2 = mc.commit(&trace);

        assert_eq!(r1, r2);
    }

    #[test]
    fn dual_commit_deterministic() {
        let mc = MerkleCommitment::with_default_params();

        let trace = vec![vec![F::from(42u64)], vec![F::from(7u64)]];

        let c1 = mc.dual_commit(&trace);
        let c2 = mc.dual_commit(&trace);

        assert_eq!(c1, c2);
    }

    #[test]
    fn poseidon_binds_trace_hash() {
        let mc = MerkleCommitment::with_default_params();

        let t1 = vec![vec![F::from(1u64)]];
        let t2 = vec![vec![F::from(2u64)]];

        let c1 = mc.dual_commit(&t1);
        let c2 = mc.dual_commit(&t2);

        assert_ne!(c1.poseidon_root, c2.poseidon_root);
        assert_ne!(c1.trace_hash, c2.trace_hash);
    }
}
