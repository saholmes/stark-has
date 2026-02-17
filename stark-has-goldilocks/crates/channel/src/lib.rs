use ark_goldilocks::Goldilocks as F;
use transcript::Transcript;
use merkle::{
    MerkleChannelCfg,
    MerkleTreeChannel,
    MerkleOpening,
};

/// =========================
/// Transcript-backed channel
/// =========================

pub struct ProverChannel {
    tr: Transcript,
}

pub struct VerifierChannel {
    tr: Transcript,
}

impl ProverChannel {
    pub fn new(tr: Transcript) -> Self {
        Self { tr }
    }

    pub fn transcript_mut(&mut self) -> &mut Transcript {
        &mut self.tr
    }

    pub fn absorb_field(&mut self, label: &[u8], f: &F) {
        self.tr.absorb_bytes(label);
        self.tr.absorb_field(*f);
    }

    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.tr.challenge(label)
    }
}

impl VerifierChannel {
    pub fn new(tr: Transcript) -> Self {
        Self { tr }
    }

    pub fn transcript_mut(&mut self) -> &mut Transcript {
        &mut self.tr
    }

    pub fn absorb_field(&mut self, label: &[u8], f: &F) {
        self.tr.absorb_bytes(label);
        self.tr.absorb_field(*f);
    }

    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.tr.challenge(label)
    }
}

/// =========================
/// Merkle channel (minimal)
/// =========================

pub struct MerkleProver<'a> {
    chan: &'a mut ProverChannel,
    tree: MerkleTreeChannel,
}

pub struct MerkleVerifier<'a> {
    chan: &'a mut VerifierChannel,
    cfg: MerkleChannelCfg,
    root: Option<F>,
}

impl<'a> MerkleProver<'a> {
    pub fn new(
        chan: &'a mut ProverChannel,
        cfg: MerkleChannelCfg,
        trace_hash: [u8; 32],
    ) -> Self {
        let tree = MerkleTreeChannel::new(cfg, trace_hash);
        Self { chan, tree }
    }

    /// Commit to a vector of field elements.
    /// Each leaf is (f, s, q) = (value, 0, 0),
    /// matching your FRI base-layer convention.
    pub fn commit(&mut self, values: &[F]) -> F {
        for v in values {
            self.tree
                .push_leaf(&[
                    *v,
                    F::from(0u64),
                    F::from(0u64),
                ]);
        }
        let root = self.tree.finalize();
        self.chan.absorb_field(b"merkle/root", &root);
        root
    }

    pub fn open(&self, index: usize) -> MerkleOpening {
        self.tree.open(index)
    }

    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.chan.challenge(label)
    }
}

impl<'a> MerkleVerifier<'a> {
    pub fn new(
        chan: &'a mut VerifierChannel,
        cfg: MerkleChannelCfg,
    ) -> Self {
        Self {
            chan,
            cfg,
            root: None,
        }
    }

    pub fn receive_root(&mut self, root: &F) {
        self.chan.absorb_field(b"merkle/root", root);
        self.root = Some(*root);
    }

    pub fn verify_opening(
        &self,
        opening: &MerkleOpening,
        trace_hash: &[u8; 32],
    ) -> bool {
        let root = match self.root {
            Some(r) => r,
            None => return false,
        };

        MerkleTreeChannel::verify_opening(
            &self.cfg,
            root,
            opening,
            trace_hash,
        )
    }

    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.chan.challenge(label)
    }
}

/// =========================
/// Tests (sanity check)
/// =========================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn merkle_channel_roundtrip() {
        let params = transcript::default_params();

        let p_tr = Transcript::new(b"CHAN-TEST", params.clone());
        let v_tr = Transcript::new(b"CHAN-TEST", params.clone());

        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        // ✅ Explicit Merkle config:
        // binary tree (arity = 2), fixed tree label
        let cfg = MerkleChannelCfg::new(vec![2, 2, 2, 2], 12345u64);
        let trace_hash = [0u8; 32];

        let mut rng = StdRng::seed_from_u64(42);
        let values: Vec<F> = (0..16).map(|_| F::rand(&mut rng)).collect();

        let mut prover = MerkleProver::new(&mut pchan, cfg.clone(), trace_hash);
        let root = prover.commit(&values);

        let mut verifier = MerkleVerifier::new(&mut vchan, cfg.clone());
        verifier.receive_root(&root);

        // Transcript consistency check
        let alpha_p = prover.challenge(b"alpha");
        let alpha_v = verifier.challenge(b"alpha");
        assert_eq!(alpha_p, alpha_v);

        // Merkle opening check
        let idx = 7usize;
        let opening = prover.open(idx);

        assert!(
            verifier.verify_opening(&opening, &trace_hash),
            "Merkle opening failed"
        );
    }
}