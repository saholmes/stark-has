use ark_ff::PrimeField;
use ark_pallas::Fr as F;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha3::{Digest, Sha3_256};

/// =======================
/// Canonical field wrapper
/// =======================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerFr(pub F);

impl From<F> for SerFr {
    fn from(x: F) -> Self {
        SerFr(x)
    }
}

impl From<SerFr> for F {
    fn from(w: SerFr) -> F {
        w.0
    }
}

impl Serialize for SerFr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut buf = Vec::new();
        self.0
            .serialize_compressed(&mut buf)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&buf)
    }
}

impl<'de> Deserialize<'de> for SerFr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let x = F::deserialize_compressed(&bytes[..]).map_err(serde::de::Error::custom)?;
        Ok(SerFr(x))
    }
}

/// =======================
/// Domain separation
/// =======================

#[derive(Clone, Copy, Debug)]
pub struct DsLabel {
    pub arity: usize,
    pub level: u32,
    pub position: u64,
    pub tree_label: u64,
}

impl DsLabel {
    fn to_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&(self.arity as u64).to_le_bytes());
        out[8..16].copy_from_slice(&(self.level as u64).to_le_bytes());
        out[16..24].copy_from_slice(&self.position.to_le_bytes());
        out[24..32].copy_from_slice(&self.tree_label.to_le_bytes());
        out
    }
}

const LEAF_LEVEL_DS: u32 = u32::MAX;

/// =======================
/// Merkle config
/// =======================

#[derive(Clone)]
pub struct MerkleChannelCfg {
    pub layer_arities: Vec<usize>,
    pub tree_label: u64,
}

impl MerkleChannelCfg {
    pub fn new(layer_arities: Vec<usize>, tree_label: u64) -> Self {
        Self {
            layer_arities,
            tree_label,
        }
    }
}

/// =======================
/// Merkle opening
/// =======================

#[derive(Clone, Debug)]
pub struct MerkleOpening {
    pub leaf: F,
    pub path: Vec<Vec<F>>, // sibling groups
    pub index: usize,
}

/// =======================
/// Merkle tree (trace‑bound)
/// =======================

pub struct MerkleTreeChannel {
    cfg: MerkleChannelCfg,
    trace_hash: [u8; 32],
    levels: Vec<Vec<F>>,
}

impl MerkleTreeChannel {
    pub fn new(cfg: MerkleChannelCfg, trace_hash: [u8; 32]) -> Self {
        Self {
            cfg,
            trace_hash,
            levels: Vec::new(),
        }
    }

    /// Canonical compression: DS || trace_hash || children
    fn compress_static(ds: DsLabel, trace_hash: &[u8; 32], children: &[F]) -> F {
        let mut h = Sha3_256::new();

        h.update(ds.to_bytes());
        h.update(trace_hash);

        for c in children {
            let mut buf = Vec::new();
            c.serialize_compressed(&mut buf).unwrap();
            h.update(&buf);
        }

        let out = h.finalize();
        F::from_le_bytes_mod_order(&out)
    }

    fn compress(&self, ds: DsLabel, children: &[F]) -> F {
        Self::compress_static(ds, &self.trace_hash, children)
    }

    /// ✅ Public helper for verifier leaf reconstruction
    pub fn compute_leaf_static(
        cfg: &MerkleChannelCfg,
        trace_hash: &[u8; 32],
        index: usize,
        f: F,
        s: F,
        q: F,
    ) -> F {
        let ds = DsLabel {
            arity: cfg.layer_arities[0],
            level: LEAF_LEVEL_DS,
            position: index as u64,
            tree_label: cfg.tree_label,
        };

        Self::compress_static(ds, trace_hash, &[f, s, q])
    }

    pub fn push_leaf(&mut self, f: F, s: F, q: F) {
        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }

        let idx = self.levels[0].len();

        let leaf = Self::compute_leaf_static(&self.cfg, &self.trace_hash, idx, f, s, q);

        self.levels[0].push(leaf);
    }

    pub fn finalize(&mut self) -> F {
        let mut level = 0;

        while self.levels[level].len() > 1 {
            let arity = self.cfg.layer_arities[level];
            let mut cur = self.levels[level].clone();

            if cur.len() % arity != 0 {
                let last = *cur.last().unwrap();
                cur.resize(cur.len() + (arity - cur.len() % arity), last);
            }

            let parents: Vec<F> = cur
                .chunks(arity)
                .enumerate()
                .map(|(i, c)| {
                    let ds = DsLabel {
                        arity,
                        level: level as u32 + 1,
                        position: i as u64,
                        tree_label: self.cfg.tree_label,
                    };
                    self.compress(ds, c)
                })
                .collect();

            self.levels.push(parents);
            level += 1;
        }

        self.levels.last().unwrap()[0]
    }

    pub fn open(&self, index: usize) -> MerkleOpening {
        let mut idx = index;
        let mut path = Vec::new();

        for level in 0..self.levels.len() - 1 {
            let nodes = &self.levels[level];
            let arity = self.cfg.layer_arities[level];
            let group_start = (idx / arity) * arity;

            let mut group = Vec::with_capacity(arity);
            for i in 0..arity {
                let pos = group_start + i;
                if pos < nodes.len() {
                    group.push(nodes[pos]);
                } else {
                    group.push(*nodes.last().unwrap());
                }
            }

            let siblings: Vec<F> = group
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| {
                    if group_start + i != idx {
                        Some(x)
                    } else {
                        None
                    }
                })
                .collect();

            path.push(siblings);
            idx /= arity;
        }

        MerkleOpening {
            leaf: self.levels[0][index],
            path,
            index,
        }
    }

    pub fn verify_opening(
        cfg: &MerkleChannelCfg,
        root: F,
        opening: &MerkleOpening,
        trace_hash: &[u8; 32],
    ) -> bool {
        let mut cur = opening.leaf;
        let mut idx = opening.index;

        for (level, siblings) in opening.path.iter().enumerate() {
            let arity = cfg.layer_arities[level];
            let pos = idx % arity;

            let mut children = Vec::with_capacity(arity);
            let mut sibs = siblings.iter();

            for i in 0..arity {
                if i == pos {
                    children.push(cur);
                } else {
                    children.push(*sibs.next().unwrap());
                }
            }

            let ds = DsLabel {
                arity,
                level: level as u32 + 1,
                position: (idx / arity) as u64,
                tree_label: cfg.tree_label,
            };

            cur = Self::compress_static(ds, trace_hash, &children);
            idx /= arity;
        }

        cur == root
    }
}
