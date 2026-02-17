use ark_ff::UniformRand;
use ark_pallas::Fr as F;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::time::Duration;

// Existing APIs from your crate(s)
use channel::{build_vk_plain, prove_plain, verify_plain};

use deep_ali::fri::{
    deep_fri_proof_size_bytes, deep_fri_prove, deep_fri_verify, AliA, AliE, AliS, AliT,
    DeepAliRealBuilder, DeepFriParams, DeepFriProof,
};

// ---------------------
// CSV record
// ---------------------

#[derive(Default, Clone)]
struct CsvRow {
    label: String,
    schedule: String,
    k: usize,
    proof_bytes: usize,
    prove_s: f64,           // single timed prove (seconds)
    verify_ms: f64,         // single timed verify (milliseconds)
    prove_elems_per_s: f64, // n0 / prove_s
    // deltas vs paper
    delta_size_pct: f64,
    delta_prove_pct: f64,
    delta_verify_pct: f64,
    delta_throughput_pct: f64,
}

impl CsvRow {
    fn header() -> &'static str {
        "csv,label,k,schedule,proof_bytes,prove_s,verify_ms,prove_elems_per_s,delta_size_pct_vs_paper,delta_prove_pct_vs_paper,delta_verify_pct_vs_paper,delta_throughput_pct_vs_paper"
    }
    fn to_line(&self) -> String {
        format!(
            "csv,{},{},{},{},{:.6},{:.3},{:.6},{:.2},{:.2},{:.2},{:.2}\n",
            self.label,
            self.k,
            self.schedule,
            self.proof_bytes,
            self.prove_s,
            self.verify_ms,
            self.prove_elems_per_s,
            self.delta_size_pct,
            self.delta_prove_pct,
            self.delta_verify_pct,
            self.delta_throughput_pct
        )
    }
    fn print_stdout(&self) {
        // Also print to stdout (without trailing newline because we add \n in to_line)
        print!(
            "csv,{},{},{},{},{:.6},{:.3},{:.6},{:.2},{:.2},{:.2},{:.2}\n",
            self.label,
            self.k,
            self.schedule,
            self.proof_bytes,
            self.prove_s,
            self.verify_ms,
            self.prove_elems_per_s,
            self.delta_size_pct,
            self.delta_prove_pct,
            self.delta_verify_pct,
            self.delta_throughput_pct
        );
    }
}

// ---------------------
// Schedule helpers
// ---------------------

fn schedule_str(s: &[usize]) -> String {
    format!(
        "[{}]",
        s.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn log2_pow2(x: usize) -> usize {
    assert!(
        x.is_power_of_two(),
        "schedule factors must be powers of two"
    );
    x.trailing_zeros() as usize
}

fn k_min_for_schedule(schedule: &[usize]) -> usize {
    schedule.iter().map(|&m| log2_pow2(m)).sum()
}

fn divides_chain(n0: usize, schedule: &[usize]) -> bool {
    let mut n = n0;
    for &m in schedule {
        if n % m != 0 {
            return false;
        }
        n /= m;
    }
    true
}

fn ks_for_schedule(schedule: &[usize], k_lo: usize, k_hi: usize) -> Vec<usize> {
    let k_min = k_min_for_schedule(schedule);
    let mut out = Vec::new();
    for k in k_lo.max(k_min)..=k_hi {
        let n0 = 1usize << k;
        if divides_chain(n0, schedule) {
            out.push(k);
        }
    }
    out
}

// ---------------------
// Plain benches (unchanged behavior)
// ---------------------

fn bench_e2e_plain(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e_plain");
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(20));
    g.sample_size(10);

    for &k in &[12usize, 14, 16] {
        let n = 1usize << k;
        let ds = F::from(2025u64);

        let mut rng = StdRng::seed_from_u64(7);
        let witness: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let vk = build_vk_plain(k, ds);
        let pre_proof = prove_plain(&vk, &witness);

        let vk_size = bincode::serialize(&vk).map(|b| b.len()).unwrap_or(0);
        let proof_size = bincode::serialize(&pre_proof).map(|b| b.len()).unwrap_or(0);
        eprintln!("plain k={} vk={}B proof={}B", k, vk_size, proof_size);

        g.throughput(Throughput::Elements(n as u64));

        // Prove
        g.bench_with_input(BenchmarkId::new("prove", k), &k, |b, &_k| {
            b.iter_batched(
                || (),
                |_| {
                    let proof = prove_plain(&vk, &witness);
                    criterion::black_box(proof);
                },
                BatchSize::SmallInput,
            )
        });

        // Verify
        g.bench_with_input(BenchmarkId::new("verify", k), &k, |b, &_k| {
            b.iter(|| {
                let ok = verify_plain(&vk, &pre_proof);
                assert!(ok);
            })
        });
    }
    g.finish();
}

// ---------------------
// MF-FRI benches + Expanded CSV + file output
// ---------------------

fn bench_e2e_mf_fri(c: &mut Criterion) {
    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("e2e_mf_fri");

    // Tuning for long benches
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(20));
    g.sample_size(10);

    // Global params
    let r: usize = 30;
    let seed_z: u64 = 0xDEEF_BAAD;

    // k window
    let k_lo = 11usize;
    // bump high end so schedules with 128 have room (k must be ≥ 7 + …)
    let k_hi = 19usize;

    // Presets: keep "paper" first so baseline is available for all ks
    let presets: &[(&str, &[usize])] = &[
        ("paper", &[16, 16, 8]),
        ("mod16", &[16, 16, 16, 16]),
        ("uni32x3", &[32, 32, 32]),
        ("uni64x2x8", &[64, 64, 8]),
        ("hi64_32_8", &[64, 32, 8]),
        ("hi32_32_16", &[32, 32, 16]),
        // New: schedules using 128-fold layers
        ("uni128", &[128]),
        ("uni128x2", &[128, 128]),
        ("hi128_64", &[128, 64]),
        ("hi128_32", &[128, 32]),
        ("hi128_16", &[128, 16]),
        ("hi128_64_8", &[128, 64, 8]),
        ("hi128_32_8", &[128, 32, 8]),
    ];

    // Deterministic input generation
    let mut rng_seed = 1337u64;

    // Store per-k baseline (paper) for delta computation
    let mut paper_baseline: HashMap<usize, CsvRow> = HashMap::new();

    // Prepare CSV file: truncate and write header once
    let file = File::create("r30benchmarkdata.csv")
        .expect("failed to create r30benchmarkdata.csv for writing");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", CsvRow::header()).expect("failed to write CSV header");
    writer.flush().ok();

    // Also print header to stdout
    println!("{}", CsvRow::header());

    for &(label, schedule) in presets {
        let ks = ks_for_schedule(schedule, k_lo, k_hi);
        if ks.is_empty() {
            eprintln!(
                "skip: label={} schedule={:?} k_min={} has no ks in [{}..={}]",
                label,
                schedule,
                k_min_for_schedule(schedule),
                k_lo,
                k_hi
            );
            continue;
        }

        for &k in &ks {
            let n0 = 1usize << k;
            g.throughput(Throughput::Elements(n0 as u64));

            // Inputs
            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let mut rng = StdRng::seed_from_u64(rng_seed);
            let a: AliA = (0..n0).map(|_| F::rand(&mut rng)).collect();
            let s: AliS = (0..n0).map(|_| F::rand(&mut rng)).collect();
            let e: AliE = (0..n0).map(|_| F::rand(&mut rng)).collect();
            let t: AliT = (0..n0).map(|_| F::rand(&mut rng)).collect();

            let params = DeepFriParams {
                schedule: schedule.to_vec(),
                r,
                seed_z,
            };
            let builder = DeepAliRealBuilder::default();

            eprintln!(
                "mf-fri setup: label={} k={} (n0={}) schedule={:?} r={}",
                label, k, n0, schedule, r
            );

            // Precompute proof for verify bench and size
            eprintln!("mf-fri precompute proof…");
            let pre_proof: DeepFriProof = deep_fri_prove(&builder, &a, &s, &e, &t, n0, &params);
            let proof_size_bytes = deep_fri_proof_size_bytes(&pre_proof);
            eprintln!(
                "mf-fri label={} k={} r={} proof≈{}B",
                label, k, r, proof_size_bytes
            );

            // Criterion bench: prove
            let prove_id = BenchmarkId::new(format!("prove-{}", label), k);
            g.bench_with_input(prove_id, &k, |b, &_k| {
                b.iter_batched(
                    || (),
                    |_| {
                        let proof = deep_fri_prove(&builder, &a, &s, &e, &t, n0, &params);
                        criterion::black_box(proof);
                    },
                    BatchSize::SmallInput,
                )
            });

            // Criterion bench: verify
            eprintln!("mf-fri precompute verify warmup…");
            assert!(deep_fri_verify(&params, &pre_proof));
            let verify_id = BenchmarkId::new(format!("verify-{}", label), k);
            g.bench_with_input(verify_id, &k, |b, &_k| {
                b.iter(|| {
                    let ok = deep_fri_verify(&params, &pre_proof);
                    assert!(ok);
                })
            });

            // Single-shot timings to populate CSV
            // Prove
            let t0 = std::time::Instant::now();
            let _tmp_proof = deep_fri_prove(&builder, &a, &s, &e, &t, n0, &params);
            let prove_s = t0.elapsed().as_secs_f64();

            // Verify
            let t1 = std::time::Instant::now();
            let ok = deep_fri_verify(&params, &pre_proof);
            assert!(ok);
            let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

            let prove_elems_per_s = (n0 as f64) / prove_s;

            let mut row = CsvRow {
                label: label.to_string(),
                schedule: schedule_str(schedule),
                k,
                proof_bytes: proof_size_bytes,
                prove_s,
                verify_ms,
                prove_elems_per_s,
                delta_size_pct: f64::NAN,
                delta_prove_pct: f64::NAN,
                delta_verify_pct: f64::NAN,
                delta_throughput_pct: f64::NAN,
            };

            // Compute deltas vs paper baseline for this k
            if label == "paper" {
                paper_baseline.insert(
                    k,
                    CsvRow {
                        label: row.label.clone(),
                        schedule: row.schedule.clone(),
                        k: row.k,
                        proof_bytes: row.proof_bytes,
                        prove_s: row.prove_s,
                        verify_ms: row.verify_ms,
                        prove_elems_per_s: row.prove_elems_per_s,
                        delta_size_pct: 0.0,
                        delta_prove_pct: 0.0,
                        delta_verify_pct: 0.0,
                        delta_throughput_pct: 0.0,
                    },
                );
                row.delta_size_pct = 0.0;
                row.delta_prove_pct = 0.0;
                row.delta_verify_pct = 0.0;
                row.delta_throughput_pct = 0.0;
            } else if let Some(base) = paper_baseline.get(&k) {
                row.delta_size_pct = 100.0 * (row.proof_bytes as f64 - base.proof_bytes as f64)
                    / (base.proof_bytes as f64);
                row.delta_prove_pct = 100.0 * (row.prove_s - base.prove_s) / base.prove_s;
                row.delta_verify_pct = 100.0 * (row.verify_ms - base.verify_ms) / base.verify_ms;
                row.delta_throughput_pct = 100.0 * (row.prove_elems_per_s - base.prove_elems_per_s)
                    / base.prove_elems_per_s;
            } else {
                eprintln!(
                    "warn: missing paper baseline for k={}, deltas set to NaN",
                    k
                );
            }

            // Emit to stdout
            row.print_stdout();

            // Append to file
            let line = row.to_line();
            writer
                .write_all(line.as_bytes())
                .expect("failed to write CSV row");
            writer.flush().ok();
        }
    }

    g.finish();
}

criterion_group!(e2e, bench_e2e_plain, bench_e2e_mf_fri);
criterion_main!(e2e);
