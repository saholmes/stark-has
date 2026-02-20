High-Arity STARKS via DEEP-ALI Constraint merging

This is the implementation of the paper for ZK Proof workshop.  

The repository is structured as follows:  

results:

This has the collected results from benchmark runs on t4g.micro and AVX512.xlarge instances on AWS
We have results for r = 26, r = 32 and r = 52
We also have results for FS-transform supported hash's 
- SHA3-256
- keccak
- Blake3

An interactive graph visualization tool is available to display benchmark results
https://saholmes.github.io/stark-has/

The code is split into 2 code bases:  

stark-has-golidlocks
stark-has-pallas

The intention of the code is to collect benchmark data using RUST criterion.  

The parameter file for r values and arity parameters is in the file channel/benches/end_to_end.rs

To run the benchmarks you need to move to the crates/channel directory
We implement rayon threads and this is enabled through the following command

We recommend using nohup if using remote server.  If not, simply run without nohup

For IoT with 2 vcpu:  
RAYON_NUM_THREADS=2 nohup cargo bench \
  --features parallel,sha3 \
  --bench end_to_end \
  -- --sample-size 20 --measurement-time 10 \
  > benchmark.log 2>&1 & 

For AVX512 8vcpu

RAYON_NUM_THREADS=8 nohup cargo bench \
  --features parallel,sha3 \
  --bench end_to_end \
  -- --sample-size 20 --measurement-time 10 \
  > benchmark.log 2>&1 &

You can then see progress by typing
tails -f benchmark.log

The default output file is 
benchmarkdata.csv

