# ⚡CUDACyclone: GPU Satoshi Puzzle Solver

Cyclone CUDA is the GPU-powered version of the **Cyclone** project, designed to achieve extreme performance in solving Satoshi puzzles on modern NVIDIA GPUs.  
Leveraging **CUDA**, **warp-level parallelism**, and **batch EC operations**, Cyclone CUDA pushes the limits of cryptographic key search.

Secp256k1 math is based on the excellent work from [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch), and [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul) with major CUDA-specific modifications.  
Special thanks to Jean-Luc Pons for his foundational contributions to the cryptographic community.

Cyclone CUDA also is the **simplest CUDA-based project** for solving Satoshi puzzles on GPU.  
It was designed with clarity and minimalism in mind — making it easy to **compile, understand, and run**, even for those new to CUDA programming.  

Despite its simplicity, Cyclone CUDA leverages **massive GPU parallelism** to achieve extreme performance in elliptic curve computations and Hash160 pipelines. 

⚠️ **Achieved 6Gkeys/s on RTX4090.**  

---

## 🚀 Key Features

- **GPU Acceleration**: Optimized for NVIDIA GPUs with full CUDA support.
- **Massive Parallelism**: Tens of thousands of threads computing elliptic curve points and **hash160** simultaneously.
- **Batch EC Operations**: Efficient group addition and modular inversion with warp-level optimizations.
- **Grid/Batch Control**: Fully configurable GPU execution with `--grid` parameter (threads per batch × points per batch).
- **Cross-Platform**: Works on Linux and Windows (via WSL2 or MinGW cross-compilation).
- **Cross Architecture**: Automatic compilation for different architectures (75 86 89).
- **Extremely low VRAM usage**: Key feature! For low price rented GPU.
---

## 🚀 Options
- **--range**: range of search. Must be a power of two!
- **--address**: P2PKH address.
- **--target-hash160**: the same as address but hash160.
- **--grid**: very usefull parameter. Example --grid 512,512 - first 512 - number of points each thread will process in one batch (Points batch size)., second 512 - number of threads in one group (Threads per batch).
- **--slices**: batch per thread for one kernel launch.

---

### ❔ Community benchmarks

Users have reported the following speeds:

| GPU               | Grid      | Speed (Mkeys/s) | Notes                  |
|-------------------|-----------|-----------------|------------------------|
| RTX 4090          | 128,1024  | 6214 Mkeys/s    | Community report       |
| RTX 4090          | 512,512   | 6038 Mkeys/s    | Community report       |
| RTX 4060          | 512,512   | 1238 Mkeys/s    | My own GPU             |
| RTX 4070 Ti Super | 512,1024  | 3170 Mkeys/s    | Community report       |
| L4-2Q             | 512,256   | 1360 Mkeys/s    | Community report       |
| RTX3070 mobile    | 256,256   | 1150 Mkeys/s    | Community report       |

---

## 🔷 Example Output

Below is an example run of **Cyclone CUDA**.  

**RTX4060**

```bash
./CUDACyclone --range 2000000000:3FFFFFFFFF --address 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2 --grid 512,256
======== PrePhase: GPU Information ====================
Device               : NVIDIA GeForce RTX 4060 (compute 8.9)
SM                   : 24
ThreadsPerBlock      : 256
Blocks               : 4096
Points batch size    : 512
Batches/SM           : 256
Memory utilization   : 6.9% (538.3 MB / 7.63 GB)
------------------------------------------------------- 
Total threads        : 1048576

======== Phase-1: Brooteforce =========================
Time: 8.0 s | Speed: 1268.9 Mkeys/s | Count: 10204470016 | Progress: 7.42 %

======== FOUND MATCH! =================================
Private Key   : 00000000000000000000000000000000000000000000000000000022382FACD0
Public Key    : 03C060E1E3771CBECCB38E119C2414702F3F5181A89652538851D2E3886BDD70C6
```

**RTX4090**
```bash
./CUDACyclone --range 400000000000000000:7fffffffffffffffff --address 1PWo3JeB9jrGwfHDNpdGK54CRas/fsVzXU --grid 512,512
======= PrePhase: GPU Information
Device: NVIDIA GeForce RTX 4090 (compute 8.9)
SM                   : 128
ThreadsPerBlock      : 256
Blocks               : 65536
Points batch size    : 512
Batches/SM           : 512
Memory utilization   : 14.5% (3.42 GB / 23.5 GB)
------------------------------------------------------- 
Total threads: 16777216

======== Phase-1: Brooteforce =========================
Time: 30.1 s | Speed: 6038.9 Mkeys/s | Count: 182904267648 | Progress: 0.00 %
```
**RTX5090**
```bash
./CUDACyclone --range 200000000000:3fffffffffff --address 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP —-grid 128,256
======== PrePhase: GPU Information ====================
Device               : NVIDIA GeForce RTX 5090 (compute 12.0)
SM                   : 170
ThreadsPerBlock      : 256
Blocks               : 1024
Points batch size    : 128
Batches/SM           : 8
Memory utilization   : 1.7% (557.3 MB / 31.4 GB)
------------------------------------------------------- 
Total threads        : 262144

======== Phase-1: Brooteforce =========================
Time: 7.0 s | Speed: 8408.0 Mkeys/s | Count: 58545467200 | Progress: 0.17 %

```
## 🛠️ Getting Started
To get started with CUDACyclone, clone the repository and type **make**  
For totaly clean system (big thanks for **dev_nullish**):
```bash
apt update;
apt-get install -y joe;
apt-get install -y zip;
apt-get install -y screen;
apt-get install -y curl libcurl4;
apt-get install build-essential;
apt-get install -y gcc;
apt-get install -y make;
apt install cuda-toolkit;
git clone https://github.com/Dookoo2/CUDACyclone.git
make
```
## 🚧**Version**
**V1.1**: Switch pGx/pGy to constant memory due to VRAM thermal throttling  
**V1.0**: Release



## ✌️**TIPS**
BTC: bc1qtq4y9l9ajeyxq05ynq09z8p52xdmk4hqky9c8n
