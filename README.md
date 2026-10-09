# GEMM / cuBLAS Microbenchmark Benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/garys-gpu-benchmarks/417-gpu-bench-nvidia-gemm-cublas-micro-ubu2604/actions/workflows/ci.yml/badge.svg)](https://github.com/garys-gpu-benchmarks/417-gpu-bench-nvidia-gemm-cublas-micro-ubu2604/actions/workflows/ci.yml)

Target: Ubuntu 26.04 · NVIDIA · see Hardware Requirements. This is a host benchmark, not a laptop `pip install` project.

## Quick Start

```bash
git clone https://github.com/garys-gpu-benchmarks/417-gpu-bench-nvidia-gemm-cublas-micro-ubu2604.git
cd 417-gpu-bench-nvidia-gemm-cublas-micro-ubu2604
sudo bash setup.sh --assume-yes
bash run_benchmark.sh --profile smoke --validate
```
Results are written to `results/benchmark.db` and `results/summary.json`.

This workload is executed on the validation host after the repository is copied there. `setup.sh` and `run_benchmark.sh` do not open an outbound SSH session.

Prerequisites: Ubuntu 26.04; NVIDIA; Python 3.14.4; root or sudo for `setup.sh`. Framework: Bash, SQLite, Python, PyYAML, CUDA Runtime, NVCC, cuBLAS (GEMM). This is a host benchmark, not a laptop `pip install` project.

```mermaid
flowchart LR
  setup.sh --> run_benchmark.sh --> parse_results.py --> results/benchmark.db
```

## 1. Overview

Builds src/cublas_gemm.cu into bin/cublas_gemm and times C = alphaAB + betaC via cuBLAS. function, dtype (default f32_r), M/N/K, transposeA/B, alpha/beta, lda/ldb/ldc, batch_count, warmup_iters, num_iterations, and norm_check come from yaml. Collector runs the harness twice and writes pass1, pass2, and summary CSV rows. This is not cublas-bench Sweep dimensions: function, dtype, M, N, K, transposeA, transposeB, alpha.

## 2. What It Validates

- Validates cuBLAS GEMM TFLOPS, kernel time, bandwidth, and optional norm errors from bin/cublas_gemm
- #1: Achieved compute throughput (achieved_compute_tflops); is present and physically sensible.
- #2: Kernel execution time (kernel_time_msec); is present and physically sensible.
- #3: Minimum operand traffic, GB/s (minimum_operand_traffic_gb_s); is present and physically sensible.
- #4: Relative L1 of checked columns (relative_l1_checked_columns); is present and physically sensible.
- #5: L2 norm error (norm_error_2) is present and physically sensible.

## 3. Metrics Captured

- **#1: Achieved compute throughput** — stored as `achieved_compute_tflops`.
- **#2: Kernel execution time** — stored as `kernel_time_msec`.
- **#3: Minimum operand traffic, GB/s** — stored as `minimum_operand_traffic_gb_s`.
- **#4: Relative L1 of checked columns** — stored as `relative_l1_checked_columns`.
- **#5: L2 norm error** — stored as `norm_error_2`.

## 4. Hardware Requirements

### Supported environment

- OS: Ubuntu 26.04
- GPU vendor: NVIDIA
- Framework family: Bash, SQLite, Python, PyYAML, CUDA Runtime, NVCC, cuBLAS (GEMM)
- Python: Python 3.14.4

### Reference validation environment

The tables below describe the machine used to generate the reference results. They are not a requirement that every user buy that exact cloud instance.

### System

Builds src/cublas_gemm.cu into bin/cublas_gemm and times C = alphaAB + betaC via cuBLAS. function, dtype (default f32_r), M/N/K, transposeA/B, alpha/beta, lda/ldb/ldc, batch_count, warmup_iters, num_iterations, and norm_check come from yaml. Collector runs the harness twice and writes pass1, pass2, and summary CSV rows.

### GPU

Ubuntu 26.04 / NVIDIA / Bash, SQLite, Python, PyYAML, CUDA Runtime, NVCC, cuBLAS (GEMM)

## 5. Software Requirements

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 |
| Kernel | kernel 7.0.0 |
| Python | Python 3.14.4 |
| ROCm | CUDA 13.3 |
| rocBLAS | cuBLAS (bundled with CUDA 13.3) |

Builds src/cublas_gemm.cu into bin/cublas_gemm and times C = alphaAB + betaC via cuBLAS. function, dtype (default f32_r), M/N/K, transposeA/B, alpha/beta, lda/ldb/ldc, batch_count, warmup_iters, num_iterations, and norm_check come from yaml. Collector runs the harness twice and writes pass1, pass2, and summary CSV rows.

## 6. Installation

```bash
Compile cublas_gemm.cu with nvcc; run bin/cublas_gemm twice via collect_cublas_gemm.py
```

## 7. Running the Benchmark

```bash
Compile cublas_gemm.cu with nvcc; run bin/cublas_gemm twice via collect_cublas_gemm.py
```

**Validating results separately:**

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional
python3 -m venv .venv
source ".venv/bin/activate"
".venv/bin/python" scripts/validate_results.py
```

## 8. Output

### `results/benchmark.db` (SQLite)

cuBLAS GEMM CSV from collect_cublas_gemm.py

check,status,function,dtype,M,N,K,transposeA,transposeB,alpha,beta,lda,ldb,ldc,batch_count,warmup_iters,num_iterations,norm_check,achieved_compute_tflops,kernel_time_msec,minimum_operand_traffic_gb_s,relative_l1_checked_columns,norm_error_2
gemm,ok,gemm,f16,4096,4096,4096,N,N,1,0,4096,4096,4096,1,5,20,1,80,0.5,2000,1e-3,1e-3

```bash
Compile cublas_gemm.cu with nvcc; run bin/cublas_gemm twice via collect_cublas_gemm.py
```

### `results/summary.json`

Consolidated metrics from the most recent run — suitable for CI artifact upload or dashboard ingestion.

### `results/raw/<timestamp>.txt`

cuBLAS GEMM CSV from collect_cublas_gemm.py

check,status,function,dtype,M,N,K,transposeA,transposeB,alpha,beta,lda,ldb,ldc,batch_count,warmup_iters,num_iterations,norm_check,achieved_compute_tflops,kernel_time_msec,minimum_operand_traffic_gb_s,relative_l1_checked_columns,norm_error_2
gemm,ok,gemm,f16,4096,4096,4096,N,N,1,0,4096,4096,4096,1,5,20,1,80,0.5,2000,1e-3,1e-3

## 9. Baselines / Thresholds

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

## 10. Troubleshooting

**`setup.sh` missing collector**
Create cannot finish without `scripts/collect_workload.py`.

**`self_check` overlay rewritten**
Do not overwrite files listed in `results/overlay_lock.json`.

**Remote SSH drop during setup**
Reconnect and resume `bash setup.sh --assume-yes`. Do not wipe `.venv` or `.cache`.

## 11. NVIDIA H100 Coding Differences

Native NVIDIA CUDA workload. Execute on the stated Ubuntu release with the host NVIDIA driver and CUDA userspace. ROCm porting notes do not apply.

## Repository layout

```text
.
├── setup.sh
├── run_benchmark.sh
├── benchmark_specification.json
├── .github/workflows/      # thin CI callers (see Continuous Integration)
├── config/
├── scripts/
├── src/
├── tests/
├── docs/
├── results/
└── LICENSE
```

## Continuous Integration

| Workflow | Runs on | When | What it does |
|---|---|---|---|
| [CI](.github/workflows/ci.yml) | GitHub-hosted runner | every pull request, and every push to `main` | shellcheck, ruff, `bash -n`, `compileall`, `run_benchmark.sh --help`, specification schema, the results validator on a seeded fixture, required files, and actionlint. No GPU and no benchmark run. |
| [GPU Smoke Benchmark](.github/workflows/gpu-smoke.yml) | self-hosted runner labeled `gpu`, `nvidia`, `ubu2604` | only when started by hand: **Actions → GPU Smoke Benchmark → Run workflow** (choose `smoke`, `baseline` or `extended`) | Verifies the pre-provisioned GPU stack, records `results/environment.json` (driver, runtime, kernel, GPU), runs the profile with `--validate`, shows headline metrics on the run page, and uploads the results. |

Both files are short callers. The steps themselves live once, for every workload in the suite, in [`garys-gpu-benchmarks/shared-workflows`](https://github.com/garys-gpu-benchmarks/shared-workflows), pinned at `@v1`. The GPU workflow is never triggered by pull requests, so code from a fork cannot run on the GPU host.

### Running it as part of the NVIDIA Ubuntu 26.04 bundle

This repository is one of the 32 workloads in [`bundle-nvidia-ubuntu-2604`](https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2604), which holds them as git submodules. To put the whole bundle on a GPU host and run this workload from it:

```bash
git clone --recurse-submodules https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2604 /opt/benchmarks
cd /opt/benchmarks/417-gpu-bench-nvidia-gemm-cublas-micro-ubu2604
bash run_benchmark.sh --profile smoke --validate
```
