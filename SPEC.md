# SPEC.md: "The Human How"; Exact technical requirements, environment setup, implementation details, etc.

## Execution Description

Builds src/cublas_gemm.cu into bin/cublas_gemm and times C = alphaAB + betaC via cuBLAS. function, dtype (default f32_r), M/N/K, transposeA/B, alpha/beta, lda/ldb/ldc, batch_count, warmup_iters, num_iterations, and norm_check come from yaml. Collector runs the harness twice and writes pass1, pass2, and summary CSV rows. This is not cublas-bench Sweep dimensions: function, dtype, M, N, K, transposeA, transposeB, alpha.

## Parameters

| Parameter | CLI Flag | Tested Values | Default | Description |
| --- | --- | --- | --- | --- |
| function | `--function` | smoke=true, baseline=true, extended=true | true | From Parameter list; see Execution Description With Parameters. |
| dtype | `--dtype` | smoke=f32_r, baseline=f32_r, extended=f32_r | f32_r | From Parameter list; see Execution Description With Parameters. |
| M | `--m` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| N | `--n` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| K | `--k` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| transposeA | `--transposea` | smoke=N, baseline=N, extended=N | N | From Parameter list; see Execution Description With Parameters. |
| transposeB | `--transposeb` | smoke=N, baseline=N, extended=N | N | From Parameter list; see Execution Description With Parameters. |
| alpha | `--alpha` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| beta | `--beta` | smoke=0, baseline=0, extended=0 | 0 | From Parameter list; see Execution Description With Parameters. |
| lda | `--lda` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| ldb | `--ldb` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| ldc | `--ldc` | smoke=64, baseline=4096, extended=4096 | 4096 | From Parameter list; see Execution Description With Parameters. |
| batch_count | `--batch-count` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| warmup_iters | `--warmup-iters` | smoke=2, baseline=5, extended=10 | 5 | From Parameter list; see Execution Description With Parameters. |
| num_iterations | `--num-iterations` | smoke=2, baseline=30000, extended=84000 | 30000 | From Parameter list; see Execution Description With Parameters. |
| norm_check | `--norm-check` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |

### Derived quantities

Matrix extents come from the Parameters table. Example: use the tested value of `function` from Profile Parameter Values. Do not invent a leading dimension that the spec does not state.

## Invocation

```bash
Compile cublas_gemm.cu with nvcc; run bin/cublas_gemm twice via collect_cublas_gemm.py
```

## Raw Output Format

cuBLAS GEMM CSV from collect_cublas_gemm.py

check,status,function,dtype,M,N,K,transposeA,transposeB,alpha,beta,lda,ldb,ldc,batch_count,warmup_iters,num_iterations,norm_check,achieved_compute_tflops,kernel_time_msec,minimum_operand_traffic_gb_s,relative_l1_checked_columns,norm_error_2
gemm,ok,gemm,f16,4096,4096,4096,N,N,1,0,4096,4096,4096,1,5,20,1,80,0.5,2000,1e-3,1e-3

## Metrics

- **#1: Achieved compute throughput** — stored as `achieved_compute_tflops`.
- **#2: Kernel execution time** — stored as `kernel_time_msec`.
- **#3: Minimum operand traffic, GB/s** — stored as `minimum_operand_traffic_gb_s`.
- **#4: Relative L1 of checked columns** — stored as `relative_l1_checked_columns`.
- **#5: L2 norm error** — stored as `norm_error_2`.

## Framework

Builds src/cublas_gemm.cu into bin/cublas_gemm and times C = alphaAB + betaC via cuBLAS. function, dtype (default f32_r), M/N/K, transposeA/B, alpha/beta, lda/ldb/ldc, batch_count, warmup_iters, num_iterations, and norm_check come from yaml. Collector runs the harness twice and writes pass1, pass2, and summary CSV rows.

## Installation and Execution Summary

Compile src/cublas_gemm.cu with nvcc and run bin/cublas_gemm twice with CUDA-event timing and optional norm_check, then write pass1/pass2/summary METRICS_CSV (achieved_compute_tflops, kernel_time_msec, bandwidth, norm errors), to measure cuBLAS GEMM throughput. This is not the cublas-bench CLI

## Platform Portability

- **AMD (primary):** ```bash
Compile cublas_gemm.cu with nvcc; run bin/cublas_gemm twice via collect_cublas_gemm.py
```
- **NVIDIA:** Native NVIDIA CUDA workload. Execute on the stated Ubuntu release with the host NVIDIA driver and CUDA userspace. ROCm porting notes do not apply.

## Model Context Protocols

- **Active:** None

## Execution-Loop Validation Contract

EXECUTION CHAIN: `run_benchmark.sh` ➔ raw output ➔ `scripts/parse_results.py` ➔ `results/benchmark.db` ➔ `scripts/validate_results.py`

This benchmark uses a lightweight, SQLite-integrated execution loop for result validation. All validation is performed by `scripts/validate_results.py`.

### Validation script usage

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional; select the installed interpreter

# After a live run:
".venv/bin/python" scripts/validate_results.py --db results/benchmark.db

# CI / no-GPU path (seeds fixture and validates it):
".venv/bin/python" scripts/validate_results.py --seed-fixture --quiet

# Override DB path via environment variable:
BENCHMARK_DB=tests/fixtures/benchmark.db \
  ".venv/bin/python" scripts/validate_results.py
```

### Run artifact contract

cuBLAS GEMM CSV from collect_cublas_gemm.py

check,status,function,dtype,M,N,K,transposeA,transposeB,alpha,beta,lda,ldb,ldc,batch_count,warmup_iters,num_iterations,norm_check,achieved_compute_tflops,kernel_time_msec,minimum_operand_traffic_gb_s,relative_l1_checked_columns,norm_error_2
gemm,ok,gemm,f16,4096,4096,4096,N,N,1,0,4096,4096,4096,1,5,20,1,80,0.5,2000,1e-3,1e-3

```bash
bash run_benchmark.sh --help
bash run_benchmark.sh --profile smoke --validate
bash run_benchmark.sh --profile baseline --validate
bash run_benchmark.sh --profile extended --validate
```
`run_benchmark.sh --help` prints usage and exits. The harness calls `scripts/ensure_setup.sh` when `.setup_state` is absent.

### Required integrity checks (built into `validate_results.py`)

1. Latest run exists and `runs.status = 'ok'`.
2. `run.error_message` is NULL.
3. `started_at` and `finished_at` are valid ISO-8601 UTC strings.
4. All required aggregate metrics in `runs` are non-NULL and finite.
5. All required aggregate metrics are physically sensible (positive values). Builds src/cublas_gemm.cu into bin/cublas_gemm and times C = alphaAB + betaC via cuBLAS. function, dtype (default f32_r), M/N/K, transposeA/B, alpha/beta, lda/ldb/ldc, batch_count, warmup_iters, num_iterations, and norm_check come from yaml. Collector runs the harness twice and writes pass1, pass2, and summary CSV rows.
6. At least 2 sample rows exist for the latest `run_id` (sweep coverage).
7. No sample has `status = 'error'`.
8. Builds src/cublas_gemm.cu into bin/cublas_gemm and times C = alphaAB + betaC via cuBLAS. function, dtype (default f32_r), M/N/K, transposeA/B, alpha/beta, lda/ldb/ldc, batch_count, warmup_iters, num_iterations, and norm_check come from yaml. Collector runs the harness twice and writes pass1, pass2, and summary CSV rows.

### Baseline / Threshold configuration (`config/benchmark_config.yaml`)

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

Threshold key suffixes encode comparison direction when `thresholds:` is present: `_min` → observed value must be ≥ threshold. `_max` → observed value must be ≤ threshold. Informational `baselines:` ranges are not pass/fail gates.
