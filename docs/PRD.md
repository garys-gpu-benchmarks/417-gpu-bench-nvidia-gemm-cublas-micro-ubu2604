# PRD.md:  "The Why"; Product requirements, benchmark metadata table, high-level requirements, etc.

Product Requirements Document

"The Why"; Product requirements, benchmark metadata table, high-level requirements, etc. Defines the benchmark goal, validation objective, test name, benchmark number, category, and high-level success criteria.

## Benchmark Matrix Document Metadata (via benchmark_specification.json)

This PRD.md section is populated from benchmark_specification.json, which is the structured source of benchmark-specific product requirements.

## Workload Number
417

## Workload Name
GEMM / cuBLAS Microbenchmark

## Execution Summary (Run and Measure)
Compile src/cublas_gemm.cu with nvcc and run bin/cublas_gemm twice with CUDA-event timing and optional norm_check, then write pass1/pass2/summary METRICS_CSV (achieved_compute_tflops, kernel_time_msec, bandwidth, norm errors), to measure cuBLAS GEMM throughput. This is not the cublas-bench CLI

## Main Goal
Measure GEMM throughput across dtypes/shapes

## Validation Objective
Validates cuBLAS GEMM TFLOPS, kernel time, bandwidth, and optional norm errors from bin/cublas_gemm

## Workload Category
Compute & Math Kernels

## Validation Requirement

The benchmark must include an automated SQLite-integrated validation layer that verifies persisted results from `results/benchmark.db`. Validation must confirm:

1. The benchmark run completed successfully with no tool errors.
2. Required samples and aggregate metrics were persisted for every swept shape.
3. Metrics are finite and physically sensible (positive, within plausible bounds).
4. Measured values satisfy configured thresholds when the workload defines pass/fail gates.
5. The benchmark fails validation when required data is missing, invalid, or outside bounds.

## Non-Functional Requirements

| Requirement | Target |
|---|---|
| Automation | Runs to completion without manual intervention after `bash run_benchmark.sh` |
| Idempotency | Re-running `run_benchmark.sh` appends a new run; never corrupts existing rows |
| Persistence | All metrics survive script exit; `results/benchmark.db` is the durable record |
