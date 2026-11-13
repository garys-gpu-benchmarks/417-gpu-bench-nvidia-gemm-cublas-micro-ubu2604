# Results: 417 GEMM / cuBLAS Microbenchmark

**Main goal:** Measure GEMM throughput across dtypes/shapes  
**Tools:** Compile cublas_gemm.cu with nvcc; run bin/cublas_gemm twice via collect_cublas_gemm.py

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_138 · recorded 2026-11-12). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 80GB HBM3 (81559 MiB) · Intel(R) Xeon(R) Platinum 8470 · Ubuntu 26.04.1 LTS · kernel 6.8.0-147-generic · driver 595.91.07
- **Run:** 2026-11-12T23:07:54Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 00:37 · benchmark 00:03 · whole run 00:47
- **Source:** run 417_20261112_230754
- **Command:** `bash run_benchmark.sh --smoke --dtype f32_r --M 64 --N 64 --K 64 --transposeA N --transposeB N --alpha 1 --beta 0 --lda 64 --ldb 64 --ldc 64 --batch-count 1 --warmup-iters 2 --num-iterations 2 --norm-check 1`

| # | Metric | Result |
|---|---|---|
| 1 | Achieved compute throughput, TFLOP/s (achieved_compute_tflops) | 0.039418499999999995 |
| 2 | Kernel execution time (kernel_time_msec) | 0.013544 |
| 3 | Minimum operand traffic, GB/s (minimum_operand_traffic_gb_s) | 3.2 |
| 4 | Relative L1 of checked columns (relative_l1_checked_columns) | 2.59177062e-08 |
| 5 | Relative L2 error of checked columns (norm_error_2) | 5.07249905e-08 |
<!-- /profile:smoke -->
