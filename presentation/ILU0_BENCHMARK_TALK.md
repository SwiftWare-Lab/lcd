# ILU(0) dependency scheduling — presentation outline

Use one section per slide. This is a talk track for the repository benchmark, not a claim about numerical ILU(0) performance.

## Slide 1 — The question

**Can a GPU execute ILU(0)-style row dependencies faster than a sequential CPU loop?**

- We compare two GPU schedules: wavefront and sync-free.
- We report both end-to-end and prepared execution.
- Main result: the CPU wins the end-to-end comparison on these matrices; prepared sync-free GPU execution is fastest when its setup is reused.

**Say:** “The timing boundary changes the answer, so I will show exactly what each number includes.”

---

## Slide 2 — What ILU(0) means, and what we actually benchmark

- Numerical ILU(0) approximates a sparse matrix with lower and upper factors while allowing **no new stored nonzeros**.
- Computing a row can depend on results from earlier rows.
- **Our code does not compute L or U.** It uses the matrix sparsity pattern to run a synthetic row-dependency recurrence; matrix values are ignored.
- This isolates dependency scheduling and overhead, not full factorization performance.

**Visual:** Show a sparse matrix pattern next to a small dependency graph.

**Say:** “These are ILU(0)-like dependencies, not an ILU(0) factorization benchmark.”

---

## Slide 3 — From matrix entries to a dependency graph

If row 2 has stored entries in columns 0 and 1, then row 2 waits for rows 0 and 1:

    row 0 ─┐
           ├──→ row 2
    row 1 ─┘

For each row, the benchmark sums scaled outputs of earlier rows named by its CSR column indices. Every method computes the same recurrence, and GPU results are checked against the CPU reference.

**Say:** “The graph tells us which rows may execute together; the schedules decide how to enforce it.”

---

## Slide 4 — Three execution strategies

| Strategy | Processor | How a row becomes ready |
| --- | --- | --- |
| Sequential CPU | CPU | Process rows in order; earlier rows are already complete. |
| Wavefront | GPU | Precompute levels, then launch one level after another. |
| Sync-free | GPU | Maintain dependency counters; a row runs when its counter reaches zero. |

Wavefront has a barrier between levels. Sync-free has no global level barrier, but it **still synchronizes** through counters, atomics, and memory ordering.

**Visual:** Put rows 0–1 in one wavefront and row 2 in the next; alongside it, show row 2 starting as soon as both counters arrive.

---

## Slide 5 — What “end-to-end” and “prepared” include

| Reported method | Timed work | Setup excluded from timing |
| --- | --- | --- |
| CPU row dependencies | Sequential recurrence plus output-array copy/allocation | Matrix file load and CSR conversion |
| CPU prepared | Sequential recurrence using a reusable output buffer | Matrix load, CSR conversion, output allocation |
| GPU end-to-end | DAG preparation, transfers, launches, synchronization, result copy | Matrix file load and JIT compilation |
| GPU wavefront prepared | Host launch/synchronization loop over prepared levels | DAG preparation, matrix upload, result download |
| GPU sync-free prepared | Dependency-counter reset, launch, synchronization | DAG preparation, matrix upload, result download |

**Key point:** “Prepared” is not GPU-only. It means reusable setup is moved outside the timed region. The GPU prepared paths exclude more work than the CPU prepared path because the GPU has DAG and transfer setup to reuse.

**Say:** “Compare end-to-end with end-to-end for one new matrix pattern; prepared timings answer a repeated-use question.”

---

## Slide 6 — Experimental setup

- Four downloaded SuiteSparse matrices: 1138_bus, bcsstk08, bcsstk09, and bcsstk36.
- Largest case, bcsstk36: 23,052 rows and 1,143,140 stored entries.
- AMD Ryzen 9 9955HX3D CPU and NVIDIA GeForce RTX 5080 Laptop GPU.
- Five warm-up calls and 30 timed calls per method and matrix; median wall-clock time is reported.
- GPU results are validated against the CPU reference before throughput is reported.

**Say:** “These are measurements from one laptop and one benchmark implementation, not universal hardware rankings.”

---

## Slide 7 — Main results: original five-method run

Median milliseconds; **lower is better**. These five columns were measured in one complete run on 2026-09-30.

| Matrix | CPU | Wavefront E2E | Sync-free E2E | Wavefront prepared | Sync-free prepared |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1138_bus | 1.384 | 3.158 | 2.110 | 1.443 | 0.134 |
| bcsstk08 | 2.148 | 8.401 | 3.954 | 5.125 | 0.429 |
| bcsstk09 | 2.722 | 9.789 | 4.489 | 5.787 | 0.382 |
| bcsstk36 | 125.766 | 547.433 | 297.310 | 290.369 | 50.536 |

**What the table supports:**

- CPU beats both GPU end-to-end paths on all four matrices.
- Sync-free beats wavefront within both GPU timing scopes on all four matrices.
- Prepared sync-free is fastest overall, but it excludes one-time GPU setup and transfers.

**Visual:** For a slide chart, plot only bcsstk36 and group bars by timing scope. Do not put all four matrices on one linear-scale chart.

---

## Slide 8 — New CPU-prepared check

After adding a reusable CPU output buffer, we ran a **separate one-matrix check** on bcsstk36:

| CPU method | Median time |
| --- | ---: |
| Original CPU, with output copy | 126.544 ms |
| CPU prepared, reusable output | 125.644 ms |

The difference is about **0.7%**. Both paths passed correctness validation. The original CPU already excluded matrix loading and CSR conversion, so removing the output copy changes little for this large loop.

**Important:** These two numbers are from a later run. Do **not** combine them with Slide 7 to claim a controlled six-method speedup.

---

## Slide 9 — Conclusion and honest next step

1. For a new CSR matrix pattern, the measured CPU path is faster than the GPU end-to-end paths in this benchmark.
2. When GPU setup is reused, sync-free is much faster than wavefront here; the GPU prepared number is not an end-to-end speedup claim.
3. A prepared CPU path is possible and now exists, but its gain is small because CPU setup was already mostly outside the timing loop.
4. None of these numbers measures numerical ILU(0) factorization.

**Next experiment:** Run all six methods in one session, measure DAG construction and transfers separately, add a compiled CPU baseline, then benchmark actual numerical ILU(0).

**Closing line:** “Our result is about how dependency scheduling and setup costs change the CPU–GPU tradeoff.”

---

## Source files and reproducibility

- Benchmark implementation: [benchmark/ilu0_parallel_benchmark.py](../benchmark/ilu0_parallel_benchmark.py)
- Detailed methodology and caveats: [benchmark/ILU0_PARALLEL_BENCHMARK.md](../benchmark/ILU0_PARALLEL_BENCHMARK.md)
- Original five-method results: [benchmark/results/ilu0-row-dependencies-validated.csv](../benchmark/results/ilu0-row-dependencies-validated.csv)

To regenerate a six-method report from the project root:

    ./.venv/bin/python benchmark/ilu0_parallel_benchmark.py

The command overwrites the saved report CSV and plot; run it when you are ready to replace the original five-method artifacts.
