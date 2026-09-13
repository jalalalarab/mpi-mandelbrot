# MPI Mandelbrot

Three implementations of Mandelbrot set rendering, comparing a
**sequential baseline** against **two parallel strategies** — static
partitioning and dynamic master-worker — using **MPI**.

The point of the project is not the fractal; it is the comparison.
Mandelbrot is a classic **load-imbalance** problem: points inside the
set require the full iteration budget, points outside escape quickly.
Static partitioning distributes the image evenly across processes and
pays a tail-latency cost; a dynamic master-worker scheduler assigns
small work units on demand and stays busy. This project quantifies
that difference.

## The three implementations

| File | Strategy | Where the imbalance goes |
|---|---|---|
| `Mandelbrotseq.c` | Single process computes the whole image row by row. | Baseline for speedup measurements. |
| `Mandelbrot_mpi_static.c` | Image rows split into equal contiguous stripes, one per process. Each process computes its stripe and sends it back to root. | Processes assigned stripes with more "inside" points finish last — the whole run waits for the slowest stripe. |
| `Mandelbrot_mpi_dynamic.c` | Master process holds a queue of small row-groups. Workers request the next group when they finish the previous one. | Load balances automatically; slower work is absorbed by faster workers. |

## What the code shows

- Correct use of **`MPI_Init` / `MPI_Finalize`**, and rank/size handling.
- **Point-to-point communication** (`MPI_Send` / `MPI_Recv`) for the
  master-worker version.
- **Gather / collective** communication for the static version.
- Careful boundary handling so no pixel is computed twice and none is
  missed at process boundaries.
- Rendering the escape-iteration count as a colour, and writing the
  output image.

## Build & run

Requires an MPI implementation (Open MPI or MPICH) and a C compiler.

```bash
# Sequential (plain gcc, no MPI needed)
gcc Mandelbrotseq.c -o mandelbrot_seq -lm
./mandelbrot_seq

# Static — MPI, 4 processes
mpicc Mandelbrot_mpi_static.c -o mandelbrot_static -lm
mpirun -np 4 ./mandelbrot_static

# Dynamic master-worker — MPI, 8 processes (1 master + 7 workers)
mpicc Mandelbrot_mpi_dynamic.c -o mandelbrot_dynamic -lm
mpirun -np 8 ./mandelbrot_dynamic
```

## Results (headline)

The dynamic scheduler consistently outperforms static partitioning on
Mandelbrot precisely because the workload is unbalanced: static leaves
processes idle while one finishes its stripe of "inside" points, while
dynamic keeps every worker busy until the last work unit is drawn.

## Context

Course project for LAU's parallel-computing course. Written in C,
built with `mpicc`, run under `mpirun`. The focus is the comparison
and the discussion of *why* dynamic wins on load-imbalanced problems,
not the Mandelbrot mathematics.
