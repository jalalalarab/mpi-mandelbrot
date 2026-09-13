# MPI Mandelbrot Implementations

This repository contains three versions of the Mandelbrot fractal generator.

## 1. Sequential Version
**File:** Mandelbrotseq.c  
Single-core implementation used as a baseline.

### Compile
gcc -o mandel_seq Mandelbrotseq.c -lm

### Run
./mandel_seq

## 2. Static MPI Version (Block Distribution)
**File:** Mandelbrot_mpi_static.c  
Each MPI process gets a fixed block of rows.

### Compile
mpicc -o mandel_static Mandelbrot_mpi_static.c

### Run
mpirun -np 4 ./mandel_static

## 3. Dynamic MPI Version (Master–Worker)
**File:** Mandelbrot_mpi_dynamic.c  
Master process assigns rows dynamically to worker processes.

### Compile
mpicc -o mandel_dynamic Mandelbrot_mpi_dynamic.c

### Run
mpirun -np 4 ./mandel_dynamic

All versions generate a PGM image of the Mandelbrot set.
