#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;

    double z_real2, z_imag2, lengthsq;
    int iter = 0;

    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;

        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE *pgmimg = fopen(filename, "wb");
    if (!pgmimg) {
        perror("fopen");
        return;
    }

    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }

    fclose(pgmimg);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;

    int base_rows = HEIGHT / size;
    int remainder = HEIGHT % size;

    int local_rows;
    int start_row;

    if (rank < remainder) {
        local_rows = base_rows + 1;
        start_row = rank * local_rows;
    } else {
        local_rows = base_rows;
        start_row = remainder * (base_rows + 1) + (rank - remainder) * base_rows;
    }

    int *local_image = (int *)malloc(local_rows * WIDTH * sizeof(int));
    if (!local_image) {
        fprintf(stderr, "Rank %d: Error allocating local_image\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    struct complex c;
    for (int i = 0; i < local_rows; i++) {
        int global_i = start_row + i;
        for (int j = 0; j < WIDTH; j++) {
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (global_i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            local_image[i * WIDTH + j] = cal_pixel(c);
        }
    }

    int *recv_counts = NULL;
    int *displs = NULL;
    int image[HEIGHT][WIDTH];

    if (rank == 0) {
        recv_counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int r = 0; r < size; r++) {
            int r_base = HEIGHT / size;
            int r_rem = HEIGHT % size;
            int r_local_rows = (r < r_rem) ? (r_base + 1) : r_base;
            recv_counts[r] = r_local_rows * WIDTH;
            displs[r] = offset;
            offset += recv_counts[r];
        }
    }

    MPI_Gatherv(local_image, local_rows * WIDTH, MPI_INT,
                (rank == 0) ? &image[0][0] : NULL,
                recv_counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        save_pgm("mandelbrot_static.pgm", image);
        double elapsed = end_time - start_time;
        printf("Static MPI Mandelbrot: %d processes, time = %f seconds\n", size, elapsed);
        free(recv_counts);
        free(displs);
    }

    free(local_image);
    MPI_Finalize();
    return 0;
}
