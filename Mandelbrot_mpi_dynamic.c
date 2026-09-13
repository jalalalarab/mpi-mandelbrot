#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

#define MASTER 0
#define WORK_TAG 1
#define STOP_TAG 2

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

    if (size == 1) {
        if (rank == MASTER) {
            fprintf(stderr, "Run with at least 2 processes for dynamic scheduling.\n");
        }
        MPI_Finalize();
        return 0;
    }

    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    if (rank == MASTER) {
        int image[HEIGHT][WIDTH];
        int next_row = 0;
        int rows_completed = 0;
        MPI_Status status;

        int *recv_buffer = (int *)malloc((WIDTH + 1) * sizeof(int));
        int active_workers = (size - 1 < HEIGHT) ? (size - 1) : HEIGHT;

        for (int worker = 1; worker <= active_workers; worker++) {
            int row = next_row++;
            MPI_Send(&row, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
        }

        while (rows_completed < HEIGHT) {
            MPI_Recv(recv_buffer, WIDTH + 1, MPI_INT, MPI_ANY_SOURCE,
                     WORK_TAG, MPI_COMM_WORLD, &status);

            int worker = status.MPI_SOURCE;
            int row = recv_buffer[0];

            for (int j = 0; j < WIDTH; j++) {
                image[row][j] = recv_buffer[j + 1];
            }

            rows_completed++;

            if (next_row < HEIGHT) {
                int new_row = next_row++;
                MPI_Send(&new_row, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
            } else {
                int stop_row = -1;
                MPI_Send(&stop_row, 1, MPI_INT, worker, STOP_TAG, MPI_COMM_WORLD);
            }
        }

        free(recv_buffer);

        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();

        save_pgm("mandelbrot_dynamic.pgm", image);
        double elapsed = end_time - start_time;
        printf("Dynamic MPI Mandelbrot: %d processes, time = %f seconds\n", size, elapsed);

    } else {
        MPI_Status status;
        struct complex c;
        int *send_buffer = (int *)malloc((WIDTH + 1) * sizeof(int));

        while (1) {
            int row;
            MPI_Recv(&row, 1, MPI_INT, MASTER, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == STOP_TAG || row < 0) {
                break;
            }

            send_buffer[0] = row;
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (row - HEIGHT / 2.0) * 4.0 / HEIGHT;
                send_buffer[j + 1] = cal_pixel(c);
            }

            MPI_Send(send_buffer, WIDTH + 1, MPI_INT, MASTER, WORK_TAG, MPI_COMM_WORLD);
        }

        free(send_buffer);
        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();
    }

    MPI_Finalize();
    return 0;
}
