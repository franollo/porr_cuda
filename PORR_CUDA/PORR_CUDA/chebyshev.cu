#include <math.h>
#include <iostream>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "chebyshev.cuh"
#include "kernel.cu"
#include <string>
#include <stdlib.h>
#include <math.h>

#define Im1 0               // i - 1
#define I 1                 // i
#define Ip1 2               // i + 1


#define THREADS_PER_BLOCK 512


using namespace std;

double* chebyshev(int matrix_size, double** Ab, int s, int max_iter) {
	int N = matrix_size;
	dim3 threadsPerBlock(matrix_size);
	dim3 blocksPerGrid(1);
	if (matrix_size > 512) {
		threadsPerBlock.x = 512;
		blocksPerGrid.x = ceil(double(matrix_size) / double(threadsPerBlock.x));
	}

	double delta, x_2_norm, a, w_0, c, L, B, scalar_1, scalar_2;
	double *x_start, *A_vector, *b_vector, *w, *norm_vector;										//HOST	
	double *d_x, *d_scalar_1, *d_scalar_2, *d_temp_vector, *d_x_start, *d_A_vector, *d_b_vector;    //DEVICE
	int iteration, k;

	delta = 0.000001;       // accuracy
	x_2_norm = 0;           // second norm for stop criteria
	a = 100;                // alpha
	bool stop = false;      // stop criteria bool

	//ALLOCATE MEMORY ON HOST
	x_start = (double *)malloc(matrix_size * sizeof(double));
	A_vector = (double *)malloc(matrix_size * matrix_size * sizeof(double));
	b_vector = (double *)malloc(matrix_size * sizeof(double));
	w = (double *)malloc(matrix_size * 2 * sizeof(double));
	norm_vector = (double *)malloc(matrix_size * sizeof(double));

	//ALLOCATE MEMORY ON DEVICE
	cudaMalloc((void **)&d_x, matrix_size * 3 * sizeof(double));
	cudaMalloc((void **)&d_scalar_1, sizeof(double));
	cudaMalloc((void **)&d_scalar_2, sizeof(double));
	cudaMalloc((void **)&d_temp_vector, matrix_size * sizeof(double));
	cudaMalloc((void **)&d_x_start, matrix_size * sizeof(double));
	cudaMalloc((void **)&d_A_vector, matrix_size * matrix_size * sizeof(double));
	cudaMalloc((void **)&d_b_vector, matrix_size * sizeof(double));


	B = Ab[0][0];

	//init x_start, find Beta, copy Ab to vectors
	for (int i = 0; i < matrix_size; i++) {
		x_start[i] = 0;
		b_vector[i] = Ab[i][matrix_size];
		if (Ab[i][i] > B) {
			B = Ab[i][i];
		}
		for (int j = 0; j < matrix_size; j++) {
			A_vector[i * N + j] = Ab[i][j];
		}
	}

	B = 2 * B;

	//COPY FROM HOST TO DEVICE
	cudaMemcpy(d_x_start, x_start, matrix_size * sizeof(double), cudaMemcpyHostToDevice);	cudaMemcpy(d_A_vector, A_vector, matrix_size * matrix_size * sizeof(double), cudaMemcpyHostToDevice);	cudaMemcpy(d_b_vector, b_vector, matrix_size * sizeof(double), cudaMemcpyHostToDevice);

	//Step 0:
	iteration = 0;
	w_0 = (B - a) / (B + a);
	c = 2 / (B + a);
	L = 2 * (B + a) / (B - a);

	while (iteration < max_iter && stop == false) {
		//Step 1
		k = 0;

		copyVectorToMatRowKernel <<<blocksPerGrid, threadsPerBlock >>>(d_x_start, d_x, I, N);

		w[Im1] = 0;
		w[I] = w_0;

		while (iteration < max_iter) {
			//Step 2
			scalar_1 = c * (1 + w[I] * w[Im1]);
			scalar_2 = w[I] * w[Im1];
			cudaMemcpy(d_scalar_1, &scalar_1, sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_scalar_2, &scalar_2, sizeof(double), cudaMemcpyHostToDevice);

			//copies x(i) to temp_vector
			copyMatRowToVectorKernel <<<blocksPerGrid, threadsPerBlock>>>(d_x, d_temp_vector, I, N);

			//multiples A matrix in vector form with x(i) stored in temp_vector
			multMatrixVectorKernel <<<blocksPerGrid, threadsPerBlock>>>(d_A_vector, d_temp_vector, d_temp_vector, N);

			//calculates x(i+1) and sets x(i-1) and x(i)
			calculateXplus1Kernel <<<blocksPerGrid, threadsPerBlock>>>(d_x, d_temp_vector, d_b_vector, d_scalar_1, d_scalar_2, N);

			w[Im1] = w[I];
			w[I] = 1 / L - w[I];
			
			x_2_norm = 0;

			normKernel <<<blocksPerGrid, threadsPerBlock >> >(d_x, d_temp_vector, N);

			cudaMemcpy(norm_vector, d_temp_vector, N * sizeof(double), cudaMemcpyDeviceToHost);

			for (int i = 0; i < N; i++) {
				x_2_norm += norm_vector[i];
			}
			x_2_norm = sqrt(x_2_norm);
			if (x_2_norm < delta) {
				stop = true;
				break;
			}

			// Step 3
			iteration++;
			k++;
			if (k >= s) {
				copyMatRowToVectorKernel << <blocksPerGrid, threadsPerBlock >> >(d_x, d_x_start, I, N);
				break;
			}
		}
	}
	copyMatRowToVectorKernel << <blocksPerGrid, threadsPerBlock >> >(d_x, d_temp_vector, I, N);
	cudaMemcpy(x_start, d_temp_vector, N * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_scalar_1);
	cudaFree(d_scalar_2);
	cudaFree(d_temp_vector);
	cudaFree(d_x_start);
	cudaFree(d_A_vector);
	cudaFree(d_b_vector);

	free(A_vector);
	free(b_vector);
	free(w);
	free(norm_vector);
	return x_start;
}

void matrix_test() {

}



