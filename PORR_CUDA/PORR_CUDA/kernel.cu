#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <math.h>

#define Im1 0               // i - 1
#define I 1                 // i
#define Ip1 2               // i + 1

using namespace std;

//Multiplies NxN matrix with N vector
__global__ void multMatrixVectorKernel(double *matrix, double *vector, double *res_vector, int N) {

	int ROW = blockIdx.x*blockDim.x + threadIdx.x;

	double sum = 0;

	if (ROW < N) {
		for (int i = 0; i < N; i++) {
			sum += matrix[ROW * N + i] * vector[i];
		}
		res_vector[ROW] = sum;
	}
}


__global__ void copyVectorToMatRowKernel(double *src, double *dst, int ROW_NUM, int N) {

	int ROW = blockIdx.x*blockDim.x + threadIdx.x;

	if (ROW < N) {
		dst[ROW_NUM * N + ROW] = src[ROW];
	}
}

__global__ void copyMatRowToVectorKernel(double *src, double *dst, int ROW_NUM, int N) {

	int ROW = blockIdx.x*blockDim.x + threadIdx.x;

	if (ROW < N) {
		dst[ROW] = src[ROW_NUM * N + ROW];
	}
}

void copyMatRowToVector(double *src, double *dst, int ROW_NUM, int N) {
	dim3 threadsPerBlock(N);
	dim3 blocksPerGrid(1);
	if (N > 512) {
		threadsPerBlock.x = 512;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
	}

	copyMatRowToVectorKernel << <blocksPerGrid, threadsPerBlock >> >(src, dst, ROW_NUM, N);
}

__global__ void calculateXplus1Kernel(double *x, double *Ax, double *b, double *scalar_1, double *scalar_2, int N) {

	int ROW = blockIdx.x*blockDim.x + threadIdx.x;

	//x[Ip1][i] = x[I][i] + w[I] * w[Im1] * (x[I][i] - x[Im1][i]) - Axb[i];

	if (ROW < N) {
		x[Ip1 * N + ROW] = x[I * N + ROW] + *scalar_1 * (x[I * N + ROW] - x[Im1 * N + ROW]) - *scalar_2 * (Ax[ROW] - b[ROW]);
		x[Im1 * N + ROW] = x[I * N + ROW];
		x[I * N + ROW] = x[Ip1 * N + ROW];
	}
}



__global__ void normKernel(double *x, double *res_vector, int N) {

	int ROW = blockIdx.x*blockDim.x + threadIdx.x;

	if (ROW < N) {
		res_vector[ROW] = x[I * N + ROW] - x[Im1 * N + ROW];
		res_vector[ROW] = pow(res_vector[ROW], 2);
	}
}




