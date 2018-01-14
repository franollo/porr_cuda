#ifndef KERNEL_CUH_
#define KERNEL_CUH_

void matrixMultiplication(float *A, float *B, float *C, int N);

//Multiplies NxN matrix with N vector
void multMatrixVector(double *matrix, double *vector, double *res_vector, int N);

//Adds two N vectors
void addVectors(double *vector1, double *vector2, double *res_vector, int N);

//Adds two N vectors
void subtrackVectors(double *vector1, double *vector2, double *res_vector, int N);

//Multiples N vector with scalar
void multScalarVector(double scalar, double *vector, double *res_vector, int N);
#endif