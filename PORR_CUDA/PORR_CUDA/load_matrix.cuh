#ifndef LOAD_MATRIX_CUH_
#define LOAD_MATRIX_CUH_

int load_matrix(double ***matrix, int *size, char* filename);

void dealloc_mem(double** matrix, int rows_count);

double** alloc_mem(int rows_count, int cols_count);

#endif