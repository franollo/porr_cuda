#include <iostream>
#include <stdlib.h>
#include "load_matrix.cuh"

int load_matrix(double ***matrix, int *size, char* filename) {

	FILE* ip;
	int i, j;

	if ((ip = fopen(filename, "r")) == NULL) {
		return 1;
	}
	fscanf(ip, "%d\n\n", size);
	(*matrix) = alloc_mem(*size, (*size) + 1);
	for (i = 0; i < *size; ++i) {
		for (j = 0; j < *size; ++j)
			fscanf(ip, "%lf\t", &(*matrix)[i][j]);
		fscanf(ip, "\n");
	}
	fscanf(ip, "\n");
	for (i = 0; i < *size; ++i)
		fscanf(ip, "%lf\n", &(*matrix)[i][(*size - 1) + 1]);
	fclose(ip);
	return 0;
}


double **alloc_mem(int rows_count, int cols_count) {
	int i;
	double ** mem;
	mem = (double **)malloc(rows_count * sizeof(double*));
	for (i = 0; i < rows_count; ++i) {
		mem[i] = (double *)malloc(cols_count * sizeof(double));
	}
	return mem;
}


void dealloc_mem(double** matrix, int rows_count) {
	int i;
	for (i = 0; i < rows_count; ++i) {
		free(matrix[i]);
	}
	free(matrix);
}