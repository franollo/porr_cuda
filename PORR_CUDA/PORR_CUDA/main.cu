#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "chebyshev.cu"
#include "load_matrix.cuh"
#include <sys/timeb.h>
#include <math.h>

using namespace std;


void test(int liczba_prob, char* nazwa_pliku);

int main()
{
	test(1, "data_1000");
	return 0;
}


void test(int liczba_prob, char* nazwa_pliku) {
	printf("gauss-jordan;cheb s10; cheb s100; cheb s1000; cheb s10000;liczba_watkow \n");
	struct timeb start_gauss_jordan, start_cheb_1, start_cheb_2, start_cheb_3, start_cheb_4, end_gauss_jordan, end_cheb_1, end_cheb_2, end_cheb_3, end_cheb_4;
	for (int i = 0; i < liczba_prob; i++) {
		double ** matrix;
		int matrix_size;

		load_matrix(&matrix, &matrix_size, nazwa_pliku);
		ftime(&start_gauss_jordan);
		//uss_jordan(matrix_size, matrix, thread_cnt);
		ftime(&end_gauss_jordan);
		dealloc_mem(matrix, matrix_size);

		load_matrix(&matrix, &matrix_size, nazwa_pliku);
		ftime(&start_cheb_1);
		chebyshev(matrix_size, matrix, 10, 1000);
		ftime(&end_cheb_1);
		dealloc_mem(matrix, matrix_size);

		load_matrix(&matrix, &matrix_size, nazwa_pliku);
		ftime(&start_cheb_2);
		chebyshev(matrix_size, matrix, 100, 1000);
		ftime(&end_cheb_2);
		dealloc_mem(matrix, matrix_size);

		load_matrix(&matrix, &matrix_size, nazwa_pliku);
		ftime(&start_cheb_3);
		chebyshev(matrix_size, matrix, 1000, 1000);
		ftime(&end_cheb_3);
		dealloc_mem(matrix, matrix_size);

		load_matrix(&matrix, &matrix_size, nazwa_pliku);
		ftime(&start_cheb_4);
		chebyshev(matrix_size, matrix, 10000, 1000);
		ftime(&end_cheb_4);
		dealloc_mem(matrix, matrix_size);




		int diff_milis_gauss_jord = (int)(1000.0 * (end_gauss_jordan.time - start_gauss_jordan.time)
			+ (end_gauss_jordan.millitm - start_gauss_jordan.millitm));
		int diff_milis_cheb_1 = (int)(1000.0 * (end_cheb_1.time - start_cheb_1.time)
			+ (end_cheb_1.millitm - start_cheb_1.millitm));
		int diff_milis_cheb_2 = (int)(1000.0 * (end_cheb_2.time - start_cheb_2.time)
			+ (end_cheb_2.millitm - start_cheb_2.millitm));
		int diff_milis_cheb_3 = (int)(1000.0 * (end_cheb_3.time - start_cheb_3.time)
			+ (end_cheb_3.millitm - start_cheb_3.millitm));
		int diff_milis_cheb_4 = (int)(1000.0 * (end_cheb_4.time - start_cheb_4.time)
			+ (end_cheb_4.millitm - start_cheb_4.millitm));
		printf("%d;%d;%d;%d;%d\n", diff_milis_gauss_jord, diff_milis_cheb_1, diff_milis_cheb_2, diff_milis_cheb_3, diff_milis_cheb_4);
	}
}

