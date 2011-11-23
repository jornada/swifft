
#define PRINT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw.h>

#include <sys/types.h>
#include <time.h>
#include "fft.h"
#include "utils.h"

#define n 20
#define REP 1

int main(int argc, char **argv){
	long long int N, i;
	double complex *vec_in;
	double complex *vec_out;
	fftw_plan p;

	time_t  t0, t1; /* time_t is defined on <time.h> and <sys/types.h> as long */
	clock_t c0, c1; /* clock_t is defined on <time.h> and <sys/types.h> as int */

	N = pow(2, n);
	vec_in  = (double complex*) malloc(sizeof(double complex)*N);
	vec_out = (double complex*) malloc(sizeof(double complex)*N);

	printf("Testing FFTm vector size N=%lli\n",N);
	printf("\n");

	printf("Testing recursive FFT\n");

	//sample input vector
	for (i=0; i<N; i++) vec_in[i] = i;
	//printf("Input vector:\n");
	//print_cvec(vec_in, N);

	prepare_fft(N);	

	t0 = time(NULL); c0 = clock();
	for (i=0; i<REP; i++)
		fft(vec_in, vec_out, N);
	t1 = time(NULL); c1 = clock();
	printf("\tTime (WALL): %ld\tTime (clock): %f\n", (long int) (t1-t0), (float)(c1-c0)/CLOCKS_PER_SEC);
	
	free_fft();	

	//printf("Input vector (after reordering):\n");
	//print_cvec(vec_in, N);

	#ifdef PRINT
	printf("Result:\n");
	print_cvec(vec_out, N);
	#endif

	printf("Testing out-of-place recursive FFT\n");

	//sample input vector
	for (i=0; i<N; i++) vec_in[i] = i;
	//printf("Input vector:\n");
	//print_cvec(vec_in, N);

	prepare_fft(N);	

	t0 = time(NULL); c0 = clock();
	for (i=0; i<REP; i++)
		fft2(vec_in, vec_out, N);
	t1 = time(NULL); c1 = clock();
	printf("\tTime (WALL): %ld\tTime (clock): %f\n", (long int) (t1-t0), (float)(c1-c0)/CLOCKS_PER_SEC);
	
	free_fft();	

	//printf("Input vector (after reordering):\n");
	//print_cvec(vec_in, N);

	#ifdef PRINT
	printf("Result:\n");
	print_cvec(vec_out, N);
	#endif
	
	printf("Testing bit-reversed recursive FFT\n");

	//sample input vector
	for (i=0; i<N; i++) vec_in[i] = i;
	//printf("Input vector:\n");
	//print_cvec(vec_in, N);

	prepare_fft(N);	

	t0 = time(NULL); c0 = clock();
	for (i=0; i<REP; i++)
		fft3(vec_in, vec_out, N);
	t1 = time(NULL); c1 = clock();
	printf("\tTime (WALL): %ld\tTime (clock): %f\n", (long int) (t1-t0), (float)(c1-c0)/CLOCKS_PER_SEC);
	
	free_fft();	

	//printf("Input vector (after reordering):\n");
	//print_cvec(vec_in, N);

	#ifdef PRINT
	printf("Result:\n");
	print_cvec(vec_out, N);
	#endif

	printf("\n");
	printf("Testing fftw\n");

	//sample input vector
	for (i=0; i<N; i++) vec_in[i] = i;

	//printf("Input vector:\n");
	//print_cvec(vec_in, N);

	p = fftw_create_plan(N, FFTW_FORWARD, FFTW_ESTIMATE);

	t0 = time(NULL); c0 = clock();
	for (i=0; i<REP; i++)
	fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_out);
	t1 = time(NULL); c1 = clock();
	printf("\tTime (WALL): %ld\tTime (clock): %f\n", (long int) (t1-t0), (float)(c1-c0)/(float)CLOCKS_PER_SEC);

	fftw_destroy_plan(p);

	#ifdef PRINT
	printf("Output vector:\n");
	print_cvec(vec_out, N);
	#endif

	return 0;
}

