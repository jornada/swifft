
#define PRINT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#include <sys/types.h>
#include <time.h>
#include "wavelets.h"
#include "utils.h"

#define n 3
#define REP 1

int main(int argc, char **argv){
	long long int N, i;
	double complex *vec_in;
	double complex *vec_out;
	double *h,*g;

	time_t  t0, t1; /* time_t is defined on <time.h> and <sys/types.h> as long */
	clock_t c0, c1; /* clock_t is defined on <time.h> and <sys/types.h> as int */

	N = pow(2, n);
	vec_in  = (double complex*) malloc(sizeof(double complex)*N);
	vec_out = (double complex*) malloc(sizeof(double complex)*N);

	printf("Testing FAFT vector size N=%lli\n",N);
	printf("\n");

	printf("Testing recursive FAFT\n");

	//sample input vector
	for (i=0; i<N; i++) vec_in[i] = i;
	
	//use lazy wavelet filter;
	h = (double*) calloc(N, sizeof(double));
	g = (double*) calloc(N, sizeof(double));
	//h[0]=1; g[1]=1;
	//h[0]=sqrt(2.)*0.5; h[1]=h[0];
	//g[0]=h[0];g[1]=-g[0];
	//prepare_faft(N, h, 2, g, 2);

	h[0]=-0.12940952255092145;h[1]=-0.22414386804185735;h[2]=0.83651630373746899;h[3]=-0.48296291314469025;
	g[0]=0.48296291314469025;g[1]=0.83651630373746899;g[2]=0.22414386804185735;g[3]=-0.12940952255092145;
	prepare_faft(N, h, 4, g, 4);

	//printf("Input vector:\n");
	//print_cvec(vec_in, N);



	t0 = time(NULL); c0 = clock();
	for (i=0; i<REP; i++)
		faft(vec_in, vec_out, N);
	t1 = time(NULL); c1 = clock();
	printf("\tTime (WALL): %ld\tTime (clock): %f\n", (long int) (t1-t0), (float)(c1-c0)/CLOCKS_PER_SEC);
	
	free_faft();	

	//printf("Input vector (after reordering):\n");
	//print_cvec(vec_in, N);

	#ifdef PRINT
	printf("Result:\n");
	print_cvec(vec_out, N);
	#endif

	return 0;
}

