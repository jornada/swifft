
#define PRINT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#include <time.h>
#include "fft.h"
#include "wavelets.h"
#include "utils.h"

//#define n 3
//#define REP 1
#define n 23
#define REP 1

int main(int argc, char **argv){
	long long int N, i;
	double complex *vec_in;
	double complex *vec_tmp;
	double complex *vec_ans1;
	double complex *vec_ans2;
	double *h,*g, diff;
	struct timespec ts0, ts1;
	clock_t c0, c1;

	N = pow(2, n);
	vec_in  = (double complex*) malloc(sizeof(double complex)*N);
	vec_tmp = (double complex*) malloc(sizeof(double complex)*N);
	vec_ans1 = (double complex*) malloc(sizeof(double complex)*N);
	vec_ans2 = (double complex*) malloc(sizeof(double complex)*N);

	printf("\n");
	printf("Testing FAFT vector size N=%lli\n",N);

	//sample input vector
	for (i=0; i<N; i++) vec_in[i] = sin(i*M_PI/N*1000);
	vec_in[N>>1]=-100;
	//for (i=0; i<N; i++) vec_in[i] = sin(i*M_PI/N*10);
	//for (i=0; i<N; i++) vec_in[i] = i;
	prepare_fft(N);
	
	printf("\n");
	printf("Testing exact recursive FFT\n");

	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();
	fft3(vec_in, vec_ans1, N);
	for (i=1; i<REP; i++) fft3(vec_in, vec_tmp, N);
	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);

	free_fft();
	printf("Answer:\n");
	print_cvec(vec_ans1, N);

	printf("\n");
	printf("Testing recursive FAFT\n");

	//sample input vector
	for (i=0; i<N; i++) vec_in[i] = sin(i*M_PI/N*1000);
	vec_in[N>>1]=-100;
	//for (i=0; i<N; i++) vec_in[i] = sin(i*M_PI/N*10);
	//for (i=0; i<N; i++) vec_in[i] = i;
	
	//use lazy wavelet filter;
	h = (double*) calloc(N, sizeof(double));
	g = (double*) calloc(N, sizeof(double));
	//h[0]=1; g[1]=1;
	g[0]=sqrt(2.)*0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_faft(N, h, 2, g, 2);

	//h[0]=-0.12940952255092145;h[1]=-0.22414386804185735;h[2]=0.83651630373746899;h[3]=-0.48296291314469025;
	//g[0]=0.48296291314469025;g[1]=0.83651630373746899;g[2]=0.22414386804185735;g[3]=-0.12940952255092145;
	//prepare_faft(N, h, 4, g, 4);

	//printf("Input vector:\n");
	print_cvec(vec_in, N);

	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();
	faft(vec_in, vec_ans2, N);
	diff = diff_norm(vec_ans2, vec_ans1, N);
	for (i=1; i<REP; i++) faft(vec_in, vec_tmp, N);
	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	
	free_faft();
	
	printf("Answer:\n");
	print_cvec(vec_ans2, N);

	//printf("Input vector (after reordering):\n");
	//print_cvec(vec_in, N);

	#ifdef PRINT
	printf("Error (2-norm): %lf\n", diff);
	//printf(vec_out, N);
	#endif

	return 0;
}

