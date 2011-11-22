
#define PRINT

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw.h>

#include <sys/types.h>
#include <time.h>

#define n 20
#define REP 1

#define min(a,b) (a<b?a:b)

double complex *scratch;
double complex *w;
long long int fft_step;

void prepare_fft(long long int sz){
	int i;
	double alpha;

	scratch = (double complex*) malloc(sizeof(double complex)*sz);
	w = (double complex*) malloc(sizeof(double complex)*sz);
	alpha = 2.0*M_PI/(double)sz; 

	for (i=0; i<sz; i++){
		w[i] = cos(alpha*i) - I*sin(alpha*i);
	}
	fft_step=1;
}

void free_fft(){
	free(scratch);
	free(w);
}

void fft(double complex *in, double complex *out, long long int sz){
	long long int i,j, sz2;
	double complex aa, bb;

	//printf("Caling fft, sz=%lld, fftw_step=%lld\n", sz, fft_step);
	if (sz==1){
		*out = (double complex) *in;
	} else {
		sz2 = sz>>1;
		//reorder
		memcpy(scratch, in, sz*sizeof(double complex));
		for (i=0; i<sz2; i++) {
			in[i] = scratch[2*i];
			in[i+sz2] = scratch[2*i+1];
		}
	
		fft_step = fft_step<<1;
		fft(in, out, sz2);
		fft(in+sz2, out+sz2, sz2);
		fft_step = fft_step>>1;

		//precalculate 0-element, since w[0]=1
		aa = out[0];
		bb = out[sz2];
		out[0] = aa + bb;
		out[sz2] = aa - bb;
		j=0;
		for (i=1; i<sz2; i++){
			j += fft_step;
			aa = out[i];
			bb = w[j]*out[i+sz2];
			//bb = (M_PI+M_PI*I)*out[i+sz2];
			out[i] = aa + bb;
			out[i+sz2] = aa - bb;
		}
		
	}
}

void print_vec(double *vec, int sz){
	int i;

	if (sz<10){
		printf("  [ ");
		for (i=0; i<sz; i++) printf("% .4g, ", vec[i]);
		printf("\b\b ]\n");
	} else {
		printf("  [\n");
		for (i=0; i<5; i++) printf("    % .4g,\n", vec[i]);
		printf("    ...\n");
		for (i=sz-5; i<sz-1; i++) printf("    % .4g,\n", vec[i]);
		i=sz-1;
		for (i=sz-5; i<sz; i++) printf("    % .4g\n", vec[i]);
		printf("  ]\n");
	}
}

void print_cvec(complex double *vec, int sz){
	int i;

	if (sz<=10){
		printf("  [ ");
		for (i=0; i<sz; i++) printf("% .4g + % .4g I, ", creal(vec[i]), cimag(vec[i]));
		printf("\b\b ]\n");
	} else {
		printf("  [\n");
		for (i=0; i<3; i++) printf("    % .4g + % .4g I,\n", creal(vec[i]), cimag(vec[i]));
		printf("    ...\n");
		for (i=sz-3; i<sz-1; i++) printf("    % .4g + % .4g I,\n", creal(vec[i]), cimag(vec[i]));
		i=sz-1;
		printf("    % .4g + % .4g I\n", creal(vec[i]), cimag(vec[i]));
		printf("  ]\n");
	}
}

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

	//sample input vector
	for (i=0; i<N; i++) vec_in[i] = i;

	printf("Testing recursive FFT with vector size N=%lli\n",N);
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
	printf("Output vector:\n");
	print_cvec(vec_out, N);
	#endif

	printf("\n");
	printf("Now testing fftw\n");

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

