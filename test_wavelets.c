
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw.h>

#include <time.h>
#include "utils.h"
#include "fft.h"
#include "wavelets.h"

#define n 12
#define REP 50
//#define n 23
//#define REP 1
#define OUTPUT
//#define PRINT
#define SIGNAL 2

void under_sample(double complex *vec, int sz){
	int i, sz2;
	
	sz2 = sz>>1;
	for (i=0; i<sz2; i++){
		vec[i] = (vec[i<<1]+vec[(i<<1)+1])*0.5;
	}
}

void over_sample(double complex *vec, int sz){
	int i, sz2;
	
	sz2 = sz>>1;
	for (i=sz2-1; i>=0; i--){
		vec[i<<1] = vec[i];
		vec[(i<<1)+1] = vec[i];
	}
}

void ifft(double complex *vec_in, double complex *vec_out, int sz){
	fftw_plan p;
	int i;

	p = fftw_create_plan(sz, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_out);
	fftw_destroy_plan(p);
	for (i=0; i<sz; i++) vec_out[i] /= sz;
}

int main(int argc, char **argv){
	int N, i;
	double complex *vec_in;
	double complex *vec_tmp;
	double complex *vec_fftw, *vec_us1, *vec_ffft, *vec_haar;
	double *h,*g, diff;
	struct timespec ts0, ts1;
	clock_t c0, c1;
	fftw_plan p;

	N = pow(2, n);
	#define ALLOC(x) x = (double complex*) malloc(sizeof(double complex)*N);
	ALLOC(vec_in);
	ALLOC(vec_tmp);
	ALLOC(vec_fftw);
	ALLOC(vec_us1);
	ALLOC(vec_ffft);
	ALLOC(vec_haar);

	printf("\n");
	printf("SWIFFT - BENCHMARK\n");
	printf("------------------\n");
	printf("  signal size: N = %i\n", N);
	printf("  signal type: kind = %i\n", (int) SIGNAL);
	printf("\n");


	/************************
	*         FFTW          *
	************************/

	printf("\nTesting FFTW\n");

	create_signal(vec_in, N, SIGNAL);
	write_cvec(vec_in, N, "orig.dat");

	p = fftw_create_plan(N, FFTW_FORWARD, FFTW_ESTIMATE);
	//p = fftw_create_plan(N, FFTW_FORWARD, FFTW_MEASURE);
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_fftw);
	for (i=1; i<REP; i++)
		fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	fftw_destroy_plan(p);

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_fftw, N);
	#endif
	write_cvec(vec_fftw, N, "freq_fftw.dat");
	ifft(vec_fftw, vec_tmp, N);
	write_cvec(vec_tmp, N, "inv_fftw.dat");

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);



	/*************************
	*   Under-sampled FFTW   *
	**************************/

	printf("\nTesting Under-sampled FFTW\n");

	create_signal(vec_in, N, SIGNAL);
	under_sample(vec_in, N);


	p = fftw_create_plan(N>>1, FFTW_FORWARD, FFTW_ESTIMATE);
	//p = fftw_create_plan(N, FFTW_FORWARD, FFTW_MEASURE);
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	under_sample(vec_tmp, N); //just for timing purposes
	fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_us1);
	over_sample(vec_tmp, N); //just for timing purposes
	for (i=1; i<REP; i++) {
		under_sample(vec_tmp, N); //just for timing purposes
		fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_tmp);
		over_sample(vec_tmp, N); //just for timing purposes
	}

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	fftw_destroy_plan(p);

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_tmp, N);
	#endif
	ifft(vec_us1, vec_tmp, N>>1);
	over_sample(vec_tmp, N);
	write_cvec(vec_us1, N>>1, "freq_us1.dat"); //note: this FFT contains N-1 freqs!
	write_cvec(vec_tmp, N, "inv_us1.dat");

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);



	/********************************************
	*   Felipe's FFT  (bit-reversed in-place)   *
	********************************************/

	printf("\nTesting FFFT\n");

	create_signal(vec_in, N, SIGNAL);
	prepare_fft(N);
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	fft3(vec_in, vec_ffft, N);
	for (i=1; i<REP; i++)
		fft3(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_fft();



	/******************************
	*    SWIFFT - Haar Wavelet    *
	*******************************/

	printf("\nTesting SWIFFT - Haar Wavelet\n");

	create_signal(vec_in, N, SIGNAL);
	
	//use lazy wavelet filter;
	h = (double*) calloc(N, sizeof(double));
	g = (double*) calloc(N, sizeof(double));
	//h[0]=1; g[1]=1;
	g[0]=sqrt(2.)*0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_swifft(N, h, 2, g, 2);

	//h[0]=-0.12940952255092145;h[1]=-0.22414386804185735;h[2]=0.83651630373746899;h[3]=-0.48296291314469025;
	//g[0]=0.48296291314469025;g[1]=0.83651630373746899;g[2]=0.22414386804185735;g[3]=-0.12940952255092145;
	//prepare_swifft(N, h, 4, g, 4);

	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_haar(vec_in, vec_haar, N);
	for (i=1; i<REP; i++)
		swifft_haar(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_haar, N);
	#endif
	write_cvec(vec_haar, N, "freq_haar.dat");
	ifft(vec_haar, vec_tmp, N);
	write_cvec(vec_tmp, N, "inv_haar.dat");

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);

	return 0;
}

