
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw.h>

#include <time.h>
#include "utils.h"
#include "fft.h"
#include "swifft.h"

//#define n 8
//#define REP 10000
//#define REP 1

#define n 15
#define REP 200
//#define REP 1

//#define n 20
//#define REP 5

//#define n 24
//#define REP 1

//#define n 23
//#define REP 1
//#define OUTPUT
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
	double complex *vec_ans;
	double *h,*g, diff;
	struct timespec ts0, ts1;
	clock_t c0, c1;
	fftw_plan p;

	N = pow(2, n);
	#define ALLOC(x) x = (double complex*) malloc(sizeof(double complex)*N);
	ALLOC(vec_in);
	ALLOC(vec_tmp);
	ALLOC(vec_ans);

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

	fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_ans);
	for (i=1; i<REP; i++)
		fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	fftw_destroy_plan(p);

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_ans, N);
	#endif

	ifft(vec_ans, vec_tmp, N);
	#ifdef OUTPUT
	write_cvec(vec_tmp, N, "inv_fftw.dat");
	write_cvec(vec_ans, N, "freq_fftw.dat");
	#endif

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);


	//not computed under-sampled FFT anymore, since it's hard
	//to get the timing info accuratelly.

	#if 0

	/*************************
	*   Under-sampled FFTW   *
	**************************/

	printf("\nTesting Under-sampled FFTW\n");

	create_signal(vec_in, N, SIGNAL);
	under_sample(vec_in, N);


	p = fftw_create_plan(N>>1, FFTW_FORWARD, FFTW_ESTIMATE);
	//p = fftw_create_plan(N, FFTW_FORWARD, FFTW_MEASURE);
	//create_signal(vec_in, N, SIGNAL);

	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();
	//under_sample(vec_in, N);

	under_sample(vec_tmp, N); //just for timing purposes
	fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_ans);
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

	//use under-sampled FFT to reconstruct signal, and over-sample it
	ifft(vec_ans, vec_tmp, N>>1);
	over_sample(vec_tmp, N);
	//write_cvec(vec_us1, N>>1, "freq_us1.dat"); //note: this FFT contains N-1 freqs!

	//recalculate exact FFT from over-sampled signal
	p = fftw_create_plan(N, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_one(p, (fftw_complex*) vec_tmp, (fftw_complex*) vec_ans);
	fftw_destroy_plan(p);

	#ifdef OUTPUT
	write_cvec(vec_tmp, N, "inv_us1.dat");
	write_cvec(vec_ans, N, "freq_us1.dat"); //note: this FFT contains N-1 freqs!
	#endif

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);

	#endif

	/********************************************
	*   Felipe's FFT  (bit-reversed in-place)   *
	********************************************/

	printf("\nTesting FFFT\n");

	create_signal(vec_in, N, SIGNAL);
	prepare_fft(N);
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	fft3(vec_in, vec_ans, N);
	for (i=1; i<REP; i++)
		fft3(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_fft();


	/******************************
	*    SWIFFT - Haar Wavelet 1  *
	*******************************/

	printf("\nTesting SWIFFT - Haar Wavelet - Alg. #1\n");

	create_signal(vec_in, N, SIGNAL);
	
	h = (double*) calloc(N, sizeof(double));
	g = (double*) calloc(N, sizeof(double));
	//g[0]=0.5; g[1]=g[0];
	g[0]=sqrt(2.)*0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_swifft(N, h, 2, g, 2, 1);

	for (i=0; i<N; i++) vec_ans[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_haar1(vec_in, vec_ans, N);
	for (i=1; i<REP; i++)
		swifft_haar1(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_ans, N);
	#endif

	ifft(vec_ans, vec_tmp, N);
	#ifdef OUTPUT
	write_cvec(vec_tmp, N, "inv_haar1.dat");
	write_cvec(vec_ans, N, "freq_haar1.dat");
	#endif

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);


	/*********************************************
	*    SWIFFT - Haar Wavelet 1 - Non-orthog    *
	**********************************************/

	printf("\nTesting SWIFFT - Haar Wavelet - Alg. #1 - Non-orthog\n");

	create_signal(vec_in, N, SIGNAL);
	h = (double*) calloc(N, sizeof(double));
	g = (double*) calloc(N, sizeof(double));
	g[0]=0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_swifft(N, h, 2, g, 2, 1);

	for (i=0; i<N; i++) vec_ans[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	//XXX
	swifft_haar1_non_orthog(vec_in, vec_ans, N);
	//swifft_gen1(vec_in, vec_ans, N);
	for (i=1; i<REP; i++)
		//XXX
		swifft_haar1_non_orthog(vec_in, vec_tmp, N);
		//swifft_gen1(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_ans, N);
	#endif

	ifft(vec_ans, vec_tmp, N);
	#ifdef OUTPUT
	write_cvec(vec_tmp, N, "inv_haar1.dat");
	write_cvec(vec_ans, N, "freq_haar1.dat");
	#endif

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);


	#if 0
	/******************************
	*    SWIFFT - Haar Wavelet 2  *
	*******************************/

	printf("\nTesting SWIFFT - Haar Wavelet - Alg. #2\n");

	create_signal(vec_in, N, SIGNAL);
	
	//use lazy wavelet filter;
	h = (double*) calloc(N, sizeof(double));
	g = (double*) calloc(N, sizeof(double));
	//g[0]=0.5; g[1]=g[0];
	g[0]=sqrt(2.)*0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_swifft(N, h, 2, g, 2, 1);

	for (i=0; i<N; i++) vec_ans[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_haar2(vec_in, vec_ans, N);
	for (i=1; i<REP; i++)
		swifft_haar2(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_ans, N);
	#endif

	ifft(vec_ans, vec_tmp, N);
	#ifdef OUTPUT
	write_cvec(vec_tmp, N, "inv_haar2.dat");
	write_cvec(vec_ans, N, "freq_haar2.dat");
	#endif

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);
	#endif


	/*******************************
	*    SWIFFT - DB2 - Alg. #1    *
	********************************/

	printf("\nTesting SWIFFT - DB2 Wavelet - Alg. #1\n");

	create_signal(vec_in, N, SIGNAL);
	h = (double*) calloc(N, sizeof(double));
	g = (double*) calloc(N, sizeof(double));

	h[0]=-0.12940952255092145;h[1]=-0.22414386804185735;h[2]=0.83651630373746899;h[3]=-0.48296291314469025;
	g[0]=0.48296291314469025;g[1]=0.83651630373746899;g[2]=0.22414386804185735;g[3]=-0.12940952255092145;
	prepare_swifft(N, h, 4, g, 4, 2);

	for (i=0; i<N; i++) vec_ans[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_gen1(vec_in, vec_ans, N);
	for (i=1; i<REP; i++)
		swifft_gen1(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_ans, N);
	#endif

	ifft(vec_ans, vec_tmp, N);
	#ifdef OUTPUT
	write_cvec(vec_tmp, N, "inv_db2.dat");
	write_cvec(vec_ans, N, "freq_db2.dat");
	#endif

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);

	/*******************************
	*    SWIFFT - DB3 - Alg. #1    *
	********************************/

	printf("\nTesting SWIFFT - DB3 Wavelet - Alg. #1\n");

	create_signal(vec_in, N, SIGNAL);
	h = (double*) calloc(N, sizeof(double));
	g = (double*) calloc(N, sizeof(double));

	h[0]=0.035226291882100656;
	h[1]=0.085441273882241486;
	h[2]=-0.13501102001039084;
	h[3]=-0.45987750211933132;
	h[4]=0.80689150931333875;
	h[5]=-0.33267055295095688;

	g[0]=0.33267055295095688;
	g[1]=0.80689150931333875;
	g[2]=0.45987750211933132;
	g[3]=-0.13501102001039084;
	g[4]=-0.085441273882241486;
	g[5]=0.035226291882100656;
	prepare_swifft(N, h, 6, g, 6, 2);

	for (i=0; i<N; i++) vec_ans[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_gen1(vec_in, vec_ans, N);
	for (i=1; i<REP; i++)
		swifft_gen1(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();

	#ifdef PRINT
	printf("\nAnswer:\n");
	print_cvec(vec_ans, N);
	#endif

	ifft(vec_ans, vec_tmp, N);
	#ifdef OUTPUT
	write_cvec(vec_tmp, N, "inv_db3.dat");
	write_cvec(vec_ans, N, "freq_db3.dat");
	#endif

	create_signal(vec_in, N, SIGNAL);
	diff = diff_norm(vec_tmp, vec_in, N);
	printf("\nError (2-norm): %lf\n", diff);

	return 0;
}

