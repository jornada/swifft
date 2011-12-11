/* test_swifft.c
 * 
 * SWIFFT - Swift Wavelet-based Inexact FFT
 * Copyright (C) 2011 Felipe H. da Jornada <jornada@civet.berkeley.edu>
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw.h>
#include <string.h>
#include <time.h>

#include "utils.h"
#include "fft.h"
#include "swifft.h"

//! Don't do any error analysis
//#define FAST
//#define OUTPUT
//#define PRINT
#define SIGNAL 5

//uncoment to enable/disable a particular test
//#define FFFT
//#define GB_FFT
//#define HAAR1
#define HAAR1_NON_ORTHOG
//#define HAAR2
#define DB2
//#define DB3
#define DB4
//#define DB5
#define DB6
#define DB10

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

void finalize_test(double complex *v_ans, double complex *v_approx, double complex *v_tmp, double complex *v_in, int sz, char *in_str){
	double diff;
	char tmp_str[128];

#ifndef FAST
#ifdef PRINT
	printf("\n\tAnswer:\n");
	print_cvec(v_approx, sz);
#endif

#ifdef OUTPUT
	ifft(v_approx, v_tmp, sz);
	sprintf(tmp_str, "inv_%s.dat", in_str);
	write_cvec(v_tmp, sz, tmp_str);
	sprintf(tmp_str, "freq_%s.dat", in_str);
	write_cvec(v_approx, sz, tmp_str);
	create_signal(v_in, sz, SIGNAL);
#endif

	diff = diff_norm(v_approx, v_ans, sz);
	printf("\tError (2-norm): %lf\n", diff);
#endif
}
	
#define ALLOC(x) x = (double complex*) malloc(sizeof(double complex)*N)

//! Test various implementations, but all with the same pruning depth
void swifft_test1(int n, int REP){
	int N, i;
	double complex *vec_in, *vec_tmp;
	double complex *vec_approx, *vec_ans;
	double *h,*g, diff;
	struct timespec ts0, ts1;
	clock_t c0, c1;
	int depth=2;
	fftw_plan p;

	N = pow(2, n);
	ALLOC(vec_in);
	ALLOC(vec_tmp);
	ALLOC(vec_approx);
	ALLOC(vec_ans);
	// initialize wavelet filter banks
	h = (double*) malloc(N*sizeof(double));
	g = (double*) malloc(N*sizeof(double));

	printf("\n");
	printf("SWIFFT - BENCHMARK\n");
	printf("------------------\n");
	printf("  signal size: n = %i\n", n);
	printf("         N = 2^n = %i\n", N);
	printf("  repetitions    = %i\n", REP);
	printf("  signal type    = %i\n", (int) SIGNAL);
	printf("\n");


	/************************
	*         FFTW          *
	************************/

	printf("\n# FFTW\n");

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
	finalize_test(vec_ans, vec_ans, vec_tmp, vec_in, N, "fftw");

#ifdef FFFT
	/********************************************
	*   Felipe's FFT  (bit-reversed in-place)   *
	********************************************/

	// This is the (C) FFT in my paper
	printf("\n# FFFT\n");

	create_signal(vec_in, N, SIGNAL);
	prepare_fft(N);
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	fft3(vec_in, vec_approx, N);
	for (i=1; i<REP; i++)
		fft3(vec_in, vec_tmp, N);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_fft();
	//I know it's right, and I just want the time info
	//finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "gb_fft");
#endif

#ifdef GB_FFT
	/**********************************
	*       Full Guo-Burrus FFT       *
	***********************************/

	printf("\n# Full Guo-Burrus FFT (via Haar Wavelet)\n");

	//wavelet filter
	memset(h, 0, N*sizeof(double));
	memset(g, 0, N*sizeof(double));
	g[0]=sqrt(2.)*0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_swifft(N, h, 2, g, 2, depth);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_full(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_full(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	//I know it's right, and I just want the time info
	//finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "gb_fft");
#endif

#ifdef HAAR1
	/******************************
	*    SWIFFT - Haar Wavelet 1  *
	*******************************/

	printf("\n# Haar Wavelet - Alg. #1\n");

	//wavelet filter
	memset(h, 0, N*sizeof(double));
	memset(g, 0, N*sizeof(double));
	g[0]=sqrt(2.)*0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_swifft(N, h, 2, g, 2, depth);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_haar1(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_haar1(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "haar1");
#endif

#ifdef HAAR1_NON_ORTHOG
	/*********************************************
	*    SWIFFT - Haar Wavelet 1 - Non-orthog    *
	**********************************************/

	printf("\n# Haar Wavelet - Alg. #1 - Non-orthog\n");

	//wavelet filter	
	memset(h, 0, N*sizeof(double));
	memset(g, 0, N*sizeof(double));
	g[0]=0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_swifft(N, h, 2, g, 2, depth);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_haar1_non_orthog(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_haar1_non_orthog(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "haar1_non_orthog");
#endif

#ifdef HAAR2
	/******************************
	*    SWIFFT - Haar Wavelet 2  *
	*******************************/

	printf("\n# Haar Wavelet - Alg. #2\n");

	//wavelet filter	
	memset(h, 0, N*sizeof(double));
	memset(g, 0, N*sizeof(double));
	g[0]=sqrt(2.)*0.5; g[1]=g[0];
	h[0]=g[0];h[1]=-g[0];
	prepare_swifft(N, h, 2, g, 2, 1);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_haar2(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_haar2(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "haar2");
#endif

#ifdef DB2
	/*******************************
	*    SWIFFT - DB2 - Alg. #1    *
	********************************/

	printf("\n# DB2 Wavelet - Alg. #1\n");

	//wavelet filter	
	load_filters(h, g, N, "filters/db2.filter");
	prepare_swifft(N, h, 4, g, 4, depth);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_gen1(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_gen1(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db2");
#endif

#ifdef DB3
	/*******************************
	*    SWIFFT - DB3 - Alg. #1    *
	********************************/

	printf("\n# DB3 Wavelet - Alg. #1\n");

	//wavelet filter	
	load_filters(h, g, N, "filters/db3.filter");
	prepare_swifft(N, h, 6, g, 6, depth);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_gen1(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_gen1(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db3");
#endif

#ifdef DB4
	/*******************************
	*    SWIFFT - DB4 - Alg. #1    *
	********************************/

	printf("\n# DB4 Wavelet - Alg. #1\n");

	//wavelet filter	
	load_filters(h, g, N, "filters/db4.filter");
	prepare_swifft(N, h, 8, g, 8, depth);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_gen1(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_gen1(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db4");
#endif

#ifdef DB5
	/*******************************
	*    SWIFFT - DB5 - Alg. #1    *
	********************************/

	printf("\n# DB5 Wavelet - Alg. #1\n");

	//wavelet filter	
	load_filters(h, g, N, "filters/db5.filter");
	prepare_swifft(N, h, 10, g, 10, depth);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_gen1(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_gen1(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db5");
#endif

#ifdef DB6	
	/*******************************
	*    SWIFFT - DB6 - Alg. #1    *
	********************************/

	printf("\n# DB6 Wavelet - Alg. #1\n");

	//wavelet filter	
	load_filters(h, g, N, "filters/db6.filter");
	prepare_swifft(N, h, 10, g, 10, depth);

	create_signal(vec_in, N, SIGNAL);
	for (i=0; i<N; i++) vec_approx[i] = 0.0;
	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

	swifft_gen1(vec_in, vec_approx);
	for (i=1; i<REP; i++)
		swifft_gen1(vec_in, vec_tmp);

	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	free_swifft();
	finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db6");
#endif

	/*******************************/

	free(h);
	free(g);
	free(vec_in);
	free(vec_tmp);
	free(vec_approx);
	free(vec_ans);
}

//! Test some implementations using different pruning depths
void swifft_test2(int n, int REP){
	int N, i;
	double complex *vec_in, *vec_tmp;
	double complex *vec_approx, *vec_ans;
	double *h,*g, diff;
	struct timespec ts0, ts1;
	clock_t c0, c1;
	fftw_plan p;
	int depth;

	N = pow(2, n);
	ALLOC(vec_in);
	ALLOC(vec_tmp);
	ALLOC(vec_approx);
	// initialize wavelet filter banks
	h = (double*) malloc(N*sizeof(double));
	g = (double*) malloc(N*sizeof(double));

	printf("\n");
	printf("SWIFFT - BENCHMARK\n");
	printf("------------------\n");
	printf("  signal size: n = %i\n", n);
	printf("         N = 2^n = %i\n", N);
	printf("  repetitions    = %i\n", REP);
	printf("  signal type    = %i\n", (int) SIGNAL);
	printf("\n");

	/************************
	*         FFTW          *
	************************/

	printf("\n# FFTW\n");

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
	finalize_test(vec_ans, vec_ans, vec_tmp, vec_in, N, "fftw");

	for (depth=1; depth<5; depth++){
		printf("\n\nDEPTH = %d\n\n",depth);

		/*********************************************
		*    SWIFFT - Haar Wavelet 1 - Non-orthog    *
		**********************************************/

		printf("\n# Haar Wavelet - Alg. #1 - Non-orthog\n");

		//wavelet filter	
		memset(h, 0, N*sizeof(double));
		memset(g, 0, N*sizeof(double));
		g[0]=0.5; g[1]=g[0];
		h[0]=g[0];h[1]=-g[0];
		prepare_swifft(N, h, 2, g, 2, depth);

		create_signal(vec_in, N, SIGNAL);
		for (i=0; i<N; i++) vec_approx[i] = 0.0;
		clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

		swifft_haar1_non_orthog(vec_in, vec_approx);
		for (i=1; i<REP; i++)
			swifft_haar1_non_orthog(vec_in, vec_tmp);

		clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
		print_times(ts0, ts1, c0, c1);
		free_swifft();
		finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "haar1_non_orthog");
	
	
		/*******************************
		*    SWIFFT - DB2 - Alg. #1    *
		********************************/

		printf("\n# DB2 Wavelet - Alg. #1\n");

		//wavelet filter	
		load_filters(h, g, N, "filters/db2.filter");
		prepare_swifft(N, h, 4, g, 4, depth);

		create_signal(vec_in, N, SIGNAL);
		for (i=0; i<N; i++) vec_approx[i] = 0.0;
		clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

		swifft_gen1(vec_in, vec_approx);
		for (i=1; i<REP; i++)
			swifft_gen1(vec_in, vec_tmp);

		clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
		print_times(ts0, ts1, c0, c1);
		free_swifft();
		finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db2");

		
		/*******************************
		*    SWIFFT - DB4 - Alg. #1    *
		********************************/

		printf("\n# DB4 Wavelet - Alg. #1\n");

		//wavelet filter	
		load_filters(h, g, N, "filters/db4.filter");
		prepare_swifft(N, h, 8, g, 8, depth);

		create_signal(vec_in, N, SIGNAL);
		for (i=0; i<N; i++) vec_approx[i] = 0.0;
		clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

		swifft_gen1(vec_in, vec_approx);
		for (i=1; i<REP; i++)
			swifft_gen1(vec_in, vec_tmp);

		clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
		print_times(ts0, ts1, c0, c1);
		free_swifft();
		finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db4");


		/*******************************
		*    SWIFFT - DB6 - Alg. #1    *
		********************************/

		printf("\n# DB6 Wavelet - Alg. #1\n");

		//wavelet filter	
		load_filters(h, g, N, "filters/db6.filter");
		prepare_swifft(N, h, 12, g, 12, depth);

		create_signal(vec_in, N, SIGNAL);
		for (i=0; i<N; i++) vec_approx[i] = 0.0;
		clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

		swifft_gen1(vec_in, vec_approx);
		for (i=1; i<REP; i++)
			swifft_gen1(vec_in, vec_tmp);

		clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
		print_times(ts0, ts1, c0, c1);
		free_swifft();
		finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db6");


		/*******************************
		*    SWIFFT - DB10 - Alg. #1    *
		********************************/

		printf("\n# DB10 Wavelet - Alg. #1\n");

		//wavelet filter	
		load_filters(h, g, N, "filters/db10.filter");
		prepare_swifft(N, h, 20, g, 20, depth);

		create_signal(vec_in, N, SIGNAL);
		for (i=0; i<N; i++) vec_approx[i] = 0.0;
		clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();

		swifft_gen1(vec_in, vec_approx);
		for (i=1; i<REP; i++)
			swifft_gen1(vec_in, vec_tmp);

		clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
		print_times(ts0, ts1, c0, c1);
		free_swifft();
		finalize_test(vec_ans, vec_approx, vec_tmp, vec_in, N, "db10");

		/*******************************/
	}

	free(h);
	free(g);
	free(vec_in);
	free(vec_tmp);
	free(vec_ans);
	free(vec_approx);
}

int main(int argc, char **argv){

	swifft_test1(8, 10000);
	swifft_test1(10, 1000);
	swifft_test1(12, 500);
	swifft_test1(16, 50);
	swifft_test1(18, 10);
	swifft_test1(20, 5);
	swifft_test1(22, 1);
	swifft_test1(23, 1);
	swifft_test1(24, 1);

	//swifft_test2(8, 10000);
	//swifft_test2(16, 50);
	//swifft_test2(22, 1);

	//swifft_test1(8, 100000);
	//swifft_test1(22, 1);
	//swifft_test1(20, 1);

	return 0;
}

