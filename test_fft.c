/* test_fft.c
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

#define PRINT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw.h>

#include <time.h>
#include "fft.h"
#include "utils.h"

#define n 16
#define REP 1

int main(int argc, char **argv){
	long long int N, i;
	double complex *vec_in;
	double complex *vec_out;
	fftw_plan p;
	struct timespec ts0, ts1;
	clock_t c0, c1;

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

	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();
	for (i=0; i<REP; i++)
		fft(vec_in, vec_out, N);
	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	
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

	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();
	for (i=0; i<REP; i++)
		fft2(vec_in, vec_out, N);
	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	
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

	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();
	for (i=0; i<REP; i++)
		fft3(vec_in, vec_out, N);
	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);
	
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

	clock_gettime(CLOCK_REALTIME, &ts0); c0 = clock();
	for (i=0; i<REP; i++)
	fftw_one(p, (fftw_complex*) vec_in, (fftw_complex*) vec_out);
	clock_gettime(CLOCK_REALTIME, &ts1); c1 = clock();
	print_times(ts0, ts1, c0, c1);

	fftw_destroy_plan(p);

	#ifdef PRINT
	printf("Output vector:\n");
	print_cvec(vec_out, N);
	#endif

	return 0;
}

