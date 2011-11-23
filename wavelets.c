
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
//#include <rfftw.h>
//#include <fftw.h>
#include "wavelets.h"
#include "utils.h"

double complex *scratch;
//note: 1st half of the coefficients: real!=0 , 2nd half: imaginary=0
double *h, *g;
double complex *H, *G;
int h_sz, g_sz;
long long int faft_step;
double complex *w;

#define max(a,b) a>b?a:b

//waffle?
//civet? - concrete implementation of v... ex

//FFT specialized for sparse data
void sparse_FFT(double *in, int sz_in, double complex *out, long long int sz_out){
	int i,j;
	for (i=0; i<sz_out; i++){
		out[i] = (double complex) 0;
		for (j=0; j<sz_in; j++){
			out[i] += w[i*j]*in[j];
		}
	}
}

void prepare_faft(long long int sz, double *h_, int h_sz_, double *g_, int g_sz_){
	//rfftw_plan p;
	//fftw_plan p;
	double alpha;
	int i;

	scratch = (double complex*) malloc(sizeof(double complex)*sz);

	w = (double complex*) malloc(sizeof(double complex)*sz);
	alpha = 2.0*M_PI/(double)sz;

	for (i=0; i<sz; i++){
		w[i] = cos(alpha*i) - I*sin(alpha*i);
	}

	H = (double complex*) malloc(sizeof(double complex)*sz);
	G = (double complex*) malloc(sizeof(double complex)*sz);

	h = h_;	h_sz = h_sz_;
	g = g_;	g_sz = g_sz_;

	sparse_FFT(h, h_sz, H, sz);
	sparse_FFT(g, g_sz, G, sz);
	
	/*	
	p = rfftw_create_plan(sz, FFTW_FORWARD, FFTW_ESTIMATE);
	rfftw_one(p, (fftw_real*) h, (fftw_real*) H);
	rfftw_one(p, (fftw_real*) g, (fftw_real*) G);
	fftw_destroy_plan(p);
	*/

	/*
	p = fftw_plan_dft_r2c_1d(sz, h, (fftw_complex*) H, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);

	p = fftw_plan_dft_r2c_1d(sz, g, (fftw_complex*) G, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
	*/

	// note: H and G are organized as follows:
	// [r0, r1, r2, ..., rn/2, i(n+1)/2-1, ..., i2, i1]
	print_vec(h, sz);
	print_cvec(H, sz);
	print_vec(g, sz);
	print_cvec(G, sz);

	faft_step=1;
}

void free_faft(){
	free(scratch);
	free(w);
	free(H);
	free(G);
}

void faft(double complex *in, double complex *out, long long int sz){
	long long int i,j,j2, sz2;
	double complex tmp;

	//if (sz==8) printf("!\n");
	//printf("Caling fft, sz=%lld, fftw_step=%lld\n", sz, fft_step);
	if (sz==1){
		*out = (double complex) *in;
	} else {
		sz2 = sz>>1;
		//wavelet transform
		memcpy(scratch, in, sz*sizeof(double complex));
		//XXX I think this is right, but double check meh!
		//if (sz==8) print_cvec(in, sz);
		//if (sz==8) {print_vec(h, h_sz); print_vec(g, g_sz);}
		for (i=0; i<sz2; i++) {
			in[i]=0;
			for (j=0; j<h_sz; j++){
				j2 = (2*i + j)%sz;
				in[i] += h[j]*scratch[j2];
			}
			in[i+sz2]=0;
			for (j=0; j<g_sz; j++){
				j2 = (2*i + j)%sz;
				in[i+sz2] += g[j]*scratch[j2];
			}
		}
		//if (sz==8) print_cvec(in, sz);
	
		faft_step = faft_step<<1;
		faft(in, out, sz2);
		faft(in+sz2, out+sz2, sz2);
		faft_step = faft_step>>1;

		//FIXME (i think the problem is here...)
		j = -faft_step;
		for (i=0; i<sz2; i++){
			j += faft_step;
			tmp = H[j]*out[i] + G[j]*out[i+sz2];
			out[i+sz2] = H[j+sz2]*out[i] + G[j+sz2]*out[i+sz2];
			out[i] = tmp;
		}
		
	}
}

