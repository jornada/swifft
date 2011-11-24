
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
long long int sz_full, sz_half; //full (orig.) matrix
long long int faft_step;
double complex *w;

#define max(a,b) a>b?a:b

//craft
//waffle?
//civet? - concrete implementation of v... ex

//"naive" FFT specialized for real sparse data -- only takes O(sz_out*sz_in) flops
void sparse_FFT(double *in, int sz_in, double complex *out, long long int sz_out){
	int i, j, j2, sz2;

	sz2 = sz_out>>1;

	//i=0
	for (j=0; j<sz_in; j++){
		out[0] += in[j];
	}

	//i=sz2
	j2 = -sz2;
	for (j=0; j<sz_in; j++){
		j2 = (j2+sz2)%sz_out;
		out[sz2] += w[(sz2*j)%sz_out]*in[j];
	}

	//rest
	for (i=1; i<sz2; i++){
		j2 = -i;
		for (j=0; j<sz_in; j++){
			j2 = (j2+i)%sz_out;
			out[i] += w[j2]*in[j];
		}
		out[sz_out-i] = conj(out[i]);
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

	sz_full = sz;
	sz_half = sz_full>>1;
	H = (double complex*) calloc(sz, sizeof(double complex));
	G = (double complex*) calloc(sz, sizeof(double complex));

	h = h_;	h_sz = h_sz_;
	g = g_;	g_sz = g_sz_;

	sparse_FFT(h, h_sz, H, sz);
	sparse_FFT(g, g_sz, G, sz);

	/*
	printf("h ");
	print_vec(h, sz);
	printf("H ");
	print_cvec(H, sz);
	printf("g ");
	print_vec(g, sz);
	printf("G ");
	print_cvec(G, sz);
	*/

	faft_step=1;
}

void free_faft(){
	free(scratch);
	free(w);
	free(H);
	free(G);
}

void faft(double complex *in, double complex *out, long long int sz){
	long long int i, i2, j,j2, sz2;
	double complex tmp;

	//printf("Caling faft, sz=%lld, faftw_step=%lld\n", sz, faft_step);
	if (sz==1){
		*out = (double complex) *in;
	} else {
		sz2 = sz>>1;
		//wavelet transform
		memcpy(scratch, in, sz*sizeof(double complex));
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
	
		faft_step = faft_step<<1;
		faft(in, out, sz2);
		faft(in+sz2, out+sz2, sz2);
		faft_step = faft_step>>1;

		//multiply by H and G
		j = -faft_step;
		for (i=0; i<sz2; i++){
			j += faft_step;
			j2 = j + sz_half;
			i2 = i + sz2;
			tmp = H[j]*out[i] + G[j]*out[i2];
			out[i2] = H[j2]*out[i] + G[j2]*out[i2];
			out[i] = tmp;
		
			/*
			if (sz==4) {
				printf("i=%lld, j =%lld\n",i,j);
				printf("w=(%f,%f), H=(%f,%f), G=(%f,%f)\n",
					(float) creal(w[j]), (float) cimag(w[j]),
					(float) creal(H[j]), (float) cimag(H[j]),
					(float) creal(G[j]), (float) cimag(G[j]) );
				printf("i=%lld, j'=%lld\n",i,j+sz_half);
				printf("w=(%f,%f), H=(%f,%f), G=(%f,%f)\n",
					(float) creal(w[j+sz_half]), (float) cimag(w[j+sz_half]),
					(float) creal(H[j+sz_half]), (float) cimag(H[j+sz_half]),
					(float) creal(G[j+sz_half]), (float) cimag(G[j+sz_half]) );
				printf("\n");
			}
			*/
			
			
		}
	}
}

