
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
//#include <rfftw.h>
#include <fftw.h>
#include "wavelets.h"
#include "utils.h"

double complex *scratch2, *buf1, *buf2;
//note: 1st half of the coefficients: real!=0 , 2nd half: imaginary=0
double *h, *g;
double complex *H, *G;
int h_sz, g_sz, max_window;
int sz_full, sz_half; //full (orig.) matrix
int swifft_step;
double complex *w2;
fftw_plan p;
int p_init;

#define SQRT2_2 0.7071067811865475 
#define max(a,b) (a>b?a:b)

//swifft - swifft Wavelet-based Inexact Fast Fourier Transform

//"naive" FFT specialized for real sparse data -- takes only O(sz_out*sz_in) flops
void sparse_FFT(double *in, int sz_in, double complex *out, int sz_out){
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
		out[sz2] += w2[(sz2*j)%sz_out]*in[j];
	}

	//rest
	for (i=1; i<sz2; i++){
		j2 = -i;
		for (j=0; j<sz_in; j++){
			j2 = (j2+i)%sz_out;
			out[i] += w2[j2]*in[j];
		}
		out[sz_out-i] = conj(out[i]);
	}
}

void prepare_swifft(int sz, double *h_, int h_sz_, double *g_, int g_sz_){
	//rfftw_plan p;
	//fftw_plan p;
	double alpha;
	int i;

	scratch2 = (double complex*) malloc(sizeof(double complex)*sz);
	buf1 = (double complex*) malloc(sizeof(double complex)*sz);
	buf2 = (double complex*) malloc(sizeof(double complex)*sz);

	w2 = (double complex*) malloc(sizeof(double complex)*sz);
	alpha = 2.0*M_PI/(double)sz;

	for (i=0; i<sz; i++){
		w2[i] = cos(alpha*i) - I*sin(alpha*i);
	}

	sz_full = sz;
	sz_half = sz_full>>1;
	H = (double complex*) calloc(sz, sizeof(double complex));
	G = (double complex*) calloc(sz, sizeof(double complex));

	h = h_;	h_sz = h_sz_;
	g = g_;	g_sz = g_sz_;
	max_window = max(h_sz, g_sz);

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

	swifft_step=1;
	p_init = 0;
}

void free_swifft(){
	free(scratch2);
	free(w2);
	free(H);
	free(G);
}

#define SWAP(a,b) TMP=b; b=a; a=TMP;
/*
void swifft(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2;
	double complex tmp;
	//double complex *TMP;


	//printf("Caling swifft, sz=%lld, swifftw_step=%lld\n", sz, swifft_step);
	sz2 = sz>>1;
	//wavelet transform

	*
	//idea: swap buffer in with last used buffer (buf2)
	// write result into buf1, pass buf1 as in
	SWAP(buf1, buf2)
	for (i=0; i<sz2; i++) {
		buf1[i]=0;
		buf1[i+sz2]=0;
		for (j=0; j<max_window; j++){
			j2 = (2*i + j)%sz;
			buf1[i] += h[j]*in[j2];
			buf1[i+sz2] += g[j]*in[j2];
		}
	}/
	
	if (swifft_step > 1<<3){
		*
		memcpy(scratch2, in, sz*sizeof(double complex));
		for (i=0; i<sz2; i++) {
			in[i]=0;
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (2*i + j)%sz;
				//in[i] += h[j]*scratch2[j2];
				//in[i+sz2] += g[j]*scratch2[j2];
				in[i] += h[j]*scratch2[j2];
				in[i+sz2] += g[j]*scratch2[j2];
			}
		}
		swifft_step = swifft_step<<1;
		if (sz>2){
			swifft(in, out, sz2);
			swifft(in+sz2, out+sz2, sz2);
		} else {
			out[0] = in[0];
			out[1] = in[1];
		}
		swifft_step = swifft_step>>1;
		j = -swifft_step;
		for (i=0; i<sz2; i++){
			j += swifft_step;
			j2 = j + sz_half;
			i2 = i + sz2;
			tmp = H[j]*out[i] + G[j]*out[i2];
			out[i2] = H[j2]*out[i] + G[j2]*out[i2];
			out[i] = tmp;
		}/
		
		//stop prunning, do regular FFT!
		if (!p_init){
			p = fftw_create_plan(sz, FFTW_FORWARD, FFTW_ESTIMATE);
			p_init = 1;
		}
		fftw_one(p, (fftw_complex*) in, (fftw_complex*) out);
		//fftw_destroy_plan(p);
		
	} else {
		//some condition
		if (swifft_step==1){
		
		memcpy(scratch2, in, sz*sizeof(double complex));
		for (i=0; i<sz2; i++) {
			//in[i]=0;
			(*
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (2*i + j)%sz;
				//j2=0;
				//in[i] += h[j]*scratch2[j2];
				in[i+sz2] += g[j]*scratch2[j2];
			}
			*)
			in[i+sz2] = SQRT2_2*( scratch2[2*i] + scratch2[2*i+1] );
		}
		
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}

		swifft_step = swifft_step<<1;
		//if (sz>2){
			swifft(in+sz2, out+sz2, sz2);
		//} else {
		//	out[0] = in[0];
		//	out[1] = in[1];
		//}
		swifft_step = swifft_step>>1;
		j = -swifft_step;
		for (i=0; i<sz2; i++){
			j += swifft_step;
			j2 = j + sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
	}

	*
	//multiply by H and G
	j = -swifft_step;
	for (i=0; i<sz2; i++){
		j += swifft_step;
		j2 = j + sz_half;
		i2 = i + sz2;
		tmp = H[j]*out[i] + G[j]*out[i2];
		out[i2] = H[j2]*out[i] + G[j2]*out[i2];
		out[i] = tmp;
	}
	*
}
*/


//This routine takes any input filter
void swifft_gen(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2;
	double complex tmp;
	//double complex *TMP;


	//printf("Caling swifft, sz=%lld, swifftw_step=%lld\n", sz, swifft_step);
	sz2 = sz>>1;
	//wavelet transform

	/*
	//idea: swap buffer in with last used buffer (buf2)
	// write result into buf1, pass buf1 as in
	SWAP(buf1, buf2)
	for (i=0; i<sz2; i++) {
		buf1[i]=0;
		buf1[i+sz2]=0;
		for (j=0; j<max_window; j++){
			j2 = (2*i + j)%sz;
			buf1[i] += h[j]*in[j2];
			buf1[i+sz2] += g[j]*in[j2];
		}
	}*/
	
	if (swifft_step > 1<<3){
		/*
		memcpy(scratch2, in, sz*sizeof(double complex));
		for (i=0; i<sz2; i++) {
			in[i]=0;
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (2*i + j)%sz;
				//in[i] += h[j]*scratch2[j2];
				//in[i+sz2] += g[j]*scratch2[j2];
				in[i] += h[j]*scratch2[j2];
				in[i+sz2] += g[j]*scratch2[j2];
			}
		}
		swifft_step = swifft_step<<1;
		if (sz>2){
			swifft(in, out, sz2);
			swifft(in+sz2, out+sz2, sz2);
		} else {
			out[0] = in[0];
			out[1] = in[1];
		}
		swifft_step = swifft_step>>1;
		j = -swifft_step;
		for (i=0; i<sz2; i++){
			j += swifft_step;
			j2 = j + sz_half;
			i2 = i + sz2;
			tmp = H[j]*out[i] + G[j]*out[i2];
			out[i2] = H[j2]*out[i] + G[j2]*out[i2];
			out[i] = tmp;
		}*/
		
		//stop prunning, do regular FFT!
		if (!p_init){
			p = fftw_create_plan(sz, FFTW_FORWARD, FFTW_ESTIMATE);
			p_init = 1;
		}
		fftw_one(p, (fftw_complex*) in, (fftw_complex*) out);
		//fftw_destroy_plan(p);
		
	} else {
		//some condition
		//if (swifft_step==1){
		/*
		memcpy(scratch2, in, sz*sizeof(double complex));
		for (i=0; i<sz2; i++) {
			//in[i]=0;
			(*
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (2*i + j)%sz;
				//j2=0;
				//in[i] += h[j]*scratch2[j2];
				in[i+sz2] += g[j]*scratch2[j2];
			}
			*)
			in[i+sz2] = SQRT2_2*( scratch2[2*i] + scratch2[2*i+1] );
		}
		*/
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}

		swifft_step = swifft_step<<1;
		//if (sz>2){
			swifft_gen(in+sz2, out+sz2, sz2);
		//} else {
		//	out[0] = in[0];
		//	out[1] = in[1];
		//}
		swifft_step = swifft_step>>1;
		j = -swifft_step;
		for (i=0; i<sz2; i++){
			j += swifft_step;
			j2 = j + sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
	}

	/*
	//multiply by H and G
	j = -swifft_step;
	for (i=0; i<sz2; i++){
		j += swifft_step;
		j2 = j + sz_half;
		i2 = i + sz2;
		tmp = H[j]*out[i] + G[j]*out[i2];
		out[i2] = H[j2]*out[i] + G[j2]*out[i2];
		out[i] = tmp;
	}
	*/
}


// This routine is specialized for Haar-type filter
void swifft_haar(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2;

	//printf("Caling swifft, sz=%lld, swifftw_step=%lld\n", sz, swifft_step);
	sz2 = sz>>1;

	if (swifft_step > 1<<0){
		//stop prunning, do regular FFT using FFTW.
		if (!p_init){
			p = fftw_create_plan(sz, FFTW_FORWARD, FFTW_ESTIMATE);
			p_init = 1;
		}
		fftw_one(p, (fftw_complex*) in, (fftw_complex*) out);
	} else {
		//Do wavelet transform only at the lower part, i.e., discart
		//detail coeffs. Note that we don`t need any "scratch memory".
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}

		//Recursivelly call swifft_haar
		swifft_step = swifft_step<<1;
		swifft_haar(in+sz2, out+sz2, sz2);
		swifft_step = swifft_step>>1;

		//Multiply by the FFT of the wavelet coeffs.
		j = -swifft_step;
		for (i=0; i<sz2; i++){
			j += swifft_step;
			j2 = j + sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
	}
	return;
}
