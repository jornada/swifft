
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
int swifft_shift, swifft_iter;
double complex *w2;
fftw_plan *ps, p;
int p_init;

#define SQRT2_2 0.7071067811865475 
#define max(a,b) (a>b?a:b)

//swifft - swifft Wavelet-based Inexact Fast Fourier Transform

void full_fft(double complex *in, double complex *out, int sz){
	fftw_plan p_;

	p_ = fftw_create_plan(sz, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_one(p_, (fftw_complex*) in, (fftw_complex*) out);
	fftw_destroy_plan(p_);
}

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
	int i, sz2;

	scratch2 = (double complex*) malloc(sizeof(double complex)*sz);
	buf1 = (double complex*) malloc(sizeof(double complex)*sz);
	buf2 = (double complex*) malloc(sizeof(double complex)*sz);

	w2 = (double complex*) malloc(sizeof(double complex)*sz);
	alpha = 2.0*M_PI/(double)sz;

	for (i=0; i<sz; i++){
		w2[i] = cos(alpha*i) - I*sin(alpha*i);
	}
	
	ps = malloc( sizeof(fftw_plan)*(int)(log2(sz)+1e-10) );
	sz2=sz; i=0;
	while (sz2>1){
		ps[i] = fftw_create_plan(sz2, FFTW_FORWARD, FFTW_ESTIMATE);
		sz2 >>= 1;
		i++;
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

	swifft_shift=1;
	swifft_iter=0;
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


	//printf("Caling swifft, sz=%lld, swifftw_step=%lld\n", sz, swifft_shift);
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
	
	if (swifft_shift > 1<<3){
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
		swifft_shift = swifft_shift<<1;
		if (sz>2){
			swifft(in, out, sz2);
			swifft(in+sz2, out+sz2, sz2);
		} else {
			out[0] = in[0];
			out[1] = in[1];
		}
		swifft_shift = swifft_shift>>1;
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
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
		if (swifft_shift==1){
		
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

		swifft_shift = swifft_shift<<1;
		//if (sz>2){
			swifft(in+sz2, out+sz2, sz2);
		//} else {
		//	out[0] = in[0];
		//	out[1] = in[1];
		//}
		swifft_shift = swifft_shift>>1;
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
	}

	*
	//multiply by H and G
	j = -swifft_shift;
	for (i=0; i<sz2; i++){
		j += swifft_shift;
		j2 = j + sz_half;
		i2 = i + sz2;
		tmp = H[j]*out[i] + G[j]*out[i2];
		out[i2] = H[j2]*out[i] + G[j2]*out[i2];
		out[i] = tmp;
	}
	*
}
*/

#if 0
//This routine takes any input filter
void swifft_gen(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2;
	double complex tmp;


	//printf("Caling swifft, sz=%lld, swifftw_step=%lld\n", sz, swifft_shift);
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
	
	if (swifft_shift > 1<<3){
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
		swifft_shift = swifft_shift<<1;
		if (sz>2){
			swifft(in, out, sz2);
			swifft(in+sz2, out+sz2, sz2);
		} else {
			out[0] = in[0];
			out[1] = in[1];
		}
		swifft_shift = swifft_shift>>1;
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
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

		swifft_shift = swifft_shift<<1;
		if (sz>2){
			if (swifft_shift < 1<<4)
				swifft_gen(in, out, sz2);
			full_fft(in+sz2, out+sz2, sz2);
			//swifft_gen(in+sz2, out+sz2, sz2);
		} else {
			out[0] = in[0];
			out[1] = in[1];
		}
		swifft_shift = swifft_shift>>1;

		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + sz_half;
			i2 = i + sz2;
			if (swifft_shift < 1<<3) {
				tmp = H[j]*out[i] + G[j]*out[i2];
				out[i2] = H[j2]*out[i] + G[j2]*out[i2];
			} else {
				//tmp = H[j]*out[i];
				//out[i2] = H[j2]*out[i];
				tmp = G[j]*out[i2];
				out[i2] = G[j2]*out[i2];
			}
			out[i] = tmp;
		}
	}

}
#endif

// This routine is specialized for Haar-type filter, and prunes all the detail coeffs.
// This is equivalent to doing the FFT on a under-sampled signal
void swifft_haar1(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2;

	//printf("Caling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_shift > 1){
		//stop prunning, do regular FFT using FFTW.
		fftw_one(ps[swifft_iter], (fftw_complex*) in, (fftw_complex*) out);
	} else {
		//Do wavelet transform only at the lower part, i.e., discart
		//detail coeffs. Note that we don`t need any "scratch memory".
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}

		//Recursivelly call swifft_haar, but ignore lower part of the decomposition
		swifft_shift = swifft_shift<<1;
		swifft_iter++;
		//swifft_haar(in, out, sz2);
		swifft_haar1(in+sz2, out+sz2, sz2);
		swifft_shift = swifft_shift>>1;
		swifft_iter--;

		//Multiply by the FFT of the wavelet coeffs.
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
	}
	return;
}

// This routine is specialized for Haar-type filter
#define DEPTH 2
void swifft_haar2(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2, sz4;
	double complex tmp;

	//printf("Caling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_shift < (1<<DEPTH)){
		//work on both parts of the matrix
		//but only the upper part of the input vector has to be saved
		
		memcpy(scratch2, in, sz2*sizeof(double complex));
		sz4 = sz2>>1;
		//work on the second half (up and down at the same time)
		for (i=sz2-1; i>=sz4; i--) {
			in[i] = SQRT2_2*( in[2*i] - in[2*i+1] );
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}
		//and now finish the first half
		for (i=sz4-1; i>=0; i--) {
			in[i] = SQRT2_2*( scratch2[2*i] - scratch2[2*i+1] );
			in[i+sz2] = SQRT2_2*( scratch2[2*i] + scratch2[2*i+1] );
		}
		
		/*
		memcpy(scratch2, in, sz*sizeof(double complex));
		//for(i=0; i<sz; i++) scratch2[i]=in[i];
		for (i=0; i<sz2; i++) {
			//in[i] = SQRT2_2*( scratch2[2*i] - scratch2[2*i+1] );
			in[i+sz2] = SQRT2_2*( scratch2[2*i] + scratch2[2*i+1] );
		}
		*/

	} else {
		//only work on the lower part
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}
	}

	//Recursivelly call swifft_haar2
	swifft_shift = swifft_shift<<1;
	swifft_iter++;
	//if (sz>2){
		if (swifft_shift < (1<<DEPTH))
			swifft_haar2(in, out, sz2);

		//use next line to really do a wavelet transform
		fftw_one(ps[swifft_iter], (fftw_complex*) in+sz2, (fftw_complex*) out+sz2);

		/*
		if (swifft_shift < (1<<DEPTH))
			swifft_haar2(in+sz2, out+sz2, sz2);
		else
			fftw_one(ps[swifft_iter], (fftw_complex*) in+sz2, (fftw_complex*) out+sz2);
		*/ 

		
		//swifft_gen(in+sz2, out+sz2, sz2);
	//} else {
	//	out[0] = in[0];
	//	out[1] = in[1];
	//}
	swifft_shift = swifft_shift>>1;
	swifft_iter--;

	//Multiply by the FFT of the wavelet coeffs.
	if (swifft_shift < (1<<DEPTH)){
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + sz_half;
			i2 = i + sz2;
			tmp = H[j]*out[i] + G[j]*out[i2];
			out[i2] = H[j2]*out[i] + G[j2]*out[i2];
			out[i] = tmp;
		}
	} else {
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
	}

	return;
}

// similar haar2, but out of place
void rec_swifft_haar3(double complex *in, double complex *out, int sz, double complex *in_buf){
	int i, i2, j,j2, sz2;
	double complex tmp;

	//printf("Caling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_shift < (1<<DEPTH)){
		//invert in and in_buf!

		for (i=sz2-1; i+1; i--) {
			in_buf[i] = SQRT2_2*( in[2*i] - in[2*i+1] );
			in_buf[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}

		swifft_shift = swifft_shift<<1;
		swifft_iter++;
		fftw_one(ps[swifft_iter], (fftw_complex*) in_buf+sz2, (fftw_complex*) out+sz2);
		rec_swifft_haar3(in_buf, out, sz2, in);
		swifft_shift = swifft_shift>>1;
		swifft_iter--;

		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + sz_half;
			i2 = i + sz2;
			tmp = H[j]*out[i] + G[j]*out[i2];
			out[i2] = H[j2]*out[i] + G[j2]*out[i2];
			out[i] = tmp;
		}

	} else {
		//only work on the lower part + no need to invert in and tmp
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}
		fftw_one(ps[swifft_iter+1], (fftw_complex*) in+sz2, (fftw_complex*) out+sz2);

		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}

	}

	return;
}

void swifft_haar3(double complex *in, double complex *out, int sz){
	rec_swifft_haar3(in, out, sz, scratch2);
}
