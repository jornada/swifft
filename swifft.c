/* swifft.c
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

//! Use low communication version of generic wavelet transform
#define LOW_COMM_GEN

//! Use low communication version of Haar wavelet transform
#define LOW_COMM_HAAR

//! Do all wavelet transforms before doing any FFT (optimized for cache!)
#define PRE_WAVELET

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw.h>
#include "swifft.h"
#include "utils.h"

//! Temporary memory buffer
double complex *swifft_scratch;
//! High and low freqs filters
double *h, *g;
//! FFT of h and g
double complex *H, *G;
//! Filters` size
int h_sz, g_sz, max_window;
//! Size of original signal, and sz/2
int swifft_sz, swifft_sz_half;
//! Internal variables that keep track of the swifft iteration
int swifft_shift, swifft_iter;
//! Classical FFT twiddle factor
double complex *w2;
//! FFTW plans, used by swifft after pruning is over
fftw_plan *ps, p;
//! Depth of pruning scheme. The meaning of this depends on the algorithm
int swifft_depth;

#define SQRT2_2 0.7071067811865475 
#define max(a,b) (a>b?a:b)

//! Used to be called by swifft routines after pruning is over
void full_fft(double complex *in, double complex *out, int sz){
	fftw_plan p_;

	p_ = fftw_create_plan(sz, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_one(p_, (fftw_complex*) in, (fftw_complex*) out);
	fftw_destroy_plan(p_);
}

//! Called by swifft routines when the PRE_WAVELET flag is on.
//! Tailored for Haar filters.
void trunc_DWT_gen(double complex *vec, int sz_){
	int iter;
	int i;
	int sz, sz2;
	double complex *in;
	int i_start, buf_window, j, j2, i2;

	sz2 = sz_;
	in = vec;
	for (iter=0; iter < swifft_depth; iter++){
		sz = sz2;
		sz2 >>= 1;
		//low communication version
		i_start = sz2 - max_window + 1;
		//copy what will be needed later
		buf_window = 2*(max_window - 2);
		i = sz - buf_window;
		memcpy(swifft_scratch, in+i, buf_window*sizeof(double complex));
		memcpy(swifft_scratch+buf_window, in, (max_window-2)*sizeof(double complex));
		//do wavelet transform on the easy part
		for (i=sz2 - max_window + 1; i+1; i--){
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (i<<1) + j;
				in[i+sz2] += g[j]*in[j2];
			}
		}
		//deal with the 3*(max_window - 2) "harder" points
		i2=-2;
		for (i=sz2 - max_window + 2; i<sz2; i++){
			in[i+sz2]=0;
			i2+=2;
			for (j=0; j<max_window; j++){
				in[i+sz2] += g[j]*swifft_scratch[j+i2];
			}
		}
		in += sz2;
	}
}

//! Called by swifft routines when the PRE_WAVELET flag is on.
//! Tailored for Haar filters.
void trunc_DWT_haar(double complex *vec, int sz_){
	int i;

#ifndef LOW_COMM_HAAR
	int iter;
	int sz2;
	double complex *in;

	sz2 = sz_;
	in = vec;
	for (iter=0; iter < swifft_depth; iter++){
		sz2 >>= 1;
		//Do wavelet transform only at the lower part, i.e., discard
		//detail coeffs. Note that we don`t need any "scratch memory".
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}
		in += sz2;
	}
#else
	int i2, i_min, len0;
	int j;
	double complex tmp;
	double fact;

	fact = pow(SQRT2_2, swifft_depth);
	i_min = sz_ - (sz_ >> swifft_depth);
	len0 = 1 << swifft_depth;
	i2 = sz_-1 + len0;
	for (i=sz_-1; i >= i_min; i--){
		i2 -= len0;
		tmp = 0;
		for (j=0; j<len0; j++) {
			tmp += vec[i2 - j];
		}
		vec[i] = fact * tmp;
	}
#endif
}

//! Called by swifft routines when the PRE_WAVELET flag is on.
//! Tailored for non-orthogonal Haar filters.
void trunc_DWT_haar_non_orthog(double complex *vec, int sz_){
	int i;

#ifndef LOW_COMM_HAAR
	int iter;
	int sz2;
	double complex *in;

	sz2 = sz_;
	in = vec;
	for (iter=0; iter < swifft_depth; iter++){
		sz2 >>= 1;
		//Do wavelet transform only at the lower part, i.e., discard
		//detail coeffs. Note that we don`t need any "scratch memory".
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = in[2*i] + in[2*i+1];
		}
		in += sz2;
	}
#else
	int i2, i_min, len0;
	int j;
	double complex tmp;

	i_min = sz_ - (sz_ >> swifft_depth);
	len0 = 1 << swifft_depth;
	i2 = sz_-1 + len0;
	for (i=sz_-1; i >= i_min; i--){
		i2 -= len0;
		tmp = 0;
		for (j=0; j<len0; j++) {
			tmp += vec[i2 - j];
		}
		vec[i] = tmp;
	}
#endif
}

//! "Naive" FFT specialized for real sparse data - takes only O(sz_out*sz_in) flops
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

//! This routine must be called before performing the actual swifft
//! \param sz size of input vector
//! \param h_ high frequency wavelet filter
//! \param h_sz_ length of h_
//! \param g_ low frequency wavelet filter
//! \param g_sz_ length of g_
//! \param depth pruning depth. The meaning of this depends of the swifft algorithm
void prepare_swifft(int sz, double *h_, int h_sz_, double *g_, int g_sz_, int depth){
	double alpha;
	int i, sz2;

	swifft_scratch = (double complex*) malloc(sizeof(double complex)*sz);
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

	swifft_sz = sz;
	swifft_sz_half = sz>>1;
	H = (double complex*) calloc(sz, sizeof(double complex));
	G = (double complex*) calloc(sz, sizeof(double complex));

	h = h_;	h_sz = h_sz_;
	g = g_;	g_sz = g_sz_;
	max_window = max(h_sz, g_sz);

	sparse_FFT(h, h_sz, H, sz);
	sparse_FFT(g, g_sz, G, sz);

	swifft_shift=1;
	swifft_iter=0;
	swifft_depth = depth;
}

void free_swifft(){
	free(swifft_scratch);
	free(w2);
	free(H);
	free(G);
}


/*****************************************************************************/

//! Called internally by swifft_full
void rec_swifft_full(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2;
	double complex tmp;

	//printf("Calling swifft, sz=%lld, swifftw_step=%lld\n", sz, swifft_shift);
	sz2 = sz>>1;

	//wavelet transform	
	memcpy(swifft_scratch, in, sz*sizeof(double complex));
	for (i=0; i<sz2; i++) {
		in[i]=0;
		in[i+sz2]=0;
		for (j=0; j<max_window; j++){
			j2 = (2*i + j)%sz;
			in[i] += h[j]*swifft_scratch[j2];
			in[i+sz2] += g[j]*swifft_scratch[j2];
		}
	}
	
	swifft_shift <<= 1;
	if (sz>2){
		rec_swifft_full(in, out, sz2);
		rec_swifft_full(in+sz2, out+sz2, sz2);
	} else {
		out[0] = in[0];
		out[1] = in[1];
	}
	swifft_shift >>= 1;

	//multiply by H and G
	j = -swifft_shift;
	for (i=0; i<sz2; i++){
		j += swifft_shift;
		j2 = j + swifft_sz_half;
		i2 = i + sz2;
		tmp = H[j]*out[i] + G[j]*out[i2];
		out[i2] = H[j2]*out[i] + G[j2]*out[i2];
		out[i] = tmp;
	}
}

//! Implements a full Fourier Transform via Wavelet transform without any
//! pruning. Useful for pedagogical reasons only.
void swifft_full(double complex *in, double complex *out){
	rec_swifft_full(in, out, swifft_sz);
}


/*****************************************************************************/


//! Called internally by swifft_gen1
void rec_swifft_gen1(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2;
	int i_start, buf_window;
	double complex tmp;

	//printf("Calling swifft, sz=%lld, swifftw_step=%lld\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter >= swifft_depth){
		//stop pruning, do regular FFT using FFTW.
		fftw_one(ps[swifft_iter], (fftw_complex*) in, (fftw_complex*) out);
		
	} else {
		//perform wavelet transform
#ifndef PRE_WAVELET
#ifdef LOW_COMM_GEN
		//low communication version
		i_start = sz2 - max_window + 1;
		//copy what will be needed later
		buf_window = 2*(max_window - 2);
		i = sz - buf_window;
		memcpy(swifft_scratch, in+i, buf_window*sizeof(double complex));
		memcpy(swifft_scratch+buf_window, in, (max_window-2)*sizeof(double complex));
		//do wavelet transform on the easy part
		for (i=sz2 - max_window + 1; i+1; i--) {
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (i<<1) + j;
				in[i+sz2] += g[j]*in[j2];
			}
		}
		//deal with the 3*(max_window - 2) "harder" points
		i2=-2;
		for (i=sz2 - max_window + 2; i<sz2; i++){
			in[i+sz2]=0;
			i2+=2;
			for (j=0; j<max_window; j++){
				in[i+sz2] += g[j]*swifft_scratch[j+i2];
			}
		}
#else
		//naive version
		memcpy(swifft_scratch, in, sz*sizeof(double complex));
		//separates the wavelet transform into two parts: (a) one loop 
		// where the filter lie continuously between [1,N], and (b) one
		// loop where there is a gap in the filter coeffs. This way, we
		// can save time by note computing the % operation all the time
		i2 = sz2 - ((max_window-1)>>1);
		for (i=0; i<i2; i++) {
			//in[i]=0; //detail coeffs will be thrown away
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (i<<1) + j;
				//in[i] += h[j]*swifft_scratch[j2];
				in[i+sz2] += g[j]*swifft_scratch[j2];
			}
		}
		for (i=i2; i<sz2; i++) {
			//in[i]=0; //detail coeffs will be thrown away
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = ((i<<1) + j)%sz;
				//in[i] += h[j]*swifft_scratch[j2];
				in[i+sz2] += g[j]*swifft_scratch[j2];
			}
		}
#endif
#endif
		//Recursively call swifft_gen1
		swifft_shift <<= 1;
		swifft_iter++;
		//rec_swifft_gen1(in, out, sz2); //neglect details coeffs
		rec_swifft_gen1(in+sz2, out+sz2, sz2);
		swifft_shift >>= 1;
		swifft_iter--;

		//Multiply by the FFT of the wavelet coeffs.
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + swifft_sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
	}

}

//! This routine takes any input filter.
void swifft_gen1(double complex *in, double complex *out){
#ifdef PRE_WAVELET
	trunc_DWT_gen(in, swifft_sz);
#endif
	rec_swifft_gen1(in, out, swifft_sz);
}


/*****************************************************************************/


//! Called internally by swifft_haar1
void rec_swifft_haar1(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2, sz_tmp;

	//printf("Calling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter >= swifft_depth){
		//stop pruning, do regular FFT using FFTW.
		fftw_one(ps[swifft_iter], (fftw_complex*) in, (fftw_complex*) out);
	} else {
#ifndef PRE_WAVELET
		//Do wavelet transform only at the lower part, i.e., discard
		//detail coeffs. Note that we don`t need any "scratch memory".
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}
#endif

		//Recursively call swifft_haar1
		swifft_shift <<= 1;
		swifft_iter++;
		//rec_swifft_haar1(in, out, sz2); //neglect details coeffs
		rec_swifft_haar1(in+sz2, out+sz2, sz2);
		swifft_shift >>= 1;
		swifft_iter--;

		//Multiply by the FFT of the wavelet coeffs.
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + swifft_sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
		
	}
	return;
}

//! This routine is specialized for Haar-type filter, and prunes all the detail coeffs.
//! This is equivalent to doing the FFT on a under-sampled signal, then resample it
void swifft_haar1(double complex *in, double complex *out){
#ifdef PRE_WAVELET
	trunc_DWT_haar(in, swifft_sz);
#endif
	rec_swifft_haar1(in, out, swifft_sz);
}


/*****************************************************************************/


//! Called internally by swifft_haar1_non_orthog
void rec_swifft_haar1_non_orthog(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2;

	//printf("Calling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter >= swifft_depth){
		//stop pruning, do regular FFT using FFTW.
		fftw_one(ps[swifft_iter], (fftw_complex*) in, (fftw_complex*) out);
	} else {
#ifndef PRE_WAVELET
		//Do wavelet transform only at the lower part, i.e., discard
		//detail coeffs. Note that we don`t need any "scratch memory".
		
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = in[2*i] + in[2*i+1];
		}
#endif

		//Recursively call swifft_haar, but ignore lower part of the decomposition
		swifft_shift <<= 1;
		swifft_iter++;
		//rec_swifft_haar_non_orthog(in, out, sz2); //neglect details coeffs
		rec_swifft_haar1_non_orthog(in+sz2, out+sz2, sz2);
		swifft_shift >>= 1;
		swifft_iter--;

		//Multiply by the FFT of the wavelet coeffs.
		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + swifft_sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}
	}
	return;
}

//! Same as swifft_haar1, but uses non-orthogonal Haar filter. See example
//! for the type of filter that you should input.
//! This routine is slightly faster, since there is no multiplication
//! when doing the wavelet transform.
void swifft_haar1_non_orthog(double complex *in, double complex *out){
#ifdef PRE_WAVELET
	trunc_DWT_haar_non_orthog(in, swifft_sz);
#endif
	rec_swifft_haar1_non_orthog(in, out, swifft_sz);
}


/*****************************************************************************/


//! Called internally by swifft_haar2
void rec_swifft_haar2(double complex *in, double complex *out, int sz){
	int i, i2, j,j2, sz2, sz4;
	double complex tmp;

	//printf("Calling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter < swifft_depth){
		//only save half of the input into a scratch memory
		memcpy(swifft_scratch, in, sz2*sizeof(double complex));
		sz4 = sz2>>1;
		//work on the second half (up and down at the same time)
		for (i=sz2-1; i>=sz4; i--) {
			in[i] = SQRT2_2*( in[2*i] - in[2*i+1] );
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}
		//and now finish the first half
		for (i=sz4-1; i>=0; i--) {
			in[i] = SQRT2_2*( swifft_scratch[2*i] - swifft_scratch[2*i+1] );
			in[i+sz2] = SQRT2_2*( swifft_scratch[2*i] + swifft_scratch[2*i+1] );
		}
		
		swifft_shift <<= 1;
		swifft_iter++;
		fftw_one(ps[swifft_iter], (fftw_complex*) in+sz2, (fftw_complex*) out+sz2);
		rec_swifft_haar2(in, out, sz2);
		swifft_shift >>= 1;
		swifft_iter--;

		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + swifft_sz_half;
			i2 = i + sz2;
			tmp = H[j]*out[i] + G[j]*out[i2];
			out[i2] = H[j2]*out[i] + G[j2]*out[i2];
			out[i] = tmp;
		}

	} else {
		//only work on the lower part
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}
		fftw_one(ps[swifft_iter+1], (fftw_complex*) in+sz2, (fftw_complex*) out+sz2);

		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + swifft_sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}

	}

	return;
}

//! This routine uses a less aggressive pruning scheme to approximate the
//!  FFT, and it is specialized for Haar filter.
void swifft_haar2(double complex *in, double complex *out){
	rec_swifft_haar2(in, out, swifft_sz);
}


/*****************************************************************************/


//! Called internally by swifft_haar2
void rec_swifft_haar2_op(double complex *in, double complex *out, int sz, double complex *in_buf){
	int i, i2, j,j2, sz2;
	double complex tmp;

	//printf("Calling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter < swifft_depth){
		//invert in and in_buf!

		for (i=sz2-1; i+1; i--) {
			in_buf[i] = SQRT2_2*( in[2*i] - in[2*i+1] );
			in_buf[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}

		swifft_shift = swifft_shift<<1;
		swifft_iter++;
		fftw_one(ps[swifft_iter], (fftw_complex*) in_buf+sz2, (fftw_complex*) out+sz2);
		rec_swifft_haar2_op(in_buf, out, sz2, in);
		swifft_shift = swifft_shift>>1;
		swifft_iter--;

		j = -swifft_shift;
		for (i=0; i<sz2; i++){
			j += swifft_shift;
			j2 = j + swifft_sz_half;
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
			j2 = j + swifft_sz_half;
			i2 = i + sz2;
			out[i] = G[j]*out[i2];
			out[i2] = G[j2]*out[i2];
		}

	}

	return;
}

//! Same as swifft_haar2, but out-of-place version
void swifft_haar2_op(double complex *in, double complex *out){
	rec_swifft_haar2_op(in, out, swifft_sz, swifft_scratch);
}
