
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw.h>
#include "wavelets.h"
#include "utils.h"

//! Temporary memory buffer
double complex *swifft_scratch;
//! High and low freqs filters
double *h, *g;
//! FFT of h and g
double complex *H, *G;
//! Filters` size
int h_sz, g_sz, max_window;
//! Size of original signal/2
int sz_half;
//! Internal variables that keep track of the swifft iteration
int swifft_shift, swifft_iter;
//! Classical FFT twiddle factor
double complex *w2;
//! FFTW plans, used by swifft after prunning is over
fftw_plan *ps, p;

#define SQRT2_2 0.7071067811865475 
#define max(a,b) (a>b?a:b)

//! Called by swifft routines after prunning is over
void full_fft(double complex *in, double complex *out, int sz){
	fftw_plan p_;

	p_ = fftw_create_plan(sz, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_one(p_, (fftw_complex*) in, (fftw_complex*) out);
	fftw_destroy_plan(p_);
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
void prepare_swifft(int sz, double *h_, int h_sz_, double *g_, int g_sz_){
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

	sz_half = sz>>1;
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
}

void free_swifft(){
	free(swifft_scratch);
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
		memcpy(swifft_scratch, in, sz*sizeof(double complex));
		for (i=0; i<sz2; i++) {
			in[i]=0;
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (2*i + j)%sz;
				//in[i] += h[j]*swifft_scratch[j2];
				//in[i+sz2] += g[j]*swifft_scratch[j2];
				in[i] += h[j]*swifft_scratch[j2];
				in[i+sz2] += g[j]*swifft_scratch[j2];
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
		
		memcpy(swifft_scratch, in, sz*sizeof(double complex));
		for (i=0; i<sz2; i++) {
			//in[i]=0;
			(*
			in[i+sz2]=0;
			for (j=0; j<max_window; j++){
				j2 = (2*i + j)%sz;
				//j2=0;
				//in[i] += h[j]*swifft_scratch[j2];
				in[i+sz2] += g[j]*swifft_scratch[j2];
			}
			*)
			in[i+sz2] = SQRT2_2*( swifft_scratch[2*i] + swifft_scratch[2*i+1] );
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

//! This routine takes any input filter.
void swifft_gen1(double complex *in, double complex *out, int sz, int depth){
	int i, i2, j,j2, sz2;
	double complex tmp;

	//printf("Caling swifft, sz=%lld, swifftw_step=%lld\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter >= depth){
		//stop prunning, do regular FFT using FFTW.
		fftw_one(ps[swifft_iter], (fftw_complex*) in, (fftw_complex*) out);
		
	} else {
		//wavelet transform
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

		//Recursivelly call swifft_gen1
		swifft_shift <<= 1;
		swifft_iter++;
		//swifft_gen1(in, out, sz2, depth); //neglect details coeffs
		swifft_gen1(in+sz2, out+sz2, sz2, depth);
		swifft_shift >>= 1;
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

}

//! This routine is specialized for Haar-type filter, and prunes all the detail coeffs.
//! This is equivalent to doing the FFT on a under-sampled signal, then resample it
void swifft_haar1(double complex *in, double complex *out, int sz, int depth){
	int i, i2, j,j2, sz2, sz_tmp;

	//printf("Caling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter >= depth){
		//stop prunning, do regular FFT using FFTW.
		fftw_one(ps[swifft_iter], (fftw_complex*) in, (fftw_complex*) out);
	} else {
		//Do wavelet transform only at the lower part, i.e., discart
		//detail coeffs. Note that we don`t need any "scratch memory".
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}

		//Recursivelly call swifft_haar1
		swifft_shift <<= 1;
		swifft_iter++;
		//swifft_haar1(in, out, sz2, depth); //neglect details coeffs
		swifft_haar1(in+sz2, out+sz2, sz2, depth);
		swifft_shift >>= 1;
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

//! Same as swifft_haar1, but uses non-orthogonal Haar filter. See example
//! for the type of filter that you should input.
//! This routine is slightly faster, since there is no multiplication
//! when doing the wavelet transform.
void swifft_haar1_non_orthog(double complex *in, double complex *out, int sz, int depth){
	int i, i2, j,j2, sz2;

	//printf("Caling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter >= depth){
		//stop prunning, do regular FFT using FFTW.
		fftw_one(ps[swifft_iter], (fftw_complex*) in, (fftw_complex*) out);
	} else {
		//Do wavelet transform only at the lower part, i.e., discart
		//detail coeffs. Note that we don`t need any "scratch memory".
		
		for (i=sz2-1; i+1; i--) {
			in[i+sz2] = in[2*i] + in[2*i+1];
		}

		//Recursivelly call swifft_haar, but ignore lower part of the decomposition
		swifft_shift <<= 1;
		swifft_iter++;
		//swifft_haar(in, out, sz2, depth); //neglect details coeffs
		swifft_haar1_non_orthog(in+sz2, out+sz2, sz2, depth);
		swifft_shift >>= 1;
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


//! This routine uses a less agressive prunning scheme to approximate the
//!  FFT, and it is specialized for Haar filter.
void swifft_haar2(double complex *in, double complex *out, int sz, int depth){
	int i, i2, j,j2, sz2, sz4;
	double complex tmp;

	//printf("Caling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter < depth){
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
		swifft_haar2(in, out, sz2, depth);
		swifft_shift >>= 1;
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
		//only work on the lower part
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

//! Called internally by swifft_haar2
void rec_swifft_haar2_op(double complex *in, double complex *out, int sz, double complex *in_buf, int depth){
	int i, i2, j,j2, sz2;
	double complex tmp;

	//printf("Caling swifft, sz=%d, swifftw_step=%d\n", sz, swifft_shift);
	sz2 = sz>>1;

	if (swifft_iter < depth){
		//invert in and in_buf!

		for (i=sz2-1; i+1; i--) {
			in_buf[i] = SQRT2_2*( in[2*i] - in[2*i+1] );
			in_buf[i+sz2] = SQRT2_2*( in[2*i] + in[2*i+1] );
		}

		swifft_shift = swifft_shift<<1;
		swifft_iter++;
		fftw_one(ps[swifft_iter], (fftw_complex*) in_buf+sz2, (fftw_complex*) out+sz2);
		rec_swifft_haar2_op(in_buf, out, sz2, in, depth);
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

//! Same as swifft_haar2, but out-of-place version
void swifft_haar2_op(double complex *in, double complex *out, int sz, int depth){
	rec_swifft_haar2_op(in, out, sz, swifft_scratch, depth);
}
