/* fft.c
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
#include <string.h>
#include <math.h>
#include <complex.h>
#include "fft.h"

void rec_fft2(double complex *in, double complex *out, int sz, long long int s);
void rec_fft3(double complex *in, double complex *out, int sz);

double complex *scratch;
double complex *w;
int fft_step;

void prepare_fft(int sz){
	int i;
	double alpha;

	scratch = (double complex*) malloc(sizeof(double complex)*sz);
	w = (double complex*) malloc(sizeof(double complex)*sz);
	alpha = 2.0*M_PI/(double)sz; 

	for (i=0; i<sz; i++){
		w[i] = cos(alpha*i) - I*sin(alpha*i);
	}
	fft_step=1;
}

void free_fft(){
	free(scratch);
	free(w);
}

void fft(double complex *in, double complex *out, int sz){
	int i,j, sz2;
	double complex aa, bb;

	//printf("Caling fft, sz=%lld, fftw_step=%lld\n", sz, fft_step);

	sz2 = sz>>1;
	//reorder
	memcpy(scratch, in, sz*sizeof(double complex));
	for (i=0; i<sz2; i++) {
		in[i] = scratch[2*i];
		in[i+sz2] = scratch[2*i+1];
	}

	if (sz>2){
		fft_step = fft_step<<1;
		fft(in, out, sz2);
		fft(in+sz2, out+sz2, sz2);
		fft_step = fft_step>>1;
	} else {
		out[0] = in[0];
		out[1] = in[1];
	}

	//precalculate 0-element, since w[0]=1
	aa = out[0];
	bb = out[sz2];
	out[0] = aa + bb;
	out[sz2] = aa - bb;
	j=0;
	for (i=1; i<sz2; i++){
		j += fft_step;
		aa = out[i];
		bb = w[j]*out[i+sz2];
		//bb = (M_PI+M_PI*I)*out[i+sz2];
		out[i] = aa + bb;
		out[i+sz2] = aa - bb;
	}
		
}

void fft2(double complex *in, double complex *out, int sz){
	//int i,j, sz2;

	rec_fft2(in, out, sz, 1);
}

void rec_fft2(double complex *in, double complex *out, int sz, long long int s){
	int i,j, sz2;
	double complex aa, bb;

	//printf("Caling fft, sz=%lld, fftw_step=%lld\n", sz, fft_step);
	sz2 = sz>>1;

	if (sz>2){
		fft_step = fft_step<<1;
		rec_fft2(in, out, sz2, 2*s);
		rec_fft2(in+s, out+sz2, sz2, 2*s);
		fft_step = fft_step>>1;
	} else {
		out[0] = in[0];
		out[1] = in[1];
	}

	//precalculate 0-element, since w[0]=1
	aa = out[0];
	bb = out[sz2];
	out[0] = aa + bb;
	out[sz2] = aa - bb;
	j=0;
	for (i=1; i<sz2; i++){
		j += fft_step;
		aa = out[i];
		bb = w[j]*out[i+sz2];
		//bb = (M_PI+M_PI*I)*out[i+sz2];
		out[i] = aa + bb;
		out[i+sz2] = aa - bb;
	}
}

#define SWAP(a,b) tmp=b; b=a; a=tmp;

//source: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=115695
void frbr(double complex *x, int m) {
	int br[256];
	int m2,c,odd,offset,b_size,i,j,k;
	double complex tmp;

	m2=m>>1; c=1<<m2;  
	odd=0; if(m!=m2<<1) odd=1; 
	offset=1<<(m-1); b_size=2;
	br[0]=0;  br[1]=offset; 
	SWAP(x[1], x[offset]); 
	if(odd) SWAP(x[1+c], x[offset+c]); 

	while(b_size<c){
		offset>>=1;
		for(i=b_size; i<b_size<<1; i++){
			br[i]=k=br[i-b_size]+offset; 
			SWAP(x[i], x[k]);
			if(odd) SWAP(x[i+c], x[k+c]); 
			for(j=1; j<i; j++){
				SWAP(x[i+br[j]+c], x[k+j+c]);
			}
		}
		b_size<<=1;
	}
	return;
}

//http://www.katjaas.nl/bitreversal/bitreversal.html
void bitrev(double complex *real, unsigned int logN)
{	
	unsigned int forward, rev, toggle;
	unsigned int nodd, noddrev;  // to hold bitwise negated or odd values
	unsigned int N, halfn, quartn, nmin1;
	double complex temp;
	
	N = 1<<logN;
	halfn = N>>1;    // frequently used 'constants'	
	quartn = N>>2;
	nmin1 = N-1;

	forward = halfn; // variable initialisations
	rev = 1;
	
	while(forward)	// start of bitreversed permutation loop, N/4 iterations
	{
	 
	 // adaptation of the traditional bitreverse update method

	 forward -= 2;									
	 toggle = quartn;  // reset the toggle in every iteration
	 rev ^= toggle;	   // toggle one bit in reversed unconditionally
	 while(rev&toggle) // check if more bits in reversed must be toggled
	 {
		 toggle >>= 1;
		 rev ^= toggle;							
	 }
	
		if(forward<rev)  // swap even and ~even conditionally
		{

			temp = real[forward];				
			real[forward] = real[rev];
			real[rev] = temp;

			nodd = nmin1 ^ forward;	// compute the bitwise negations
			noddrev = nmin1 ^ rev;		
			
			temp = real[nodd];      // swap bitwise-negated pairs
			real[nodd] = real[noddrev];
			real[noddrev] = temp;
		}
		
		nodd = forward ^ 1;  // compute the odd values from the even
		noddrev = rev ^ halfn;
		
		temp = real[nodd];  // swap odd unconditionally
		real[nodd] = real[noddrev];
		real[noddrev] = temp;
	}	
	// end of the bitreverse permutation loop
}
// end of bitrev function

void fft3(double complex *in, double complex *out, int sz){

	bitrev(in, round(log2(sz)));
	rec_fft3(in, out, sz);
}

void rec_fft3(double complex *in, double complex *out, int sz){
	int i,j, sz2;
	double complex aa, bb;

	//printf("Caling fft, sz=%lld, fftw_step=%lld\n", sz, fft_step);

	sz2 = sz>>1;

	if (sz>2) {
		fft_step = fft_step<<1;
		rec_fft3(in, out, sz2);
		rec_fft3(in+sz2, out+sz2, sz2);
		fft_step = fft_step>>1;
	} else {
		out[0] = in[0];
		out[1] = in[1];
	}

	//precalculate 0-element, since w[0]=1
	aa = out[0];
	bb = out[sz2];
	out[0] = aa + bb;
	out[sz2] = aa - bb;
	j=0;
	for (i=1; i<sz2; i++){
		j += fft_step;
		aa = out[i];
		bb = w[j]*out[i+sz2];
		//bb = (M_PI+M_PI*I)*out[i+sz2];
		out[i] = aa + bb;
		out[i+sz2] = aa - bb;
	}
}
