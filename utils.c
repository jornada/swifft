#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#include <fftw.h>

inline double S(double x){
	if (fabs(x)<1e-10) return 0; else return x;
}

//! Creates the test signal
// 0 (default): linear signal (saw tooth)
// 1 : sin
// 2 : sin + spike
// 3 : white noise
// 4 : white noise
void create_signal(complex double *vec, int sz, int kind){
	int i;
	static int last_wn_sz=0;   //last size used for computing white noise
	static double complex *last_wn=NULL; //last white noise
	fftw_plan p;

	switch (kind){
		case 1:
			for (i=0; i<sz; i++) vec[i] = sin(i*M_PI*0.075);
			break;
		case 2:
			srand(0);
			for (i=0; i<sz; i++) vec[i] = sin(i*M_PI*0.075) + (rand()%(2001) - 1000)*1e-4;
			vec[sz>>1]=-10;
			break;
		case 3:
			srand(0);
			for (i=0; i<sz; i++) vec[i] = rand()%2001 - 1000;
			break;
		case 4:
			if (last_wn_sz != sz) {
				if (last_wn) {
					srand(0);
					free(last_wn);
				}
				last_wn = (double complex*) malloc(sizeof(double complex)*sz);

				p = fftw_create_plan(sz, FFTW_FORWARD, FFTW_ESTIMATE);
				//first, use vec as temporary vector to store FFT of white noise
				for (i=0; i<sz; i++) vec[i] = 10 + (rand()%1000)*1e-4;
				//for (i=0; i<sz; i++) last_wn[i] += (rand()%1000)*1000;
				fftw_one(p, (fftw_complex*) vec, (fftw_complex*) last_wn);
				fftw_destroy_plan(p);

				//for (i=0; i<sz; i++) last_wn[i] += (rand()%1000)*1000;

				last_wn_sz = sz;
			}
			memcpy(vec, last_wn, sz*sizeof(double complex));
			break;
		default:
			for (i=0; i<sz; i++) vec[i] = i;
	}
}

void print_times(struct timespec ts0, struct timespec ts1, clock_t c0, clock_t c1){
	printf("\tTime (WALL): %f\tTime (clock): %f\n", \
		(float) (ts1.tv_sec-ts0.tv_sec) + (float) (ts1.tv_nsec -ts0.tv_nsec)/1e9, \
		(float)(c1-c0)/CLOCKS_PER_SEC);
}

double vec_norm(double complex *a, int sz){
	int i;
	double res;

	res=0.;
	for (i=0; i<sz; i++){
		res += creal((a[i])*conj(a[i]));
	}
	return sqrt(res);
}

void norm(double complex *a, int sz){
	int i;
	double nor;

	nor = vec_norm(a, sz);

	for (i=0; i<sz; i++){
		a[i] /= nor;
	}

}

//norm-2 of the diff between 2 vectors
double diff_norm(double complex *a, double complex *b, int sz){
	int i;
	double complex *v, res;

	v = (double complex*) malloc(sz*sizeof(double complex));
	for (i=0; i<sz; i++){
		v[i] = (a[i]-b[i]);///(b[i]);
	}

	res = vec_norm(v, sz);
	free (v);
	//return vec_norm(a, sz);
	return creal(res)/vec_norm(b,sz);
}

void print_vec(double *vec, int sz){
	int i;

	if (sz<10){
		printf("  [ ");
		for (i=0; i<sz; i++) printf("% .4g, ", S(vec[i]));
		printf("\b\b ]\n");
	} else {
		printf("  [\n");
		for (i=0; i<5; i++) printf("    % .4g,\n", S(vec[i]));
		printf("    ...\n");
		for (i=sz-5; i<sz-1; i++) printf("    % .4g,\n", S(vec[i]));
		i=sz-1;
		for (i=sz-5; i<sz; i++) printf("    % .4g\n", S(vec[i]));
		printf("  ]\n");
	}
}

void print_cvec(complex double *vec, int sz){
	int i;

	if (sz<=10){
		printf("  [ ");
		for (i=0; i<sz; i++) printf("% .4g + % .4g I, ", S(creal(vec[i])), S(cimag(vec[i])));
		printf("\b\b ]\n");
	} else {
		printf("  [\n");
		for (i=0; i<3; i++) printf("    % .4g + % .4g I,\n", S(creal(vec[i])), S(cimag(vec[i])));
		printf("    ...\n");
		for (i=sz-3; i<sz-1; i++) printf("    % .4g + % .4g I,\n", S(creal(vec[i])), S(cimag(vec[i])));
		i=sz-1;
		printf("    % .4g + % .4g I\n", S(creal(vec[i])), S(cimag(vec[i])));
		printf("  ]\n");
	}
}

void write_cvec(complex double *vec, int sz, char *fname){
	FILE *fout;
	int i;

	if ( !(fout=fopen(fname, "w")) ){
		printf("Error writting data to fname!\n");
		return;
	}

	for (i=0; i<sz; i++){
		fprintf(fout, "%d %lf %lf\n", i, creal(vec[i]), creal(vec[i]));
	}
	fclose(fout);
}
