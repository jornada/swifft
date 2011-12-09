#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#include <fftw.h>
#include <assert.h>

inline double S(double x){
	if (fabs(x)<1e-10) return 0; else return x;
}

//! Creates the test signal
// 0 (default): random
// 1 : lin. comb. of exp's
// 2 : exp + noise
// 3 : sinc
// 4 : gaussian
// 5 : audio
void create_signal(complex double *vec, int sz, int kind){
	int i, sz2;
	double tmp;
	double *buf;
	FILE *fin;

	sz2 = sz>>1;
	switch (kind){
		case 1:
			// sum of exp's -- tiled version
			for (i=0; i<sz; i++) vec[i] = cexp(I*i*M_PI*0.05) + exp(I*i*M_PI*0.075);
			break;
		case 2:
			// exp + noise -- tiled version
			srand(0);
			for (i=0; i<sz; i++) vec[i] = cexp(I*i*M_PI*0.05) + (rand()%(2001) - 1000)*1e-4;
			break;
		case 3:
			// sinc -- rescaled version
			tmp = 100./(double) sz;
			for (i=0; i<sz; i++) vec[i] = sin(tmp*(i-sz+1e-4)) / (tmp*(i-sz+1e-4));
			break;
		case 4:
			// gaussian -- rescalled version
			for (i=0; i<sz; i++) vec[i] = exp(-tmp*pow(i-sz2,2));
			break;
		case 5:
			// audio -- fixed length
			assert(sizeof(double)==8);
			fin = fopen("tests/music.dat", "rb");
			if (!fin) {
				printf("Can't open tests/music.dat. Make sure the file is there!\n");
				return;
			}
			buf = malloc( (1<<17)*sizeof(double) );
			//jump initial noisy part
			fread(buf, sizeof(double), (1<<12) , fin);
			fread(buf, sizeof(double), (1<<17) , fin);
			fclose(fin);
			for (i=0; i<sz; i++) vec[i] = (double complex) buf[i%(1<<17)];
			free(buf);
			break;
		default:
			srand(0);
			for (i=0; i<sz; i++) vec[i] = rand()%2001 - 1000;
			break;
	}
}

void print_times(struct timespec ts0, struct timespec ts1, clock_t c0, clock_t c1){
	/*
	printf("\tTime (WALL):    %lf\tTime (clock):    %lf\n", \
		(double) (ts1.tv_sec-ts0.tv_sec) + (double) (ts1.tv_nsec -ts0.tv_nsec)/1e9, \
		(double)(c1-c0)/CLOCKS_PER_SEC);
	*/
	printf("\tTime (WALL):    %lf\n", \
		(double) (ts1.tv_sec-ts0.tv_sec) + (double) (ts1.tv_nsec -ts0.tv_nsec)/1e9);
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
	double complex *v;
	double res;

	v = (double complex*) malloc(sz*sizeof(double complex));
	for (i=0; i<sz; i++){
		v[i] = (a[i]-b[i]);///(b[i]);
	}

	res = vec_norm(v, sz);
	free (v);
	//return vec_norm(a, sz);
	return res/vec_norm(b,sz);
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

void load_filters(double *h_, double *g_, int sz, char *fname){
	FILE *fin;
	int sz_h, sz_g, i;

	memset(h_, 0, sz*sizeof(double));
	memset(g_, 0, sz*sizeof(double));

	fin = fopen(fname, "r");
	if (!fin){
		printf("ERROR: could not open wavelet filter from file '%s'\n", fname);
		return;
	}
	// get size of h filter
	fscanf(fin, "%d", &sz_h);
	for (i=0; i<sz_h; i++){
		if (feof(fin)){
			printf("ERROR: read past end of file at '%s'\n", fname);
			break;
		}
		fscanf(fin, "%lf", h_ + i);
	}
	// get size of g filter
	fscanf(fin, "%d", &sz_g);
	for (i=0; i<sz_g; i++){
		if (feof(fin)){
			printf("ERROR: read past end of file at '%s'\n", fname);
			break;
		}
		fscanf(fin, "%lf", g_ + i);
	}
	fclose(fin);

}
