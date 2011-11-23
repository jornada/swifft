#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "utils.h"

inline double S(double x){
	if (fabs(x)<1e-10) return 0; else return x;
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
