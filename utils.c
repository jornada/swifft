#include <stdio.h>
#include <complex.h>
#include "utils.h"

void print_vec(double *vec, int sz){
	int i;

	if (sz<10){
		printf("  [ ");
		for (i=0; i<sz; i++) printf("% .4g, ", vec[i]);
		printf("\b\b ]\n");
	} else {
		printf("  [\n");
		for (i=0; i<5; i++) printf("    % .4g,\n", vec[i]);
		printf("    ...\n");
		for (i=sz-5; i<sz-1; i++) printf("    % .4g,\n", vec[i]);
		i=sz-1;
		for (i=sz-5; i<sz; i++) printf("    % .4g\n", vec[i]);
		printf("  ]\n");
	}
}

void print_cvec(complex double *vec, int sz){
	int i;

	if (sz<=10){
		printf("  [ ");
		for (i=0; i<sz; i++) printf("% .4g + % .4g I, ", creal(vec[i]), cimag(vec[i]));
		printf("\b\b ]\n");
	} else {
		printf("  [\n");
		for (i=0; i<3; i++) printf("    % .4g + % .4g I,\n", creal(vec[i]), cimag(vec[i]));
		printf("    ...\n");
		for (i=sz-3; i<sz-1; i++) printf("    % .4g + % .4g I,\n", creal(vec[i]), cimag(vec[i]));
		i=sz-1;
		printf("    % .4g + % .4g I\n", creal(vec[i]), cimag(vec[i]));
		printf("  ]\n");
	}
}
