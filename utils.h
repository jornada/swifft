#include <time.h>

void print_times(struct timespec ts0, struct timespec ts1, clock_t c0, clock_t c1);
double diff_norm(double complex *a, double complex *b, int sz);
void print_vec(double *vec, int sz);
void print_cvec(complex double *vec, int sz);
void create_signal(complex double *vec, int sz, int kind);
void write_cvec(complex double *vec, int sz, char *fname);
void load_filters(double *h_, double *g_, int sz, char *fname);
