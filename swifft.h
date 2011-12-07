void prepare_swifft(int sz, double *h_, int h_sz_, double *g_, int g_sz_, int depth);
void free_swifft();
void swifft_gen1(double complex *in, double complex *out, int sz);
void swifft_haar1(double complex *in, double complex *out, int sz);
void swifft_haar1_non_orthog(double complex *in, double complex *out, int sz);
void swifft_haar2(double complex *in, double complex *out, int sz);
void swifft_haar2_op(double complex *in, double complex *out, int sz);
