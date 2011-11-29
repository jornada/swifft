void prepare_swifft(int sz, double *h_, int h_sz_, double *g_, int g_sz_);
void free_swifft();
void swifft_gen(double complex *in, double complex *out, int sz);
void swifft_haar(double complex *in, double complex *out, int sz);
