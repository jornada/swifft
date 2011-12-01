CC=gcc
#CCFLAGS=-Wall -O0 -g -ggdb
CCFLAGS=-Wall -O3 -ffast-math
#CCFLAGS=-Wall -O3 
LDFLAGS=-lm -lrt -lfftw

#profiling: -pg, run normally, use gprof later

#CC=icc
#CCFLAGS=-Wall -O3 -fast -opt-jump-tables=large
#LDFLAGS=-lm -lrt -lfftw

all: test_fft test_wavelets
#wavelet

clean:
	rm -f *.o test_fft test_wavelets

test_fft: fft.o test_fft.o utils.o
	$(CC) -o test_fft test_fft.o fft.o utils.o $(LDFLAGS)

test_fft.o: test_fft.c
	$(CC) $(CCFLAGS) -c test_fft.c

fft.o: fft.c
	$(CC) $(CCFLAGS) -c fft.c

utils.o: utils.c
	$(CC) $(CCFLAGS) -c utils.c

test_wavelets: test_wavelets.o wavelets.o utils.o fft.o
	$(CC) -o test_wavelets test_wavelets.o wavelets.o utils.o fft.o $(LDFLAGS)

test_wavelets.o: test_wavelets.c
	$(CC) $(CCFLAGS) -c test_wavelets.c 

wavelets.o: wavelets.c
	$(CC) $(CCFLAGS) -c wavelets.c 

