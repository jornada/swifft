CC=gcc
CCFLAGS=-Wall -O0 -g -ggdb
#CCFLAGS=-Wall -O3 -ffast-math 
#CCFLAGS=-Wall #-O3 
LDFLAGS=-lm -lfftw -lrfftw

#CC=icc
#CCFLAGS=-Wall -O3 -fast
#LDFLAGS=-lfftw

all: test_fft test_wavelets
#wavelet

clean:
	rm *.o test_fft

test_fft: fft.o test_fft.o utils.o
	$(CC) $(LDFLAGS) -o test_fft test_fft.o fft.o utils.o

test_fft.o: test_fft.c
	$(CC) $(CCFLAGS) -c test_fft.c

fft.o: fft.c
	$(CC) $(CCFLAGS) -c fft.c

utils.o: utils.c
	$(CC) $(CCFLAGS) -c utils.c

test_wavelets: test_wavelets.o wavelets.o utils.o
	$(CC) $(LDFLAGS) -o test_wavelets test_wavelets.o wavelets.o utils.o

wavelets.o: wavelets.c
	$(CC) $(CCFLAGS) -c wavelets.c 


