CC=gcc
#CCFLAGS=-Wall -O0 -g -ggdb
CCFLAGS=-Wall -O3 -ffast-math -mtune=core2 
#CCFLAGS=-Wall #-O3 
LDFLAGS=-lm -lfftw

#CC=icc
#CCFLAGS=-Wall -O3 -fast
#LDFLAGS=-lfftw

all: fft 
#wavelet

clean:
	rm *.o fft

fft: fft.o
	$(CC) $(LDFLAGS) -o fft fft.o -lm

fft.o: fft.c
	$(CC) $(CCFLAGS) -c fft.c

wavelet: wavelet.o
	$(CC) $(LDFLAGS) -c wavelet.c 


