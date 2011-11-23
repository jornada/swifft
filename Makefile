CC=gcc
#CCFLAGS=-Wall -O0 -g -ggdb
CCFLAGS=-Wall -O3 -ffast-math 
#CCFLAGS=-Wall #-O3 
LDFLAGS=-lm -lfftw

#CC=icc
#CCFLAGS=-Wall -O3 -fast
#LDFLAGS=-lfftw

all: test_fft 
#wavelet

clean:
	rm *.o test_fft

test_fft: fft.o test_fft.o utils.o
	$(CC) $(LDFLAGS) -o test_fft test_fft.o fft.o utils.o -lm

test_fft.o: test_fft.c
	$(CC) $(CCFLAGS) -c test_fft.c

fft.o: fft.c
	$(CC) $(CCFLAGS) -c fft.c

utils.o: utils.c
	$(CC) $(CCFLAGS) -c utils.c

wavelet: wavelet.o
	$(CC) $(LDFLAGS) -c wavelet.c 


