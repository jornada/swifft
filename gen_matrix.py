#!/usr/bin/env python

'''Generates the matrix associated to the full wavelet transform.
Currently it can only use Haar wavelets.'''

from __future__ import division
from numpy import *

N=1<<6
W=eye(N)


def get_Haar(m):
	h = zeros(m)
	g = zeros(m)
	h[:2] = array([1,-1])*sqrt(2)/2
	g[:2] = array([1, 1])*sqrt(2)/2
	W=empty((m,m))
	#build 'forward' matrix
	for i in range(m//2):
		W[i] = roll(h, 2*i)
		W[i+m//2] = roll(g, 2*i)
	return W
	
#print get_Haar(8)

M = eye(1)
n = 2
while n<=N:
	#rescale old matrix
	tmp = eye(n)
	tmp[:n//2, :n//2] = M[:,:]
	M = dot(tmp, get_Haar(n))
	n = n*2

#print M
import matplotlib.pylab as plt
plt.matshow(M)
plt.show()
#savetxt('tmp.dat',M,fmt='%.3f')
