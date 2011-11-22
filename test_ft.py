#!/usr/bin/env python

from __future__ import division
from numpy import *
import numpy.fft as fft

n=3
x=arange(2**n)
print x

#x_ = fft.fftshift(fft.fft(x))
x_ = fft.fft(x)
print x_
