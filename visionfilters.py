import numpy as np
import math
from scipy import signal

# Return a box filter of size n by n
# Requires n to be an odd integer
def boxfilter(n):
  assert(n % 2 == 1), "Dimension must be odd"
  # Create n x n array, every element = n^-2
  return np.full((n, n), pow(n, -2))

# Return a 1D Gaussian filter
# Fix length of filter array to (6 * sigma) rounded up to next odd integer
def gauss1d(sigma):
  # Max distance from center of the array
  # = floor(((6 * sigma) rounded up to next odd integer)/2)
  # = (6 * sigma) rounded up to next integer
  maxDistFromCenter = int(np.ceil(6 * sigma)) / 2
  distsFromCenter = np.arange(-maxDistFromCenter, maxDistFromCenter + 1)
  # unnormalizedFilter[i] = exp(-x^2 / (2*sigma^2)), where x = distance of i from array center
  unnormalizedFilter = np.exp(-pow(distsFromCenter, 2) / (2 * pow(sigma, 2)))
  return normalize(unnormalizedFilter)

# Return a 2D Gaussian filter
# Fix side lengths to (6 * sigma) rounded up to next odd integer
def gauss2d(sigma):
  gauss1dArray = gauss1d(sigma)
  # The 2D Gaussian filter is the convolution of the 1D Gaussian and its transpose
  reshapedTo2D = gauss1dArray.reshape(1,gauss1dArray.size) # Reshape to allow us to transpose
  unnormalizedFilter = signal.convolve2d(reshapedTo2D, reshapedTo2D.transpose())
  return normalize(unnormalizedFilter)

# Return the result of applying a 2D Gaussian filter convolution to an input array
def gaussconvolve2d(array, sigma):
  gaussFilter = gauss2d(sigma)
  return signal.convolve2d(array, gaussFilter, 'same')

# Normalize an array so that its elements sum to approximately 1.00
def normalize(numpyArray):
  return numpyArray / numpyArray.sum()
