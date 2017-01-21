import numpy as np
import math

# Return a box filter of size n by n
# Requires n to be an odd integer
def boxfilter(n):
  assert(n % 2 == 1), "Dimension must be odd"
  # Create n x n array, every element = n^-2
  return np.full((n, n), pow(n, -2))

# Return a normalized 1D Gaussian filter
# Fix length of filter array to (6 * sigma) rounded up to next odd integer
def gauss1d(sigma):
  # Max distance from center of the array
  # = floor(((6 * sigma) rounded up to next odd integer)/2)
  # = (6 * sigma) rounded up to next integer
  maxDistFromCenter = int(np.ceil(6 * sigma)) / 2
  distsFromCenter = np.arange(-maxDistFromCenter, maxDistFromCenter + 1)
  # unnormalizedFilter[i] = exp(-x^2 / (2*sigma^2)), where x = distance of i from array center
  unnormalizedFilter = np.exp(-pow(distsFromCenter, 2) / (2 * pow(sigma, 2)))
  # Normalize filter so that sum of all elements ~= 1.0
  inverseScalingFactor = unnormalizedFilter.sum()
  return unnormalizedFilter / inverseScalingFactor
