import numpy as np
import math

# Return a box filter of size n by n
def boxfilter(n):
  assert(n % 2 == 1), "Dimension must be odd"
  val = pow(n, -2) # All filter elements have value n^-2
  row = [val for x in range(n)] # Represent each row as an array
  return np.array([row for x in range(n)])

# Return normalized 1D Gaussian filter
def gauss1d(sigma):
  # Length of filter = 6*sigma rounded up to next odd integer
  lenFilter = int(np.ceil(6 * sigma))
  lenFilter = lenFilter if (lenFilter % 2 == 1) else lenFilter + 1
  distFromCenter = arrayOfDistancesFromCenter(lenFilter)
  # unnormalizedFilter[i] = exp(-x^2 / (2*sigma^2)), where x = distance of i from array center
  unnormalizedFilter = np.exp(-pow(distFromCenter, 2) / (2 * pow(sigma, 2)))
  # Normalize filter so that sum of all elements ~= 1.0
  inverseScalingFactor = unnormalizedFilter.sum()
  return unnormalizedFilter / inverseScalingFactor

# Return n-element array of each element's distance from the center of the array
# Requirement: n must be a positive, odd integer
def arrayOfDistancesFromCenter(n):
  assert(n > 0 and n % 2 == 1), "Array length must be positive and odd: %d" % n
  return np.array([x - n / 2 for x in range(n)])
