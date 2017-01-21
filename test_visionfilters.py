import unittest
import visionfilters
import numpy as np
from scipy import signal

class TestVisionFilters(unittest.TestCase):

  def test_boxfilter_nEq1(self):
    boxfilter = visionfilters.boxfilter(1)
    expectedFilter = np.array([[1]])
    self.assertEqual(boxfilter, expectedFilter)

  def test_boxfilter_nEq5(self):
    boxfilter = visionfilters.boxfilter(5)
    expectedFilter = np.array([
      [ 0.04,  0.04,  0.04,  0.04,  0.04],
      [ 0.04,  0.04,  0.04,  0.04,  0.04],
      [ 0.04,  0.04,  0.04,  0.04,  0.04],
      [ 0.04,  0.04,  0.04,  0.04,  0.04],
      [ 0.04,  0.04,  0.04,  0.04,  0.04]])
    self.assertTrue(np.array_equal(boxfilter, expectedFilter))

  def test_boxfilter_nMustBeOdd(self):
    with self.assertRaisesRegexp(AssertionError, '^Dimension must be odd$'):
      visionfilters.boxfilter(102)

  def test_gauss1d(self):
    gauss1dFilterSigma1 = visionfilters.gauss1d(1.0)
    expectedFilter = visionfilters.normalize(signal.gaussian(7, 1.0))
    self.assertTrue((gauss1dFilterSigma1 == expectedFilter).all())

  def test_gauss2d(self):
    assertGauss2DFilter(self, visionfilters.gauss2d(1.0), 7)
    assertGauss2DFilter(self, visionfilters.gauss2d(2.0), 13)

# Helper functions

def assertGauss2DFilter(self, numpyArray, expectedSize):
  self.assertEqual(numpyArray.shape, (expectedSize, expectedSize))
  self.assertTrue((np.flipud(np.fliplr(numpyArray)) == numpyArray).all()) # Should be symmetric
  assertApproxNormalized(self, numpyArray)

def assertApproxNormalized(self, numpyArray):
  self.assertTrue(0.999 <= numpyArray.sum() <= 1.001)

if __name__ == '__main__':
  unittest.main()
