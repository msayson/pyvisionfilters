import unittest
import visionfilters
import numpy as np

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
    assertGauss1DFilter(self, visionfilters.gauss1d(0.3), 3)
    assertGauss1DFilter(self, visionfilters.gauss1d(1.0), 7)
    assertGauss1DFilter(self, visionfilters.gauss1d(2.0), 13)

# Helper functions

def assertGauss1DFilter(self, numpyArray, expectedLength):
  self.assertEqual(numpyArray.shape[0], expectedLength)
  assertApproxNormalized(self, numpyArray)

def assertApproxNormalized(self, numpyArray):
  self.assertTrue(0.999 <= numpyArray.sum() <= 1.001)

if __name__ == '__main__':
  unittest.main()
