from PIL import Image
import numpy as np
import visionfilters

def shrinkWithSmoothing(imgArray, scalingFactor):
  # Smooth with a 2D Gaussian filter before subsampling to minimize artifacts
  smoothedImgArray = visionfilters.gaussconvolve2d(imgArray, sigma=pow(2,1-scalingFactor))
  # Sample every (2^scalingFactor)-th pixel to obtain a smaller image
  resizedImgArray = np.array([row[0::pow(2,scalingFactor)] for row in smoothedImgArray[0::pow(2,scalingFactor)]])
  return Image.fromarray(resizedImgArray.astype('uint8'))

def shrinkWithoutSmoothing(imgArray, scalingFactor):
  resizedImgArray = np.array([row[0::pow(2,scalingFactor)] for row in imgArray[0::pow(2,scalingFactor)]])
  return Image.fromarray(resizedImgArray.astype('uint8'))

img = Image.open('images/wheat.jpg')
img = img.convert('L')     # Convert image to grayscale
imgArray = np.asarray(img) # Convert image to an array

for scalingFactor in xrange(1,4):
  resizedImgWithSmoothing = shrinkWithSmoothing(imgArray, scalingFactor)
  resizedImgWithSmoothing.save('images/output/wheat_resized_2PowerNeg%d.jpg' % scalingFactor, 'JPEG')
  resizedImgWithoutSmoothing = shrinkWithoutSmoothing(imgArray, scalingFactor)
  resizedImgWithoutSmoothing.save('images/output/wheat_resized_2PowerNeg%d_noSmoothing.jpg' % scalingFactor, 'JPEG')
