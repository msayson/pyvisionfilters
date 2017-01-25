from PIL import Image
import numpy as np
import visionfilters

def shrinkWithSmoothing(imgArray, scalingFactor, sigma):
  # Smooth with a 2D Gaussian filter before subsampling to minimize artifacts
  smoothedImgArray = visionfilters.gaussconvolve2d(imgArray, sigma)
  # Sample every (2^scalingFactor)-th pixel to obtain a smaller image
  resizedImgArray = np.array([row[0::pow(2,scalingFactor)] for row in smoothedImgArray[0::pow(2,scalingFactor)]])
  return Image.fromarray(resizedImgArray.astype('uint8'))

def shrinkWithSmoothingAndSave(imgArray, scalingFactor, sigma):
  resizedImg = shrinkWithSmoothing(imgArray, scalingFactor, sigma)
  resizedImg.save('images/output/wheat_resized_2PowerNeg%d.jpg' % scalingFactor, 'JPEG')

def shrinkWithoutSmoothing(imgArray, scalingFactor):
  resizedImgArray = np.array([row[0::pow(2,scalingFactor)] for row in imgArray[0::pow(2,scalingFactor)]])
  return Image.fromarray(resizedImgArray.astype('uint8'))

img = Image.open('images/wheat.jpg')
img = img.convert('L')     # Convert image to grayscale
imgArray = np.asarray(img) # Convert image to an array

shrinkWithSmoothingAndSave(imgArray, scalingFactor=1, sigma=1.2)
shrinkWithSmoothingAndSave(imgArray, scalingFactor=2, sigma=1.9)
shrinkWithSmoothingAndSave(imgArray, scalingFactor=3, sigma=3.2)

for scalingFactor in xrange(1,4):
  resizedImgWithoutSmoothing = shrinkWithoutSmoothing(imgArray, scalingFactor)
  resizedImgWithoutSmoothing.save('images/output/wheat_resized_2PowerNeg%d_noSmoothing.jpg' % scalingFactor, 'JPEG')
