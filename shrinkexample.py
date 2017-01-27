from PIL import Image
import numpy as np
import visionfilters

def shrinkWithSmoothing(imgArray, scaleFactor, sigma):
  # Smooth with a 2D Gaussian filter before subsampling to minimize artifacts
  smoothedImgArray = visionfilters.gaussconvolve2d(imgArray, sigma)
  # Sample every scaleFactor-th row and column to obtain a smaller image
  resizedImgArray = np.array([row[0::scaleFactor] for row in smoothedImgArray[0::scaleFactor]])
  return Image.fromarray(resizedImgArray.astype('uint8'))

def shrinkWithSmoothingAndSave(imgArray, scaleFactor, sigma):
  resizedImg = shrinkWithSmoothing(imgArray, scaleFactor, sigma)
  resizedImg.save('images/output/wheat_shrinkByFactor%d.jpg' % scaleFactor, 'JPEG')

def shrinkWithoutSmoothing(imgArray, scaleFactor):
  resizedImgArray = np.array([row[0::scaleFactor] for row in imgArray[0::scaleFactor]])
  return Image.fromarray(resizedImgArray.astype('uint8'))

img = Image.open('images/wheat.jpg')
img = img.convert('L')     # Convert image to grayscale
imgArray = np.asarray(img) # Convert image to an array

shrinkWithSmoothingAndSave(imgArray, scaleFactor=2, sigma=1.2)
shrinkWithSmoothingAndSave(imgArray, scaleFactor=4, sigma=1.9)
shrinkWithSmoothingAndSave(imgArray, scaleFactor=8, sigma=3.2)

for i in xrange(1,4):
  scaleFactor = pow(2,i)
  resizedImgWithoutSmoothing = shrinkWithoutSmoothing(imgArray, scaleFactor)
  resizedImgWithoutSmoothing.save('images/output/wheat_shrinkByFactor%d_noSmoothing.jpg' % scaleFactor, 'JPEG')
