from PIL import Image
import numpy as np
import visionfilters

img = Image.open('images/wheat.jpg')
img = img.convert('L')     # Convert image to grayscale
imgArray = np.asarray(img) # Convert image to an array

for x in xrange(1,5):
  # Smooth with a 2D Gaussian filter before subsampling to minimize artifacts
  smoothedImgArray = visionfilters.gaussconvolve2d(imgArray, sigma=pow(2,-x))
  # Sample every (2^x)-th pixel to obtain a smaller image
  resizedImgArray = np.array([row[0::pow(2,x)] for row in smoothedImgArray[0::pow(2,x)]])
  # Convert to uint8, then to an Image
  resizedImg = Image.fromarray(resizedImgArray.astype('uint8'))
  resizedImg.save('images/output/wheat_resizedBy2ToPowerOfNeg%d.jpg' % x, 'JPEG')
