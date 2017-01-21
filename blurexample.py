from PIL import Image
import numpy as np
import visionfilters

# Original image from http://www.gratisography.com/pictures/324_1.jpg
# Photograph by Ryan McGuire
# Resized to 720x480px for the purpose of this demo
img = Image.open('images/wheat.jpg')

# Convert image to grayscale
img = img.convert('L')
img.save('images/output/wheat_grayscale.jpg', 'JPEG')

# Blur image by applying a Gaussian 2D filter convolution

# Convert image to an array to apply the filter convolution
imgArray = np.asarray(img)
blurredImgArray = visionfilters.gaussconvolve2d(imgArray, sigma=1.5)

# Convert to uint8, then to an Image
blurredImg = Image.fromarray(blurredImgArray.astype('uint8'))
blurredImg.save('images/output/wheat_grayscale_blurred.jpg', 'JPEG')
