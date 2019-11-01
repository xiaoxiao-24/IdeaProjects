import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread('/Users/xiaoxiaorey/Downloads/tiger.png')
fig = plt.figure()
# imgplot = plt.imshow(img)

lum_img = img[:, :, 0]

a = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(lum_img)
a.set_title('Before')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

a = fig.add_subplot(1, 3, 2)
imgplot = plt.imshow(lum_img)
imgplot.set_clim(0.0, 0.5)
a.set_title('After 1')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

a = fig.add_subplot(1, 3, 3)
imgplot = plt.imshow(lum_img)
imgplot.set_clim(0.4, 0.7)
a.set_title('After 2')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

plt.show()
