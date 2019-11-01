import numpy as np
import matplotlib.pyplot as plt

H = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
G = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# create a fig window and set its size : long = 6 and width = 4
fig = plt.figure(figsize=(6, 4))

# divide the original fig into 2*3 subfig and put this one at the 4th place
ax = fig.add_subplot(121)
plt.imshow(H)
ax.set_title('colorMap1')
ax.set_aspect('equal')
plt.colorbar(orientation='vertical')

ax = fig.add_subplot(122)
plt.imshow(G)
ax.set_title('colorMap2')
ax.set_aspect('equal')
plt.colorbar(orientation='vertical')

# cax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
# cax.get_xaxis().set_visible(True)
# cax.get_yaxis().set_visible(True)
# cax.patch.set_alpha(5)
# cax.set_frame_on(True)

plt.show()
