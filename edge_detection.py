# Edge detection using OpenCV
import cv2
import numpy
import matplotlib.pyplot as plt
import parameters as parm

img = cv2.imread("{0}/slide_test.png".format(parm.dir_slides_cropped))
edges = cv2.Canny(img, 50, 100)

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.savefig("{0}/edge_detect_test.png".format(parm.dir_figures))