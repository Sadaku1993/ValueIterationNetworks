import cv2
import numpy as np
import math
import random

size = (100, 100)

im = np.zeros(size, dtype=np.int8)

new_im = im.copy()
rand_rad = int(math.ceil(random.random() * 1))
randx = int(math.ceil(random.random() * size[0]))
randy = int(math.ceil(random.random() * size[1]))
cv2.circle(new_im, (randx, randy), rand_rad, 1, -1)

cv2.imshow("img", new_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow("test", cv2.resize(255 - new_im * 255, (300, 300), interpolation=cv2.INTER_NEAREST))
# cv2.waitKey(1000)
