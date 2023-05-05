import numpy as np
import math
import cv2
import utils

final_cord=np.load('final_cord.npy')
print(final_cord)
print(final_cord.shape)
left_rect_cord = final_cord[:4]
print("left_rect_cord {}".format(left_rect_cord))

right_rect_cord = final_cord[4:8]
print("right_rect_cord {}".format(right_rect_cord))

middle_rect_cord = final_cord[8:12]
print("middle_rect_cord {}".format(middle_rect_cord))

v1 = utils.compute_vanishing_point([left_rect_cord[0], left_rect_cord[1], left_rect_cord[2], left_rect_cord[3]])
v2 = utils.compute_vanishing_point([right_rect_cord[0], right_rect_cord[1], right_rect_cord[2], right_rect_cord[3]])
v3 = utils.compute_vanishing_point([middle_rect_cord[0], middle_rect_cord[1], middle_rect_cord[2], middle_rect_cord[3]])

print("v1: {}".format(v1))
print("v2: {}".format(v2))
print("v3: {}".format(v3))

# Compute the camera matrix
vanishing_points = [v1, v2, v3]

K = utils.compute_K_from_vanishing_points(vanishing_points)
print("Intrinsic Matrix: {} \n".format(K))


