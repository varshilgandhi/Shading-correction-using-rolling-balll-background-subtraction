# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:50:24 2021

@author: abc
"""

import cv2
import numpy as np

#Read our image
img = cv2.imread("Alloy_gradient.jpg", 1)

#Convert our image into lab space image
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#Let's split our lab space image
l , a , b = cv2.split(lab_img)

#Apply CLAHE (Contrast liimited adaptive histogram equalization )  filter
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
CLAHE_img = cv2.merge((clahe_img, a, b))

#Let's define corrected image  and convert it into LAB2BGR
corrected_image = cv2.cvtColor(CLAHE_img, cv2.COLOR_LAB2BGR)

#Let's visualize all images
cv2.imshow("Original image", img)
cv2.imshow("Corrected image", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#####################################################################################

"""

Above shading correction using CLAHE is not that much good so now we doing shading correction using rolling 
ball background subtraction

"""

#Using roll ball background


import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt

#Read our image
img = cv2.imread("Alloy_gradient.jpg", 0)   # 0  indicate image read as gray scale

#define roll ball background subtraction method
radius = 30
final_img , background = subtract_background_rolling_ball(img, radius, light_background=True, 
                                                          use_paraboloid=False, do_presmooth=True)

#Apply clahe(contrast limited adaptive histogram equalization) for better understading 
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
clahe_img = clahe.apply(final_img)


#Let's plot our image to see
cv2.imshow("BackGround image", background)
cv2.imshow("After background subtraction", final_img)
cv2.imshow("After CLAHE", clahe_img)
cv2.waitKey(0)
cv2.destroyAllWindows()





















