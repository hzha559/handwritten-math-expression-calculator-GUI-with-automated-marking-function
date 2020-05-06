import cv2
import numpy as np
from kivy.graphics.texture import Texture

########## RUN TIME FUNCTIONS ##########
# Function to find and draw contours - uses opencv functions findContours() and drawContours()
def DrawContours(net_img, dicom_img):
    img = dicom_img.copy()
    # draw blue dashed lines
    ret, threshHuman = cv2.threshold(net_img, 249, 255, 0)
    contours, _ = cv2.findContours(threshHuman, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 3)# this affect the actual color of drawing
    return img

def RenderDisplayImage(img):
    img = cv2.flip(img, 0)
    texture = Texture.create(size=(1021, 576), colorfmt="bgr")
    s = img.flatten().astype(np.uint8)
    texture.blit_buffer(s, pos=(0, 0), size=texture.size, bufferfmt="ubyte", colorfmt="bgr")
    return texture

def CreateDisplayImage(dicom, label):
    img = DrawContours(label, dicom)
    # img = MergeImages(dicom, img, 5)
    return img

