import os
import json
import cv2
import numpy as np
from openpyxl import load_workbook


from kivy.graphics.texture import Texture


########## RUN TIME FUNCTIONS ##########

# Function to create parameters for PNG writing
def SetPNGCompression(level):
    png_params = list()
    png_params.append(cv2.IMWRITE_PNG_COMPRESSION)
    png_params.append(level)  # compression value 0 (no comp) - 9 (highest)
    return png_params

# Function to find and draw contours - uses opencv functions findContours() and drawContours()
def DrawContours(net_img, dicom_img):
    img = dicom_img.copy()
    # draw blue for human
    ret, threshHuman = cv2.threshold(net_img, 249, 255, 0)
    _, contours, _ = cv2.findContours(threshHuman, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, contours, -1, (255, 113, 17), 2)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 5)# this affect the actual color of drawing
    # # draw orange for machine
    # ret, threshMachine = cv2.threshold(net_img, 254, 255, 0)
    # _, contours, _ = cv2.findContours(threshMachine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (17, 113, 255), 2)

    return img

# Function to merge two images horizontally separated by vertical white line of thickness "gap"
def MergeImages(leftimage, rightimage, gap):
    height = leftimage.shape[0]
    width = leftimage.shape[1]
    comp_img = np.zeros((height, (width * 2) + gap, 3))  # Height,Width,Channels
    comp_img[:, 0:height, :] = leftimage
    comp_img[:, height:height + gap, :] = 255
    comp_img[:, height + gap:, :] = rightimage
    return comp_img

def CreateDisplayImage(dicom, label):
    img = DrawContours(label, dicom)
    # img = MergeImages(dicom, img, 5)
    return img

def RenderDisplayImage(img):
    img = cv2.flip(img, 0)
    texture = Texture.create(size=(1021, 576), colorfmt="bgr")
    s = img.flatten().astype(np.uint8)
    texture.blit_buffer(s, pos=(0, 0), size=texture.size, bufferfmt="ubyte", colorfmt="bgr")
    return texture

def SaveScanLabel(settings, acc, key, label):
    path = os.path.join(settings['annotationPath'],str(settings['deviceID']),acc,'ScanLabel_'+acc+'.json')
    contents = {}
    if os.path.isfile(path):
        with open(path,'r') as f:
            contents = json.load(f)
    contents[key] = label
    with open(path,'w') as f:
        json.dump(contents,f)

def SaveSliceLabel(settings, acc, slice, key, label):
    path = os.path.join(settings['annotationPath'],str(settings['deviceID']),acc,'SliceLabel_'+acc+'_'+str(slice)+'.json')
    contents = {}
    if os.path.isfile(path):
        with open(path,'r') as f:
            contents = json.load(f)
    contents[key] = label
    with open(path,'w') as f:
        json.dump(contents,f)
