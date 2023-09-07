import pandas as pd
import numpy as np

print("autonomous mobility")

# [Python] OpenCV로 만든 자율주행 코드 

import numpy as np
import cv2
import time

from Steering import *
from Preprocessing import *
from StopDetector import *

WIDTH  = 640
HEIGHT = 360

kernel_size=11

low_threshold=120
high_threshold=255

theta=np.pi/180

lower_blue = (115-30, 10, 10)
upper_blue = (115+30, 255, 255)

lower_red = (6-6, 30, 30)
upper_red = (6+4, 255, 255)

lower_yellow = (19-4, 30, 30)
upper_yellow = (19+30, 255, 255)

isUseableRed=False
isInStopArea=False

def setup_path():
    path = "./source/KakaoTalk_20221007_063507621.mp4"
    cap=cv2.VideoCapture(path) #path
    return cap

def setup_countours():
    obj_b = cv2.imread('./source/corn_data/lavacorn_nb.png', cv2.IMREAD_GRAYSCALE)#wad
    obj_s = cv2.imread('./source/corn_data/lavacorn_ns.png', cv2.IMREAD_GRAYSCALE)#wad
    obj_contours_b,_=cv2.findContours(obj_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#wad
    obj_contours_s,_=cv2.findContours(obj_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#wad
    obj_pts_b=obj_contours_b[0]#wad
    obj_pts_s=obj_contours_s[0]#wad

    return obj_pts_b, obj_pts_s

def setup_linear_reg():
    global p_r_m
    global p_r_n
    global p_l_m
    global p_l_n

    p_r_m=0.3
    p_r_n=37
    p_l_m=-0.3
    p_l_n=238

def depart_points(img, points):
    right_line=[]
    left_line=[]
    stop_line=[]

    for p in points:
        x1, y1, x2, y2 = p
        label = plot_one_box([x1, y1, x2, y2], img)

        x = int((x1 + x2)/2)
        y = int(y2)

        if label == 'blue':
            left_line.append([x, y])
        elif label == 'yellow':
            right_line.append([x, y])
        elif label == 'red':
            stop_line.append([x, y])
        else:
            pass

    return right_line, left_line, stop_line

def find_contours(img_thresh, obj_pts_b, obj_pts_s)->list:
    contours,_=cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    isSameCorn=False
    for pts in contours:
        if cv2.contourArea(pts) <60:
            continue
        rc=cv2.boundingRect(pts)
        dist_b=cv2.matchShapes(obj_pts_b, pts, cv2.CONTOURS_MATCH_I3, 0)
        dist_s=cv2.matchShapes(obj_pts_s, pts, cv2.CONTOURS_MATCH_I3, 0)
        if dist_b <0.5 or dist_s<0.4:
            mid_x = (2*rc[0]+rc[2])/2
            for p in points:
                if p[0]<=mid_x and p[2]>=mid_x:
                    isSameCorn=True
                    break
            if not isSameCorn and 40<=mid_x<=600:
                cv2.rectangle(img_thresh, rc, (255, 0,0),1)
                cv2.imshow("img", img_thresh)
                points.append([rc[0], rc[1], rc[0]+rc[2], rc[1]+rc[3]])
            isSameCorn=False

    return points

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask=np.zeros_like(img)
    
    if len(img.shape)>2:
        channel_count = img.shape[2]
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color=255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image=cv2.bitwise_and(img,mask)
    return masked_image

def preprocessing(img, low_threshold, high_threshold, kernel_size): #640*360
    
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    if len(img.shape)>2:
        channel_count=img.shape[2]
        ignore_mask_color=(255,) * channel_count
    else:
        ignore_mask_color=255

    mask=np.zeros_like(img)
    vertices=np.array([[(20, 315),
                        (20, 210),
                       (160, 130),
                       (470, 130),
                        (620, 210),
                       (620, 315)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image=region_of_interest(img_hsv, vertices)

    lower_blue = (110-3, 120, 150) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_blue = (110+5, 255, 255)

    lower_red = (6-6, 110, 120) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_red = (6+4, 255, 255)

    lower_yellow = (19-1, 110, 120) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_yellow = (19+5, 255, 255)

    mask_hsv_red = cv2.inRange(masked_image, lower_red, upper_red)
    mask_hsv_blue = cv2.inRange(masked_image, lower_blue, upper_blue)
    mask_hsv_yellow = cv2.inRange(masked_image, lower_yellow, upper_yellow)

    mask_hsv=cv2.bitwise_or(mask_hsv_red, mask_hsv_blue)
    mask_hsv=cv2.bitwise_or(mask_hsv, mask_hsv_yellow)
    
    stop_img = cv2.bitwise_and(img, img, mask=mask_hsv)
    # img_gray=grayscale(mask_hsv)
    img_blur = gaussian_blur(mask_hsv, kernel_size)
    ret, img_thresh=threshold(img_blur, low_threshold, high_threshold)
    return img_thresh