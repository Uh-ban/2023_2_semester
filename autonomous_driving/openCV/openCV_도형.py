
##import numpy as np

# 세로 480, 가로 640, 3 channel (RGB)에 해당하는 스케치북 만들기
img = np.zeros((480,640,3), dtype=np.uint8) #이미지 모든 공간 검은색0으로 나옴
img[:] = (255,255,155) #모든 공간 흰색으로 채우기. (B,G,R) 순서


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영역의 일부 칠하기

img = np.zeros((480,640,3), dtype=np.uint8) #이미지 모든 공간 검은색0으로 나옴
img[100:200, 200:300] = (255,255,255) #세로는 100-200, 가로는 200-300범위를 흰색으로

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#########
#  직선  #
#########

#cv2.LINE_4(상하좌우 4방향으로 연결된 선)

#cv2.LINE_4(대각선을 포함하여 8방향으로 연결된 선)(기본값)

#cv2.LINE_AA(부드러운 선 anti-aliasing)


import cv2
import numpy as np

img = np.zeros((480,640,3), dtype = np.uint8)

COLOR  =  (0,255,255) #(B,G,R) : yellow 
THICKNESS = 3 #두께
 
cv2.line(img,(50, 100), (400, 50), COLOR, THICKNESS, cv2.LINE_8) 
cv2.line(img,(50, 200), (400, 150), COLOR, THICKNESS, cv2.LINE_4) 
cv2.line(img,(50, 300), (400, 250), COLOR, THICKNESS, cv2.LINE_AA) 
        #그릴 위치, 시작점, 끝점, 색, 두께, 선종류

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#########
#  원   #
#########

import cv2
import numpy as np

img = np.zeros((480,640,3), dtype = np.uint8)

COLOR  =  (255,255,0) #(B,G,R) : 옥색
RADIUS = 50 # 반지름
THICKNESS = 10 #두께

cv2.circle(img,(240,320),RADIUS,COLOR,THICKNESS, cv2.LINE_AA) #속이 빈
cv2.circle(img,(400,320),RADIUS,COLOR,cv2.FILLED, cv2.LINE_AA) #속이 꽉 찬
        #그릴 위치, 중심, 반지름, 색, 두께, 선종류
 
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#########
# 사각형 #
#########

import cv2
import numpy as np

img = np.zeros((480,640,3), dtype = np.uint8)

COLOR  =  (0,255,0) #(B,G,R) : green
THICKNESS = 3 #두께

cv2.rectangle(img, (100,100),(200,200),COLOR,THICKNESS) #속이 빈
cv2.rectangle(img, (400,100),(500,200),COLOR,cv2.FILLED) #속이 꽉 찬
            #그릴 위치, 왼쪽 위, 오른쪽 아래, 색, 두께

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#########
# 다각형 #
#########

import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 0, 255)  # (B, G, R): 빨간색
THICKNESS = 3  # 두께

pts1 = np.array([[100, 100], [200, 100], [100, 200]])
pts2 = np.array([[200, 100], [300, 100], [300, 200]])

# cv2.polylines(img, [pts1], True, COLOR, THICKNESS, cv2.LINE_AA)  # 만든 배열을 리스트에 감싸서 넣기. True, False: 마지막 점과 끝점 잇기
# cv2.polylines(img, [pts2], True, COLOR, THICKNESS, cv2.LINE_AA)  # 만든 배열을 리스트에 감싸서 넣기. True, False: 마지막 점과 끝점 잇기

cv2.polylines(img, [pts1, pts2], True, COLOR, THICKNESS, cv2.LINE_AA)  # 만든 배열을 리스트(여러값 넣기 가능)에 감싸서 넣기. True, False: 마지막 점과 끝점 잇기

#속이 찬 다각형
pts3 = np.array([[[100,300],[200,300],[100,400]],[[200,300],[300,300],[300,400]]])
cv2.fillPoly(img, pts3, COLOR,cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
