#TEXT
#openCV에서 제공하는 글꼴
# 1. cv2.FONT_HERSHEY_SIMPLEX : 보통 크기의 산 세리프(sans-serif) 글꼴
# 2. cv2.FONT_HERSHEY_PLAIN : 작은 크기의 산 세리프(sans-serif) 글꼴
# 3. cv2.FONT_HERSHEY_SCRIPT_SIMPLEX : 필기체 글꼴
# 4. cv2.FONT_HERSHEY_TRIFLEX : 보통 크기의 세리프 글꼴
# 5. cv2.FONT_ITALIC : 기울임. 다른 글꼴과 함께 사용

import cv2
import numpy as np

img = np.zeros((480,640,3),dtype=np.uint8)


SCALE = 1 #크기
COLOR = (255,255,255) #white
THICKNESS = 1

        #그릴 위치, 내용, 시작 위치, 폰트, 스케일, 색, 두께
cv2.putText(img,"FONT_HERSHEY_COMPLEX", (20,50),cv2.FONT_HERSHEY_COMPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img,"FONT_HERSHEY_PLAIN", (20,150),cv2.FONT_HERSHEY_PLAIN, SCALE, COLOR, THICKNESS)
cv2.putText(img,"FONT_HERSHEY_SCRIPT_COMPLEX", (20,250),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img,"FONT_HERSHEY_TRIPLEX", (20,350),cv2.FONT_HERSHEY_TRIPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img,"FONT_ITALIC", (20,450),cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC, SCALE, COLOR, THICKNESS)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#한글 우회 방법
import cv2
import numpy as np

img = np.zeros((480,640,3),dtype=np.uint8)



COLOR = (255,255,255) #white


#PIL (Python Image Library)
from PIL import ImageFont, ImageDraw, Image

def myKoreanTxt(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc', font_size)
    draw.text(pos, text, font = font, fill = font_color)
    return np.array(img_pil)

FONT_SIZE = 30
img = myKoreanTxt(img, '코딩 잘하고싶다....', (20,50), FONT_SIZE, COLOR)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

