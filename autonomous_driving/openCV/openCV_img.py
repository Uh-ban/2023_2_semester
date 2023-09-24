
#OpenCV는 Open Source Computer Vision Library의 약어로 오픈소스 컴퓨터 비전 라이브러리입니다.
#실시간 영상처리에 사용

import cv2
print(cv2.__version__)

# 이미지 파일 경로 설정
img_path = '/Users/kimdongkyu/dev/Python/2023_2_semester/autonomous_driving/road.jpg'

# 이미지 읽기
img = cv2.imread(img_path)

# 창의 이름 설정
window_name = 'traffic_lane'

# # 이미지 표시
# cv2.imshow(window_name, img)

# # 사용자가 키를 누를 때까지 대기
# #0은 무한전 5000은 5초 ms 단위
# cv2.waitKey(5000) 

# # 모든 창 닫기
# cv2.destroyAllWindows()



#읽기 옵션
# import cv2

#1. cv2.IMREAD_COLOR: 기본값으로 컬러이미지. png 같은 경우 투명 영역 있는데 투명 영역 무시

# img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
# cv2.imshow('color img', img_color)

# #2. cv2.imread_grayscale: 흑백

# img_grayscale = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('gray img', img_grayscale)

# #3. cv2.imread_unchanged: png의 경우, 투명영역까지 포함

# img_unchanged = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
# cv2.imshow('unchanged img', img_unchanged)

# cv2.waitKey(0)
# cv2.destroyAllWindows() 




##SHAPE
# information of image height, width, channel

import cv2
img = cv2.imread(img_path)
print(img.shape) #(427, 640, 3)

