
#OpenCV는 Open Source Computer Vision Library의 약어로 오픈소스 컴퓨터 비전 라이브러리입니다.
#실시간 영상처리에 사용

import cv2
print(cv2.__version__)
import numpy as np

# 이미지 파일 경로 설정
img_path = '/Users/kimdongkyu/dev/Python/2023_2_semester/autonomous_driving/traffic lane.png'

# 이미지 읽기
img = cv2.imread(img_path)

# 창의 이름 설정
window_name = 'traffic_lane'

# 이미지 표시
cv2.imshow(window_name, img)

# 사용자가 키를 누를 때까지 대기
#0은 무한전 5000은 5초 ms 단위
key = cv2.waitKey(5000) 
print(key)

# 모든 창 닫기
cv2.destroyAllWindows()
