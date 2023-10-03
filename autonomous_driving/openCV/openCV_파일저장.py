#파일 저장

#이미지 저장

import cv2

img_path = '/Users/kimdongkyu/dev/Python/2023_2_semester/autonomous_driving/traffic line/road.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#이것도 안해도 됨
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('img_save.jpg', img) #현재 폴더에 저장됨. jpg,png등 확장자 바꿀 수 있음



#동영상 저장


import cv2

video_path ='/Users/kimdongkyu/dev/Python/2023_2_semester/autonomous_driving/openCV/traffic line/traffic lane.mp4'

cap = cv2.VideoCapture(video_path)

#코덱 정의
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # (*'DIVX') = ('D','I','V','X')

width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #width는 정수값이어야 하기에 round처리
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # /2등으로 사이즈 조절 가능
fps = cap.get(cv2.CAP_PROP_FPS)

                # 저장파일명, 코덱, fps, (너비, 높이)
out = cv2.VideoWriter('vido_save.avi', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    out.write(frame) #영상만 저장.(소리X)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

out.release() #자원 해제
cap.release()
cv2.destroyAllWindows()
