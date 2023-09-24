#동영상 출력
#프레임의 반복적인 출력으로 동영상처럼 보이게 하는 것

#파일 출력
import cv2

video_path = '/Users/kimdongkyu/dev/Python/2023_2_semester/autonomous_driving/traffic line/traffic lane.mp4'

cap = cv2.VideoCapture(video_path)

# while cap.isOpened(): #파일이 열려있다면
#     ret, frame = cap.read() #ret: 성공 여부, frame: frame
#     if not ret:
#         print('더이상 가져올 프레임이 없습니다.')
#         break

#     cv2.imshow('load', frame)

#     #Q(quit)를 눌러서 중간에 끄기
#     if cv2.waitKey(1) == ord('q'): #waitKey()안의 숫자를 조종하여 영상 속도 조절: 1frame / N ms 의미
#         print('사용자 입력에 의해 종료합니다.')
#         break

# cap.release() #자원 해제
# cv2.destroyAllWindows() #모든 창 닫기 





# 카메라 출력
import cv2
cap = cv2.VideoCapture(0) #0번째 카메라 장치 (Device ID)

if not cap.isOpened(): #카메라가 잘 열리지 않은 경우.
    exit() #종료

while True: #무한 반복.
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('camera', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


