#동영상을 640x480 크기로 변환한다.
#원근 뒤틀기(Perspective wrap)을 이용하여 차선을 일직선으로 편다.
#동영상을 64x48로 변환한다.
#흰색 차선을 검출하기 위해 HLS 필터링을 적용한다.
#동영상을 회색으로 변환하고, threshold와 가우스 블러를 적용한다.
#동영상을 640x480으로 변환한다.
#히스토그램을 사용하여 흰색 픽셀의 개수가 가장 많은 곳을 측정한다.
#흰색 픽셀의 개수에 따라 움직이는 window를 구현한다.

import cv2
import numpy as np
import matplotlib.pyplot as plt

minpix = 5

lane_bin_th = 145

path = "../source/2022-07-04_16-10-12.mp4" #동영상 주소

fps=30.
mp4_width=640
mp4_height=480
codec=cv2.VideoWriter_fourcc(*'DIVX')

path_birdView="../output/bird_view.mp4"
path_processing="../output/processing.mp4"
path_result="../output/result.mp4"

mp4_birdView=cv2.VideoWriter(path_birdView, codec, fps, (mp4_width, mp4_height))
mp4_processing=cv2.VideoWriter(path_processing, codec, fps, (mp4_width, mp4_height), False) #iscolor=False
mp4_result=cv2.VideoWriter(path_result, codec, fps, (mp4_width, mp4_height))

def warp_process_image(lane, current):
    nwindows=10
    margin = 50
    global minpix
    global lane_bin_th

    leftx_current, rightx_current = current

    window_height = 48
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []
    
    lx, ly, rx, ry = [], [], [], []

    out_img = lane
    out_img = np.dstack((lane, lane, lane))*255

    for window in range(nwindows):

        win_yl = lane.shape[0] - (window+1)*window_height
        win_yh = lane.shape[0] - window*window_height

        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin

        cv2.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2) 

        good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))

        lx.append(leftx_current)
        ly.append((win_yl + win_yh)/2)

        rx.append(rightx_current)
        ry.append((win_yl + win_yh)/2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    #left_fit = np.polyfit(nz[0][left_lane_inds], nz[1][left_lane_inds], 2)
    #right_fit = np.polyfit(nz[0][right_lane_inds] , nz[1][right_lane_inds], 2)
    
    lfit = np.polyfit(np.array(ly),np.array(lx),2)
    rfit = np.polyfit(np.array(ry),np.array(rx),2)

    out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
    out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
    
    #return left_fit, right_fit
    return out_img, lfit, rfit

cap=cv2.VideoCapture(path)

while cap.isOpened(): # cap 정상동작 확인
    ret, image = cap.read()
    # 프레임이 올바르게 읽히면 ret은 True
    if not ret:
        print("프레임을 수신할 수 없습니다. 종료 중 ...")
        break
    
    image = cv2.resize(image, dsize=(640, 480))
    #카메라 영상 촬영후 좌표조정하기
    pts1 = np.float32([[250,240], [320,240], [50,350], [600,350]])
    pts2 = np.float32([[0,0], [640,0], [0,480], [640,480]])
    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    # 원근 변환 적용
    result = cv2.warpPerspective(image, mtrx, (640, 480))
    
    smallimg=cv2.resize(result, dsize=(64, 48))

    hsl=cv2.cvtColor(smallimg, cv2.COLOR_BGR2HLS)
    imgH,imgS,imgL=cv2.split(hsl)

    imgG=cv2.cvtColor(hsl, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', imgG)

    ret, thr=cv2.threshold(imgG, 127, 255, cv2.THRESH_OTSU)
    
    blur = cv2.GaussianBlur(thr, (3,3),0)

    imgBig=cv2.resize(blur, dsize=(640,480))
    
    hist=imgBig.sum(axis=0)
    plt.plot(hist)

    current = np.argmax(hist[:320]), (np.argmax(hist[320:])+320)
    #print(current)

    out_img, lfit, rfit=warp_process_image(imgBig, current)
    
    #cv2.imshow('origin', image)
    cv2.imshow('bird_view', result)    
    cv2.imshow('processing', imgBig)
    cv2.imshow('result', out_img)

    mp4_birdView.write(result)
    mp4_processing.write(imgBig)
    mp4_result.write(out_img)
    
    if cv2.waitKey(27) == ord('q'):
        break
# 작업 완료 후 해제
cap.release()
mp4_birdView.release()
mp4_processing.release()
mp4_result.release()
cv2.destroyAllWindows()