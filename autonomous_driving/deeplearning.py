# TensorFlow 및 필요한 모듈 가져오기
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Sequential 모델 생성: 순차적인 레이어를 순서대로 쌓는 모델
model = Sequential()

# Conv2D 레이어 추가: 컨볼루션 레이어는 이미지 처리에 주로 사용되며, 2D 필터를 통해 특징을 추출합니다.
# 32개의 필터, 3x3 크기의 커널, 활성화 함수는 ReLU 사용, 입력 이미지 크기는 (128, 128, 3)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))

# Flatten 레이어 추가: 다차원 배열을 1차원으로 평탄화합니다.
model.add(Flatten())

# Dense 레이어 추가: fully connected 레이어, 128개의 뉴런과 ReLU 활성화 함수 사용
model.add(Dense(128, activation='relu'))

# Output 레이어 추가: 출력 뉴런 1개, 활성화 함수로 tanh (쌍곡 탄젠트) 사용
model.add(Dense(1, activation='tanh'))

# 모델 컴파일하기
# Optimizer로 'adam'을 사용하여 모델을 학습시키며, 손실 함수로 'mse' (Mean Squared Error)를 사용
model.compile(optimizer='adam', loss='mse')

#학습하기
# 데이터 로드하기
X_train, y_train = load_data()

# 모델 학습하기
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 자율주행 시험하기
def autonomous_driving(image):
    # 이미지 전처리하기
    preprocessed_image = preprocess(image)

    # 모델로부터 조향값 예측하기
    steering_angle = model.predict(preprocessed_image)

    # 자동차를 조향하기
    control_car(steering_angle)