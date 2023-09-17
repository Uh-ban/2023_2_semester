# 이미지 분류 CNN
# 데이터의 차원(표의 열에서의 관점)
# N개의 열을 N차원의 그래프로 변환 가능
# 거리 방향 등 상관관계를 파악하기 좋음

#데이터의 차원(포함관계 관점)
import numpy as np

x1 = np.array([5,3,1,2])
print(x1.ndim, x1.shape) # 1 (4,): 1차원 형태이고 행렬(4,1)
img1 = np.array([[0,255],
                 [255,0]])
print(img1.ndim,img1.shape)
print(img1)

d1 = np.array([1,2,3])
d2 = np.array([d1,d1,d1,d1,d1])
d3 = np.array([d2,d2,d2,d2])
#배열의 깊이 = 차원수
print(d1.ndim,d1.shape)
print(d2.ndim,d2.shape)
print(d3.ndim,d3.shape)

#MNIST(흑백 손글씨)

# 컴퓨터에게 손글씨는 숫자의 집합
# 숫자는 검정은 0 밝을수록 큰 수로 표현 
# 이런 숫자셋이 500장 있고 한 숫자당 28X28로 있다면 그 데이터의 형태는 (500,28,28)

# 컬러 이미지
# RGB의 농도에 대해 각각 다른 숫자가 들어감 즉 각 점이 3개의 숫자를 가짐
# 이런 그림셋이 3072장 있고 그림당 32X32가 있다면 (3072,32,32,3)

# 한 사진의 픽셀이 2448*3262라면
# 2448*3264*3 = 23,970,816의 숫자 있음
# 고로 컴퓨터는 이 전체를 활용하지 않고 더 효율적으로 학습해야함

# #MNIST
# (독립,종속),_=tf.keras.datasets.mnist.load_data()
# print(독립.shape,종속.shape)

# #CIFAR10
# (독립,종속),_=tf.keras.datasets.cifar10.load_data()
# print(독립.shape,종속.shape)

#TensorFlow에서 제공하는 데이터로 MNIST, CIFAR10 연습
import tensorflow as tf

#MNIST
(mnist_x,mnist_y), _= tf.keras.datasets.mnist.load_data()
print(mnist_x.shape,mnist_y.shape)
# (60000, 28, 28) (60000,)

#CIFAR10
(cifar_x,cifar_y), _= tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape,cifar_y.shape)
# (50000, 32, 32, 3) (50000, 1)

#이미지 출력해보기
import matplotlib.pyplot as plt
print(mnist_y[0:3])
plt.imshow(mnist_x[2],cmap='gray')

print(cifar_y[0:10]) #답이 숫자로 나와서 뭔지 모르겠다고 하면 cifar10 category 검색하면 됨
plt.imshow(cifar_x[7])

#Flatten Layer를 활용한 이미지 학습
#Flatten Layer:인공신경망 내에서 reshape 해주는 함수

import tensorflow as tf
import pandas as pd

#reshape을 이용한 것

#데이터 준비
(독립,종속),_ = tf.keras.datasets.mnist.load_data()
print(독립.shape, 종속.shape)
#독립,종속 모두 표의 형태로 만들기
독립 = 독립.reshape(60000,784)
종속 = pd.get_dummies(종속) #원핫인코딩
print(독립.shape, 종속.shape) #(60000, 784) (60000, 10)

#모델 생성
X = tf.keras.layers.Input(shape = [784])
H = tf.keras.layers.Dense(84, activation='swish')(X)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss = 'categorical_crossentropy', metrics='accuracy')

#모델 학습
model.fit(독립,종속,epochs=10)

#모델 이용
print(종속[0:5])
pred = model.predict(독립[0:5])
#소수점 둘째 자리까지 끊어서 데이터 프레임으로 보이기
pd.DataFrame(pred).round(2)

#Flatten Layer 이용(reshape을 데이터 준비 과정에서 하지 않음)
(독립,종속),_ = tf.keras.datasets.mnist.load_data()
print(독립.shape, 종속.shape)

종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)

#모델 생성
X = tf.keras.layers.Input(shape = [28,28])
H = tf.keras.layers.Flatten()(X)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss = 'categorical_crossentropy', metrics='accuracy')

#모델 학습
model.fit(독립, 종속, epochs=5)

#Conv2d
#Convolution: 특정한 패턴의 특징이 어디서 나타나는지 확인하는 도구

#데이터 준비
(독립,종속),_ = tf.keras.datasets.mnist.load_data()
독립 = 독립.reshape(60000,28,28,1) # 컬러 이미지가 RGB3차원이기에 그에 맞게 3차원 형태로 reshape
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)

#모델의 구조
X = tf.keras.layers.Input(shape=[28,28,1]) #데이터 형태 3차원으로 설정
H = tf.keras.layers.Conv2D(3, kernel_size=5, activation='swish')(X) #3개의 특징맵 = 3채널의 특징맵
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(H) #6개의 특징맵 = 6채널의 특징맵
#3,6은 필터셋의 개수, kernel_size는 그 필터의 사이즈(5는 5by5픽셀)
H = tf.keras.layers.Flatten()(H) #표형태로 만들기 
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activaation='softmax')(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss='scategorical_crossentropy', metrics='accuracy')

#Filter. Conv2D를 이해하는 데에 중요함

#filter set은 3차원 형태로 된 가중치 모음
#필터셋 하나는 앞선 레이어의 결과인 "특징맵"전체를 본다
#필터셋의 개수만큼 특징맵을 만든다
#Conv2D(3, kernel_size=5, activation='swish')를 예로 들면
#필터가 3개(F1,F2,F3)이고 필터 하나의 형태는 (5,5,?)이다. ?에는 앞의 특징맵의 채널(차원)수에 따라 다른데 흑백이라면 1, 컬러라면 3
#따라서 전체 필터의 형태는 (3,5,5,?) 이런 식이 된다.(필터 개수, 한 필터 크기, ?)
#필터셋이라는 말은 안쓰고 그냥 필터로 통칭
#Conv2D(3, kernel_size=5, activation='swish')
#Conv2D(6, kernel_size=5, activation='swish')
#(28,28,1)인 이미지를 필터하면 1st convolution layer(24,24,3)의 필터 생김. 크기가 24인 이유는 (필터 사이즈-1)만큼 작아짐
#2nd convolution layer에서 필터{(5,5,3)*6 3인 이유는 앞 채널의 수가 3이어서}하면 (20,20,6)의 특징맵 생김
#특징맵 하나를 만들 때 필터 하나는 앞 레이어의 특징맵 전체를 보고 만드는 것

import tensorflow as tf
import pandas as pd

#데이터 불러오기
(독립,종속),_ = tf.keras.datasets.mnist.load_data()
print(독립.shape, 종속.shape) #(60000, 28, 28) (60000,)
#Convolution layer는 이미지 하나의 형태가 이차원이 아닌 삼차원이어야 하기에 reshape
독립 = 독립.reshape(60000,28,28,1)
종속 = pd.get_dummies(종속) #원핫인코딩
print(독립.shape, 종속.shape) #(60000, 28, 28, 1) (60000, 10)

#모델 생성
X = tf.keras.layers.Input(shape=[28,28,1])
H = tf.keras.layers.Conv2D(3,kernel_size=5,activation='swish')(X)
H = tf.keras.layers.Conv2D(6,kernel_size=5,activation='swish')(H)
H = tf.keras.layers.Flatten()(H) #Flatten과정으로 표 형태로 펼침.
H = tf.keras.layers.Dense(84,activation='swish')(H) #은닉층
Y = tf.keras.layers.Dense(10,activation='softmax')(H) #출력층. 레이블 개수

model = tf.keras.models.Model(X,Y)
model.compile(loss = 'categorical_crossentropy', metrics = 'accuracy')

#모델 학습
model.fit(독립,종속,epochs=10)
#loss: 0.0197 - accuracy: 0.9945
#2min 26sec

#모델 이용
print(종속[0:5])
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)

#모델 과정 요약
model.summary()

#Pooling
#model.summary를 보면 flatten 과정 후 가중치의 수가 201,684로 매우 큰데 이를 보완하기 위한 것.
#Conv2D이후에 사용하여 크기를 반으로 줄임 ex) (24,24,3) -> (12,12,3) 2by2로 묶고 그 값들 중 가장 큰 수 남기기:MaxPooling, 평균 남기기:AveragePooling
#이렇게 해도 되는 이유는 Conv2D는 특징맵으로 값이 크다 = 필터로 찾으려는 특징이 많이 나타난 부분 으로 해석 가능하기 때문


###################################################
###################################################
#지금껏 배운 것 합쳐서 CNN 모델 만들기
import tensorflow as tf
import pandas as pd

#데이터 불러오기
(독립,종속),_ = tf.keras.datasets.mnist.load_data()
print(독립.shape, 종속.shape) #(60000, 28, 28) (60000,)
#Convolution layer는 이미지 하나의 형태가 이차원이 아닌 삼차원이어야 하기에 reshape
독립 = 독립.reshape(60000,28,28,1)
종속 = pd.get_dummies(종속) #원핫인코딩
print(독립.shape, 종속.shape) #(60000, 28, 28, 1) (60000, 10)

#모델 생성
X = tf.keras.layers.Input(shape=[28,28,1])
H = tf.keras.layers.Conv2D(3,kernel_size=5,activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Conv2D(6,kernel_size=5,activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Flatten()(H) #Flatten과정으로 표 형태로 펼침.
H = tf.keras.layers.Dense(84,activation='swish')(H) #은닉층
Y = tf.keras.layers.Dense(10,activation='softmax')(H) #출력층. 레이블 개수

model = tf.keras.models.Model(X,Y)
model.compile(loss = 'categorical_crossentropy', metrics = 'accuracy')
#'categorical_crossentropy'는 다중 클래스 분류 문제에서 주로 사용되는 손실 함수입니다. 
#모델이 예측한 클래스 확률 분포와 실제 클래스의 원-핫 인코딩 레이블 간의 차이를 측정하여 손실 값을 계산합니다. 


#모델 학습
model.fit(독립,종속,epochs=10)
#loss: 0.0681 - accuracy: 0.9815

#model summary
model.summary()
#가중치 8148로 매우 작아짐






#CNN called LeNet by Yann LeCun (1998)
import tensorflow as tf
import pandas as pd

#data load
(독립,종속),_=tf.keras.datasets.mnist.load_data()
독립 = 독립.reshape(60000,28,28,1)
종속 = pd.get_dummies(종속)
print(독립.shape,종속.shape) #(60000, 28, 28, 1) (60000, 10)

#model setting
X = tf.keras.layers.Input(shape=[28,28,1])

H = tf.keras.layers.Conv2D(6, kernel_size=5,padding='same',activation='swish')(X)
#padding='same'은 kernel_size와 관계없이 Input과 같은 사이즈로 출력

H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16,kernel_size=5,activation='swish')(H)

H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)

H = tf.keras.layers.Dense(120,activation='swish')(H)

H = tf.keras.layers.Dense(84,activation='swish')(H)

Y = tf.keras.layers.Dense(10,activation='softmax')(H)

model = tf.keras.models.Model(X,Y)

model.compile(loss='categorical_crossentropy', metrics='accuracy')

#model fit
model.fit(독립,종속,epochs=10)






#Cifar 10 by LeNet

#CNN called LeNet by Yann LeCun (1998)
import tensorflow as tf
import pandas as pd

#data load
(독립,종속),_ = tf.keras.datasets.cifar10.load_data()
print(독립.shape,종속.shape) #(50000, 32, 32, 3) (50000, 1)
#독립은 3차원이니 그대로 두고, 종속은 레이블 개수에 맞게 원핫인코딩
종속 = pd.get_dummies(종속.reshape(50000))
# 종속 = pd.get_dummies(종속) 하면 에러뜸. 2차원 데이터 형태가 표형태가 아니라서 원핫인코딩 안됨. 1차원혀형태로 reshape
print(독립.shape,종속.shape) #(50000, 32, 32, 3) (50000, 10)

#model setting
X = tf.keras.layers.Input(shape=[32,32,3])

H = tf.keras.layers.Conv2D(6, kernel_size=5,activation='swish')(X)
#padding='same'은 kernel_size와 관계없이 Input과 같은 사이즈로 출력

H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16,kernel_size=5,activation='swish')(H)

H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)

H = tf.keras.layers.Dense(120,activation='swish')(H)

H = tf.keras.layers.Dense(84,activation='swish')(H)

Y = tf.keras.layers.Dense(10,activation='softmax')(H)

model = tf.keras.models.Model(X,Y)

model.compile(loss='categorical_crossentropy', metrics='accuracy')

#model fit
model.fit(독립,종속,epochs=10)

#데이터가 더 어려워서 mnist에 비해 학습 잘 안됨

# 내 이미지 활용하기
#not mnist

#데이터 압축 해제
!wget -q https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/notMNIST_small.tar.gz
!tar -xzf notMNIST_small.tar.gz

# 1.과거의 데이터를 내 로컬로부터 이미지를 읽어들여 준비.
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 이미지 파일 경로 가져오기
paths = glob.glob('./notMNIST_small/*/*.png')

# 이미지 파일 경로를 무작위로 섞기
paths = np.random.permutation(paths)

# 이미지를 numpy 배열로 읽어들이기
독립 = np.array([plt.imread(path) for path in paths])

# 이미지 파일 경로에서 종속 변수 생성
종속 = np.array([path.split('/')[-2] for path in paths])

print(독립.shape, 종속.shape) #(18724, 28, 28) (18724,)

#이미지 사이즈가 제각각일 것. 이미지 편집 도구들의 resizing을 통해 수정

독립 = 독립.reshape(18724,28,28,1)
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape) #(18724, 28, 28, 1) (18724, 10)

import tensorflow as tf

#model setting
X = tf.keras.layers.Input(shape=[28,28,1])

H = tf.keras.layers.Conv2D(6, kernel_size=5,padding='same',activation='swish')(X)
#padding='same'은 kernel_size와 관계없이 Input과 같은 사이즈로 출력

H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16,kernel_size=5,activation='swish')(H)

H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)

H = tf.keras.layers.Dense(120,activation='swish')(H)

H = tf.keras.layers.Dense(84,activation='swish')(H)

Y = tf.keras.layers.Dense(10,activation='softmax')(H)

model = tf.keras.models.Model(X,Y)

model.compile(loss='categorical_crossentropy', metrics='accuracy')

#model fit
model.fit(독립,종속,epochs=10)

종속[0:10]

plt.imshow(독립[1], cmap='gray')