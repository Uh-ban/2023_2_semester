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


