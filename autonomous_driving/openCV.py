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
