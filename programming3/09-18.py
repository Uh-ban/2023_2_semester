import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

Data_set = np.loadtxt('https://raw.githubusercontent.com/dhshinEddie/DeepKMOU/main/W3/ThoraricSurgery3.csv', delimiter=",")

X = Data_set[:,0:16]
Y = Data_set[:,16]
print(X.shape,Y.shape) #(470, 15) (470,)

#model
model = Sequential()

model.add(Dense(30, input_dim=16, activation='relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X,Y, epochs=5, batch_size=16) 


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

Data_set = np.loadtxt('https://raw.githubusercontent.com/dhshinEddie/DeepKMOU/main/W3/ThoraricSurgery3.csv', delimiter=",")

# 데이터 셔플
np.random.shuffle(Data_set)

X = Data_set[:,0:16]
Y = Data_set[:,16]
print(X.shape,Y.shape) #(470, 15) (470,)

#model
model = Sequential()

model = Sequential()

# 첫 번째 은닉층
model.add(Dense(30, input_dim=16, activation='swish'))

# 두 번째 은닉층 (추가 은닉층)
model.add(Dense(20, activation='swish'))

# 세 번째 은닉층 (추가 은닉층)
model.add(Dense(10, activation='swish'))

# 출력 레이어
model.add(Dense(1, activation='sigmoid'))


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X,Y, epochs=5, batch_size=16)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터 로드 및 셔플
Data_set = np.loadtxt('https://raw.githubusercontent.com/dhshinEddie/DeepKMOU/main/W3/ThoraricSurgery3.csv', delimiter=",")
np.random.shuffle(Data_set)

X = Data_set[:, 0:16]
Y = Data_set[:, 16]

# 데이터 분할 (학습 데이터와 검증 데이터)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 모델 설정
model = Sequential()
model.add(Dense(30, input_dim=16, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 배치 크기 변경
batch_size = 32

# 모델 학습
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=batch_size)
