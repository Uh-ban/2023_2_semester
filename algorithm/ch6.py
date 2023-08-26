import pandas as pd
import numpy as np
import sklearn

#data preprocessing
#조건에 맞는 데이터 추출
df = pd.read_csv('exam.csv')
df.shape

#반별로 나누어 추출
# print(df.query('nclass == 1'))
# print(df.query('nclass == 2'))
# print(df.query('nclass == 3'))
# print(df.query('nclass == 4'))
# print(df.query('nclass == 5'))

#수학성적과 과학성적이 50 이상인 학생 추출
df.query('math >= 50 & science >= 50')

#1,3,5반 학생 추출
df.query('nclass == 1 | nclass == 3 | nclass == 5')
df.query('nclass in [1,3,5]')

#추출한 헹으로 데이터 만들기
nclass1 = df.query('nclass == 1')
nclass1.mean()

#외부변수 이용해 추출(@붙이기)
var = 3
df.query('nclass == @var')