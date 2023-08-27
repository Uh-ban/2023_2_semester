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

import pandas as pd
import numpy as np
import sklearn

#배기량이 4이하인 차와 5이상인 차의 hwy 평균 비교
df = pd.read_csv('mpg.csv')
df.head()
low = df.query('displ <= 4')
high = df.query('displ >= 5')
print(low['hwy'].mean())
print(high['hwy'].mean())

df.head()
#audi, toyota의 도시 연비 평균 비교
audi = df.query('manufacturer == "audi"')
toyota = df.query('manufacturer == "toyota"')

if audi['cty'].mean() > toyota['cty'].mean():
    print('audi')
else:
    print('toyota')

#chevrolet, ford, honda 세 브랜드의 hwy 평균
CFH = df.query('manufacturer in ["chevrolet","ford","honda"]')
CFH['hwy'].mean()