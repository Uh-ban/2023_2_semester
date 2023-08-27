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

df = pd.read_csv('exam.csv')
#여러 변수 추출시 [[]]로 데이터 프레임 유지
df[['nclass','math','english']]

#변수 제거하기
df.drop(columns = 'math')
df.drop(columns = ['math','english'])

#pandas 함수 조합 (query, [])
#nclass가 1인 데이터의 english 데이터
df.query('nclass == 1')['english']

df.query('math >= 50')[['id','math']]#데이터 정렬 sort_values()
df.sort_values('math') #math 오름차순
df.sort_values('math', ascending = False) #math 내림차순
df.sort_values(['nclass','math']) #반을 오름차순으로 정렬하고 반 안에서 수학성적 오름차순
df.sort_values(['nclass','math'], ascending = [True, False]) #반을 오름차순으로 정렬하고 반 안에서 수학성적 내림차순

#audi 자동자 중 hwy가 높은지 알아보려 함. 1-5등 데이터 추출
df = pd.read_csv('mpg.csv')
audi = df.query('manufacturer == "audi"')
audi.sort_values('hwy',ascending = False)[:5]

#파생변수 추가하기
#변수명 = 변수를 이루는 공식. *새로운 변수는 따옴표 안 함
df = pd.read_csv('exam.csv')
df.assign(total = df['math']+df['english'] + df['science'])

#여러 변수 추가
df.assign(total = df['math']+df['english'] + df['science'],
          mean = (df['math']+df['english'] + df['science']) / 3)
#assign에 np.where
import numpy as np
df.assign(test = np.where(df['science'] >= 60, 'pass', 'fail'))

#파생변수 추가하기
#변수명 = 변수를 이루는 공식. *새로운 변수는 따옴표 안 함
df = pd.read_csv('exam.csv')
df.assign(total = df['math']+df['english'] + df['science'])

#여러 변수 추가
df.assign(total = df['math']+df['english'] + df['science'],
          mean = (df['math']+df['english'] + df['science']) / 3)
# 변수명이 긴 경우, lambda를 이용해 간략화
df.assign(total = lambda x: x['math'] + x['english'] + x['science'],
          mean = lambda x: (x['math']+x['english'] + x['science']) / 3)
#assign에 np.where
import numpy as np
df.assign(test = np.where(df['science'] >= 60, 'pass', 'fail'))

#집단별로 요약하기
#요약 통계량 .agg()
df.agg(mean_math = ('math','mean')) #요약값 할당할 변수명과 요약에 쓸 변수명, 함수명
#.agg는 주로 groupby로 묶인 것에 쓰임
df.groupby('nclass').agg(mean_math = ('math','mean'))

#여러 통계량 한번에 구하기
df.groupby('nclass').agg(mean_math = ('math','mean'),
                         sum_math = ('math', 'sum'),
                         median_math = ('math','median'),
                         n = ('nclass','count'))




