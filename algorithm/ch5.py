#엑셀파일 불러오기 
import pandas as pd
df_exam = pd.read_excel('excel_exam.xlsx')
df_exam
#pd.read_excel은 첫번째 행을 변수명으로 인식해서 처리
#그러나 첫행이 변수가 아닌 데이터라면 어떻게 하는가
df_exam = pd.read_excel('excel_exam.xlsx', header = None) #header = None으로 처리
len(df_exam)
#엑셀 파일에 시트가 여러개라면 특정 파일만 불러오기
df_exam = pd.read_excel('excel_exam.xlsx', sheet_name = 'Sheet2') #시트 이름이나 숫자 넣으면 됨
df_exam
#CSV파일 불러오기
import pandas as pd
df_csv_exam = pd.read_csv('exam.csv')
df_csv_exam
#데이터 프레임을 CSV파일로 저장하기
df_midterm = pd.DataFrame({'english' : [90,80,60,70],
                           'math'    : [50,60,100,20],
                           'nclass'  : [1,1,2,2]})
df_midterm.to_csv('df_midterm.csv') #저장한 데이터는 워킹 디렉터리에 만들어짐.
                                    #첫번째 열에 인덱스 번호가 삽입됨.
                                    #이를 없애려면 
df_midterm.to_csv('df_midterm.csv', index = False)

#데이터 파악하기
import pandas as pd
exam = pd.read_csv("exam.csv")
exam.head(10) #기본 앞에서 5행 출력. 괄호에 원하는 크기 입력 가능
exam.tail() #기본 뒤에서 5행 출력. 괄호에 원하는 크기 입력 가능
exam.shape #(행,렬)
exam.info() #변수들의 속성
exam.describe() #변수들의 요약 통계량. 특징파악에 용이

import pandas as pd
mpg = pd.read_csv('mpg.csv')
mpg.head()
mpg.shape #(234,11)
#manufacturer, model, displ, year, cyl, trans, drv, cty, hwy, fl, category
mpg.describe(include = 'all') #include = 'all' = 변수 모두 표시

import numpy as np
import pandas as pd
count_test = mpg['Fuel efficiency test'].value_counts()
count_test.plot.bar(rot = 0) 

#중첩 조건문 횔용해서 연비 등급 나누기
#30이상이면 A, 29-20은 B, 20미만 C
mpg['Fuel efficiency grade'] = np.where(mpg['total']>= 25, 'A',
                               np.where(mpg['total']>= 20, 'B','C'))
mpg.head(10) 
count_grade = mpg['Fuel efficiency grade'].value_counts()
count_grade.plot.bar(rot = 0)
#plot.bar는 기본적으로 내림차순. 알파벳 순으로 바꾸려면 정렬 후 실행
count_grade = mpg['Fuel efficiency grade'].value_counts().sort_index()
count_grade.plot.bar(rot = 0)
#목록에 해당하는 행으로 변수 만들기
mpg['size'] = np.where((mpg['category'] == 'compact') |
                        (mpg['category'] == 'subcompact') |
                        (mpg['category'] == '2seater'),'small', 'large')
mpg['size'].value_counts()
#위 코드를 df.isin을 활용하여 간략화하면
mpg['size'] = np.where(mpg['category'].isin(['compact','subcompact','2seater']), 'small', 'large')
mpg['size'].value_counts()

#실전 분석하기
#midwest.csv는 미국 동북중부 437개 지역의 인구통계 정보를 담고있다. 이를 통해 문제를 해결해보아라

#midwest.csv 불러와 파일 특징 파악하기
import pandas as pd
df = pd.read_csv("midwest.csv")
df.head()
df.shape # (437,28)
df.describe()

#poptotal 변수를 total로, popasian 변수를 asian으로 바꾸기
df_new = df.copy()
df_new = df.rename(columns = {'poptotal' : 'total', 'popasian' : 'asian'})
df_new.head()

#total과 asian을 활용해서 전체 인구 대비 아시아 인구의 백분율을 나타내는 변수를 만드시오
df_new['asian/total(%)'] = (df_new['asian'] / df_new['total'] * 100)
df_new.head()

#히스토그램 만들어 분포 확인
df_grade = df_new['asian/total(%)'].value_counts().sort_index()
# df_grade.plot.bar()

#아시아 인구 전체의 평균을 구하고 평균 이상이면 large or small
df_new['asian/total(%)'].mean() #0.48
df_new['asian_grade'] = np.where(df_new['asian/total(%)'] >= 0.48, 'large', 'small')
df_new.head()

#그래프 만들어 비율 확인
df_grade = df_new['asian_grade'].value_counts().sort_index()
df_grade.plot.bar(rot = 0)



