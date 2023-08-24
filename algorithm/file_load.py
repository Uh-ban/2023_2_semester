#엑셀파일 불러오기 
import pandas as pd
df_exam = pd.read_excel('excel_exam.xlsx')
df_exam
#pd.read_excel은 첫번째 행을 변수명으로 인식해서 처리
#그러나 첫행이 변수가 아닌 데이터라면 어떻게 하는가
df_exam = pd.read_excel('excel_exam.xlsx', header = None) #header = None으로 처리
len(df_exam)
#엑셀 파일에 시트가 여러개라면 특정 파일만 불러오기
df_exam = pd.read_excel('excel_exam.xlsx', sheet_name = 'Sheet2')
df_exam