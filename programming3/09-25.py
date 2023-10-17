import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Linear Regression을 통해 공부시간과 중간고사 성적의 상관관계 파악하기

# 공부시간과 성적 데이터
X = np.array([2, 4, 6, 8])  # 공부시간 (hours)
Y = np.array([81, 93, 91, 97])   # 중간고사 성적

# 평균 계산
mx = np.mean(X)
my = np.mean(Y)

#최소제곱법
divisor = sum((i - mx)**2 for i in X) #20.0 (x - x평균)^2의 합
dividend = sum((x - mx) * (y - my) for x, y in zip(X, Y)) #46.0 (x-x평균)(y-y평균)의 합 

#기울기
a = dividend / divisor #2.3
#bias
b = my-(mx*a) #79.0


# predict 함수 정의
def predict(x):
    return a * x + b


# 예측값을 저장할 빈 리스트 생성
predict_result = []

# 모든 x 값에 대한 예측값 계산 및 저장
for i in range(len(X)):
    predict_result.append(predict(X[i]))

# 예측값 리스트 출력
# print(predict_result)
#[83.6, 88.2, 92.8, 97.4]


# 그래프 생성
plt.scatter(X, Y, label='Actual Scores')
plt.plot(X, predict_result, label=f'Predicted Scores: y = {a:.1f}x + {b:.1f}', color='red')
plt.scatter(X, predict_result, color='red', marker='o', label='Predicted Points')
plt.title('Correlation between Midterm Exam Scores and Study Hours')
plt.xlabel('Study Hours')
plt.ylabel('Midterm Exam Scores')

# error의 크기(거리) 나타내는 그래프
for i in range(len(X)):
    plt.plot([X[i], X[i]], [Y[i], predict_result[i]], linestyle='--', color='gray')


plt.legend()
plt.grid(True)
plt.show()

#오차
error = predict_result - Y

# 데이터 프레임 생성
data = {
    '공부시간': X,
    '실제점수': Y,
    '예측점수': predict_result,
    '오차': error
}

df = pd.DataFrame(data)

# 데이터 프레임 출력
print(df)

#Define MSE(Mean Squared Error)
def mse(Y, Y_pred):
    if len(Y) != len(Y_pred):
        raise ValueError("Y와 Y_pred의 길이가 다릅니다.")

    MSE = sum((Y - Y_pred)**2) / len(Y)
    return MSE

# 과제 1. 다음의 값들을 출력하시오
print('과제 1. 다음의 값들을 출력하여라')
print('\n')
print(f'공부시간의 평균값 : {mx:.1f}')
print(f'중간성적의 평균값 : {my:.1f}')
print('\n')
print(f'분모 : {divisor:.1f}')
print(f'분자 : {dividend:.1f}')
print('\n')
print(f'기울기 a = {a:.1f}')
print(f'y절편 b ={b:.1f}')
print('\n')

# 과제 2. a = 3, b = 76으로 했을 때 다음의 실행결과를 보이시오
print('과제 2. 기울기(a) = 3, y절편(b) = 76으로 했을 때 다음의 실행결과를 보여라')

#기울기
temp_a = 3
#bias
temp_b = 76


# predict 함수 정의
def predict(x):
    return temp_a * x + temp_b


# 예측값을 저장할 빈 리스트 생성
predict_result = []

# 모든 x 값에 대한 예측값 계산 및 저장
for i in range(len(X)):
    predict_result.append(predict(X[i]))

for i in range(len(X)):
    print(f'공부시간:{X[i]}, 실제점수:{Y[i]}, 예측점수:{predict_result[i]}')
print(f'주어진 표의 평균제곱오차는 {mse(Y,predict_result):.2f}')
