# 입력받은 정수들 중 i~j번째의 합 구하기

#1. 입력받은 정수를 리스트로 정렬
user_input = input("원하는 정수를 입력하세요: ")

input_list = user_input.split()

int_list = [int(i) for i in input_list]

print("입력받은 정수를 리스트로 반환합니다. : ", int_list)

#2. i~j번쨰까지의 합은  S(j) - S(i-1)
Sum_list = [0]
temp = 0
for i in int_list:
    temp += i
    Sum_list.append(temp)

user_output = input("어떤 구간의 합을 구할까요?: ")
output_list = user_output.split()
int_list2 = [int(i) for i in output_list]
i = int_list2[0]
j = int_list2[1]
def S(i,j):
    Sum_list[j] - Sum_list[i-1]
print(S(i,j))

#주어진 자연수를 연속된 자연수의 합으로 표현할 수 있는 경우의 수 구하기
'''
n = int(input())
count = 1 
start_index = 1
end_index = 1
sum = 1

while end_index != n:
    if sum == n: # answer case
        count += 1
        end_index += 1
        sum += end_index
    elif sum > n:
        sum -= start_index
        start_index += 1
    else:
        end_index += 1
        sum += end_index

print(count)
'''
#좋은 수(주어진 수들 중 두 수의 합으로 나타낼 수 있는 수) 구하기

def count_elements_with_sum(arr):
    # 결과를 저장할 변수 초기화
    count = 0
    
    # 리스트 길이
    n = len(arr)
    
    # 모든 가능한 두 수의 합을 계산
    for i in range(n):
        for j in range(i + 1, n):  # 같은 수를 중복해서 더하지 않기 위해 j를 i+1부터 시작
            total = arr[i] + arr[j]
            
            # 합이 리스트 내에 있는 경우, count 증가
            if total in arr:
                count += 1
    
    # 결과 반환
    return count

# 테스트를 위한 리스트 예제
numbers = list(map(int, input("띄어쓰기로 구분된 정수를 입력하세요: ").split()))

result = count_elements_with_sum(numbers)
print("서로 다른 두 수의 합으로 나타낼 수 있는 요소 개수:", result)
