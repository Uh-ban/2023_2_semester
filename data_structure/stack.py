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
print(S(4,6))

