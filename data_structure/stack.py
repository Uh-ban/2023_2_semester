print("stack and queue")

#스택으로 수열 만들기
 
N = int(input())
A = [0]*N

for i in range(N):
    A[i] = int(input())

stack = []
num = 1
result = True
answer = ""

for i in range(N):
    su = A[i]
    if su >= num: # 현재 수열값 >= 오름차순 자연수: 값이 같아질 때까지 append()진행
        while su >= num:
            stack.append(num)
            num += 1
            answer += "+\n"
        stack.pop()
        answer += "-\n"
    else: # 현재 수열값 < 오름차순 자연수: pop()을 수행해 수열 원소를 꺼냄
        n = stack.pop()
        #스택의 가장 위의 수가 만들어야 하는 수열의 수보다 크면 수열을 출력할 수 없음
        if n > su:
            print("NO")
            result = False
            break
        else:
            answer += "-\n"

if result:
    print(answer)
        
#오큰수 구하기
#오큰수란 수열에서 N의 오른쪽에 위치한 수 중 N보다 큰 가장 왼쪽에 위치한 수

user_list = list(map(int, input("수열을 입력하시오").split()))
