#파이썬이란 
#인터프리터 언어
#구글이 선정한 3대 언어(C java python)
#문법이 쉽고 공동작업과 유지보수에 유리
#무료

# print("hello World")
# age = int(input("Your age : "))
# print('Your age is ',age)



#두자리수 두개를 각각 입력받은 후 순서 바꾸기
i, j = input('두자리 정수를 입력하시오 : ')
print(i,j)
i, j = j, i
print(i,j)


#세 자리의 정수를 입력받은 후, 각 자리 별로 출력시키기
n = input('세자리 정수를 입력하시오 : ')
print(n[:1])
print(n[1:2])
print(n[2:3])