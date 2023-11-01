import time

# 현재 시간 기록
start_time = time.time()

# 출력하려는 메시지
message = "Hello, World!"

# print 문 실행
print(message)

# 현재 시간 다시 기록
end_time = time.time()

# 걸린 시간 계산
elapsed_time = end_time - start_time

# 결과 출력
print(f"Printing '{message}' took {elapsed_time} seconds.")
