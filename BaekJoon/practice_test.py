import numpy as np
import matplotlib.pyplot as plt

# 파라미터
X0 = 0.02  # 초기 변위 (m)
ζ = 1/8   # 감쇠비
ω_n = 2   # 고유진동수 (rad/s)

# 시간 범위 설정 (0부터 10초까지)
t = np.linspace(0, 10, 1000)

# x(t) 계산
x_t = X0 * np.exp(-ζ * ω_n * t) * np.sin((np.sqrt(63)/4) * t + np.arctan(1/4))

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(t, x_t)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Underdamped Oscillation')
plt.grid(True)
plt.show()
