import torch
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        # 선형 계층을 생성한다.
        self.linear = torch.nn.Linear(1, 1)

        # 시그모이드 함수를 생성한다.
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 입력값 x를 선형 계층에 통과시킨다.
        z = self.linear(x)

        # 시그모이드 함수를 사용하여 z를 출력값으로 변환한다.
        y_hat = self.sigmoid(z)

        return y_hat

# 신경망을 생성한다.
net = Net()

# 비용 함수를 생성한다.
bce = torch.nn.BCELoss()

# 최적화 알고리즘을 생성한다.
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 학습 데이터를 준비한다.
x_lst = torch.Tensor([1, 3, 5, 7, 9]).view(-1, 1)
y_lst = torch.Tensor([1, 1, 1, 0, 0]).view(-1, 1)

# 5000회 반복하여 신경망을 학습한다.
for epoch in range(5000):

    # 신경망을 통해 예측값을 계산한다.
    y_hat = net(x_lst)

    # 비용 함수를 계산한다.
    loss = bce(y_hat, y_lst)

    # 손실 함수의 기울기를 구한다.
    optimizer.zero_grad()
    loss.backward()

    # 가중치를 업데이트한다.
    optimizer.step()

    # 100회마다 학습 상태를 출력한다.
    if epoch % 100 == 0:
        print(f'epoch = {epoch}, loss = {loss.item():0.5f}')

# 학습이 완료된 후 신경망의 예측 결과를 그래프로 나타낸다.
plt.plot(x_lst, y_lst, '*')
plt.plot(x_lst, net(x_lst).detach().numpy(), '-')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(('true', 'predicted'))
plt.show()

# 8시간 수면 후 피곤할 확률을 계산한다.
x_test = torch.Tensor([[8]])
y_test = net(x_test)

# 피곤할 확률이 50% 이상이면 '피곤하다', 그렇지 않으면 '피곤하지 않다'를 출력한다.
is_tired = '하다' if y_test.item() >= 0.5 else '하지 않다'

print(f'{x_test.item()}시간 수면 후 \
피곤한 확률은 {y_test.item()*100:0.1f}%이다')

print(f'따라서 피곤{is_tired}.')

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
import torch
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

EPOCH = 10000


dataset = datasets.load_iris()
x = torch.Tensor(dataset.data)
print(x.shape)
y = torch.LongTensor(dataset.target)
print(y.shape)

#총 150:5 데이터에 마지막 5열에 답이 있음을 알 수 있음.

class Net(torch.nn.Module) :
    def __init__ (self):
        super (Net, self).__init__()
        self. fc1 = torch.nn. Linear(4, 64) #4는 feature의 갯수. 그 사이 히든 레이어의 수는 상관 없음. 4개의 특성을 64개의 특성으로 펼쳐서 보겠다. 은닉층의 입력 갯수가 기존의 갯수보다는 커야하고 너무 커도 안 좋음
        self. fc2 = torch. nn. Linear(64, 32) #이 전 레이어의 수와 같은 수로 받아야 함.
        self.fc3 = torch.nn. Linear(32, 16) #이 전 레이어의 수와 같은 수로 받아야 함.
        self.fc4 = torch.nn. Linear(16, 3) #3은
        self.relu = torch.nn.ReLU() #마지막 층에서 ReLU값을 쓰면 0이하의 값은 0으로 나머지는 선형함수로 표현. 따라서 마지막 층에서 렐루함수를 쓰면 정보의 손실이 일어나기에 마지막 층에서는 쓰지 않음

    def forward (self, x) :
        x = self. relu(self. fc1(x))
        x = self.relu(self. fc2(x))
        x = self.relu(self. fc3(x))
        z = self. fc4(x)

        return z

net = Net()
cel = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01 )
loss_lst = []

for epoch in range(EPOCH):
    z = net(x)

    loss = cel(z,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_lst.append(loss.item())

    if epoch % 1000 == 0:
        print(f'epoch = {epoch}, loss = {loss.item():0.5f}')

plt.plot(range(EPOCH),loss_lst)
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.grid(True)
plt.show()

z = net(x)
y_hat = torch.argmax(torch.softmax(z, dim = 1), dim=1)
correct = torch.sum(y==y_hat)
accuracy = correct / len(y)*100

print(f'Accuracy = {accuracy:0.2f}%')

