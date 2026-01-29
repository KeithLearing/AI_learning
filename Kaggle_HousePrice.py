import matplotlib
import matplotlib.pyplot as plt
import numpy
import torch
import pandas
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import requests

# DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv'
# r = requests.get(DATA_URL, stream=True, verify=True)
# with open("./kaggle_house_pred_train.csv", 'wb') as f:
#     f.write(r.content)
#
# DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv'
# r = requests.get(DATA_URL, stream=True, verify=True)
# with open("./kaggle_house_pred_test.csv", 'wb') as f:
#     f.write(r.content)

train_data = pandas.read_csv("train.csv")
test_data = pandas.read_csv("test.csv")

all_features = pandas.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# print(train_data.shape)
# print(train_data.iloc[:5, :8])

# 提取全是数字的特征名字
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print(numeric_features)

# 对数据做标准化处理，对应位置赋值
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())

# 标准化处理之后，将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pandas.get_dummies(all_features, dummy_na=True)

# print(all_features.shape)
# print(all_features.iloc[:5, :8])

# 划分数据集
n_train = train_data.shape[0]

# print(all_features[n_train:].values.dtype)  ## object
train_features = torch.tensor(all_features[:n_train].values.astype(float), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values.astype(float), dtype=torch.float32)

train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# print(train_features.shape)
# print(test_features.shape)
# print(train_labels.shape)

# 数据分批
batch_size = 32
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)


# print(f'每一批{len(next(iter(train_loader))[0])}个，一共{len(train_loader)}批')

# 定义神经网络
class Net(torch.nn.Module):
    def __init__(self, in_put, hidden1, hidden2, out_put):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_put, hidden1)
        self.linear2 = torch.nn.Linear(hidden1, hidden2)
        self.linear3 = torch.nn.Linear(hidden2, out_put)

    def forward(self, data):
        x = self.linear1(data)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x


# 初始化神经网络
in_features = train_features.shape[1]
hidden1, hidden2, out_put = 200, 200, 1
model = Net(in_features, hidden1, hidden2, out_put).to(device)


loss = torch.nn.MSELoss()

learn_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), learn_rate)

print(in_features)
print(model)

# 训练神经网络

epochs = 100


def train(train_loader):
    train_ls = []
    for epoch in range(epochs):
        print(f'正在执行第{epoch + 1}次训练')
        loss_sum = 0
        for train_batch, label_batch in train_loader:
            train_batch, label_batch = train_batch.to(device), label_batch.to(device)
            l = loss(model(train_batch), label_batch)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_sum += l.item()
        train_ls.append(loss_sum)
    plt.plot(range(epochs), train_ls)
    plt.show()


train(train_loader)


def test(test_features):
    test_features = test_features.to(device)
    preds = model(test_features).detach().to("cpu").numpy()
    print(preds.squeeze().shape)
    test_data['SalePrice'] = pandas.Series(preds.squeeze())

    # axis选择拼接的维度
    return pandas.concat([test_data['Id'], test_data['SalePrice']], axis=1)


submission = test(test_features)
