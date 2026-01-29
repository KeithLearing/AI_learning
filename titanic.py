import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from win32timezone import now

dftrain_raw = pd.read_csv('./eat_pytorch_datasets/titanic/train.csv')
dftest_raw = pd.read_csv('./eat_pytorch_datasets/titanic/test.csv')

print(dftrain_raw.head(10))

ax = dftrain_raw['Survived'].value_counts().plot(kind='bar', figsize=(12, 8), fontsize=15, rot=0)
ax.set_ylabel('counts', fontsize=15)
ax.set_xlabel('survived', fontsize=15)
# plt.show()

ax = dftrain_raw['Age'].value_counts().plot(kind='hist', bins=20, color='purple', figsize=(12, 8), fontsize=15)
ax.set_xlabel('age', fontsize=15)
ax.set_ylabel('frequency', fontsize=13)
# plt.show()

ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
ax.legend(['survived=0', 'survived=1'], fontsize=12)
ax.set_ylabel('Density', fontsize=15)
ax.set_xlabel('Age', fontsize=15)
# plt.show()

def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['P_class' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return dfresult

x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw[['Survived']].values

x_test = preprocessing(dftest_raw).values
y_test = dftest_raw[['Survived']].values

x_train = preprocessing(dftrain_raw).astype(np.float32).values
print(x_train.shape)
x_test = preprocessing(dftest_raw).astype(np.float32).values
# print(x_test.shape)

# x_test = preprocessing(dftest_raw).values
# y_test = dftest_raw[['Survived']].values


## 做成数据管道
dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float()), batch_size=8, shuffle=True)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float()), batch_size=8,shuffle=False)
#
# for features, labels in dl_valid:
#     print(features, labels)
#     break

def create_net():
    net= nn.Sequential()
    net.add_module("linear1", nn.Linear(15, 20))
    net.add_module("relu1", nn.ReLU())
    net.add_module("linear2", nn.Linear(20, 15))
    net.add_module("relu2", nn.ReLU())
    net.add_module("linear3", nn.Linear(15, 1))
    net.add_module("sigmoid", nn.Sigmoid())

    return net

net = create_net()
print(net)

# from torchsummary import summary
# summary(net, (712, 15))

from sklearn.metrics import accuracy_score

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(), y_pred.data.numpy()>0.5)
metric_name = "accuracy"

epochs = 10
log_step = 30

dfhistory = pd.DataFrame(columns=['epoch', 'train_loss', metric_name, "val_loss", "val"+metric_name])
print("start training...")
print("==========" * 8)
for epoch in range(1, epochs + 1):
    ## 模型训练
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1

    for step,(features, labels) in enumerate(dl_train, 1):
        optimizer.zero_grad()

        ## 正向传播
        preditions = net(features)
        loss = loss_func(preditions, labels)
        metric = metric_func(preditions, labels)

        ## 反向传播求梯度
        loss.backward()
        optimizer.step()

        ## 打印batch级别的日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step % log_step == 0:
            print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                  (step, loss_sum / step, metric_sum / step))


    ## 验证循环
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1
    for val_step, (features, labels) in enumerate(dl_valid, 1):
        preditions = net(features)
        val_loss = loss_func(preditions, labels)
        val_metric = metric_func(preditions, labels)

        val_loss_sum += val_loss.item()
        val_metric = metric_func(preditions, labels)

        val_metric_sum += val_metric.item()
        val_loss_sum += val_loss.item()

        info = (epoch, loss_sum / step, metric_sum / step,
                val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info
        # 打印epoch级别日志
    print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
               " = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
              % info)
print('Finished Training...')




