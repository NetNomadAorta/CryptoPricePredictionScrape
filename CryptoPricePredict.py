import torch
import torch.nn as nn
import torch.utils 
import torch.utils.data
from torch.autograd import Variable
import os
from datetime import datetime
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
device = torch.device(dev)
torch.cuda.set_device(device)

df=pandas.read_csv('datasets/bitcoin/data.csv')

print(df.shape)
df.head()


test_days   = 20 # must be greater or equal to numdays + forwarddays
numdays     = 7 # number of days to look back10.5.3.
forwarddays = 1 # number of days to look forward
df_train    = df[:len(df)-test_days]
df_test     = df[len(df)-test_days:]
training_set = df_train.values
test_set    = df_test.values
transformer = MinMaxScaler().fit(training_set)
training_set = transformer.transform(training_set)
test_set    = df_test.values
test_set    = transformer.transform(test_set)

x_train=[];y_train=[];
x_test=[];y_test=[];
for i in range(len(training_set)-numdays-forwarddays):
    x_train.append(training_set[i:(i+numdays)])
    y_train.append(training_set[i+numdays+(forwarddays-1)])
for i in range(len(test_set)-numdays-forwarddays):
    x_test.append(test_set[i:(i+numdays)])
    y_test.append(test_set[i+numdays+(forwarddays-1)])
x_train=torch.from_numpy(np.array(x_train).squeeze().astype(np.float32))
y_train=torch.from_numpy(np.array(y_train).astype(np.float32))
x_test=torch.from_numpy(np.array(x_test).squeeze().astype(np.float32))
#y_test=torch.from_numpy(np.array(y_test).astype(np.float32))

# setting arrays or variables to cuda (gpu) from cpu
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
# y_test = y_test.to(device) #WHY NO WORK?


print(x_train.shape, x_test.shape)

hidden_size = 300
num_layers=1

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # using a GRU (Gated Recurrent Unit), also try and LSTM
        self.rnn1 = nn.GRU(input_size=numdays, hidden_size=hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.20)
        self.dense1 = nn.Linear(hidden_size, int(hidden_size/8))
        self.dense2 = nn.Linear(int(hidden_size/8), 1)

    def forward(self, x, hidden):
        x_batch = x.view(len(x), 1, -1)
        x_r, hidden = self.rnn1(x_batch, hidden)
        x_d = self.dropout(x_r)
        x_l = self.dense1(x_d)
        x_l2 = self.dense2(x_l)
        return x_l2, hidden

    def init_hidden(self):
        return Variable(torch.randn(num_layers, 1, hidden_size))


model = MyModel().cuda()
lossfn = nn.MSELoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
initial_hidden = model.init_hidden().cuda()
print(model)


for i in range(100000):
    model.zero_grad()
    hidden=initial_hidden
    out, hidden = model(x_train, hidden)
    loss = lossfn(out.view(-1,1), y_train)
    if i % 20 == 0:
        print('{:%H:%M:%S} epoch {} loss: {}'.format(datetime.now(), i, loss.cpu().data.numpy().tolist()), flush=True)
    loss.backward()
    optimizer.step()


# Training set
pred, new_hidden = model(x_train, hidden)
prices=transformer.inverse_transform(pred.cpu().detach().numpy().reshape(-1,1))
actual=transformer.inverse_transform(y_train.cpu().numpy().reshape(-1,1))

import matplotlib.pyplot as plt
plt.plot(range(prices.shape[0]-2), prices[2:], label='Predicted Price')
plt.plot(range(prices.shape[0]-2), actual[2:], 'r', label='Actual Price')
plt.legend()
plt.show()


# Training set
pred, new_hidden = model(x_test, hidden)
prices=transformer.inverse_transform(pred.cpu().detach().numpy().reshape(-1,1))
actual=transformer.inverse_transform(y_test)

import matplotlib.pyplot as plt
plt.plot(range(prices.shape[0]-2), prices[2:], label='Predicted Price')
plt.plot(range(prices.shape[0]-2), actual[2:], 'r', label='Actual Price')
plt.legend()
plt.show()