# Inspired by this article: https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

df = pd.read_csv('../data/preprocessing/pp_sg_cleaned.csv', sep=';')
timeseries = df.head(10000)[["P24"]].values.astype('float32')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_on_gpu = torch.cuda.is_available()

# train-test split for time series
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


lookback = 10
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        # self.to(device)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
model.to(device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        # if train_on_gpu:
        #     X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 20 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train.to(device))
        train_rmse = np.sqrt(loss_fn(y_pred.to("cpu"), y_train))
        y_pred = model(X_test.to(device))
        test_rmse = np.sqrt(loss_fn(y_pred.to("cpu"), y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train.to(device))
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train.to(device))[:, -1, :].to("cpu")
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size + lookback:len(timeseries)] = model(X_test.to(device))[:, -1, :].to("cpu")
# plot
plt.plot(timeseries)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()