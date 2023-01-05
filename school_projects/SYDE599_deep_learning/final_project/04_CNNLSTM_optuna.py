# %%
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import signal

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
data_dir = "/home/ob1/workplace/trevor/syde599/norm_fog_data/"
# data_dir = 'drive/MyDrive/599_project_data/norm_fog_data/'

data_files = glob.glob(data_dir + "*.csv")
len(data_files)

# %% [markdown]
# # Load the data

# %%
WINDOW_SIZE = 650
WINDOW_STEP = 200

train_X = []
train_Y = []
test_X = []
test_Y = []

for file in data_files:
    print(file)
    path_parts = file.split('/')
    patient_num = path_parts[-1].split('_')[0]
    
    data = np.loadtxt(file, delimiter=",", skiprows=1)
    # Timestamp is in col 0, labels are in col 1
    # Data is in cols 2 to end
    y = data[:, 1]
    x = data[:, 2:]
    y[y<0] = 0
    y[y>0] = 1
    print(y.sum())
    # Remove both waist and right shank columns since test data is missing there
    #channels_to_remove = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # These are channel inds AFTER removing timestamp and labels
    #x = np.delete(x, channels_to_remove, axis=1)


    # Split into non-overlapping windows
    # Don't use the last bit of data that doesn't fill a whole window
    n_windows = y.size // WINDOW_SIZE
    end = WINDOW_SIZE * n_windows
    x = x[:end, :]  # (n_samples, d)
    y = y[:end]

    # Reshape into (n_windows, window_size, d)
    #x = x.reshape(n_windows, WINDOW_SIZE, -1)
    #y = y.reshape(n_windows, WINDOW_SIZE)
    # Split into overlapping windows
    x = np.lib.stride_tricks.sliding_window_view(x, WINDOW_SIZE, axis=0)[::WINDOW_STEP]
    y = np.lib.stride_tricks.sliding_window_view(y, WINDOW_SIZE, axis=0)[::WINDOW_STEP]

    x = x.transpose(0,2,1)
    
    
    if patient_num in ('001', '009', '010'):
        test_X.append(x)
        test_Y.append(y)
    else:
        train_X.append(x)
        train_Y.append(y)

# %%
train_X = np.concatenate(train_X, axis=0)
train_Y = np.concatenate(train_Y, axis=0)

test_X = np.concatenate(test_X, axis=0)
test_Y = np.concatenate(test_Y, axis=0)

# %% [markdown]
# # Example of augmentation applied

# %%
train_X.shape

# %%
train_X.shape

# %%
y = train_X[0, :, 0]

# %%
y_stretch = signal.resample(y, int(len(y) * 1.1))
y_squeeze = signal.resample(y, int(len(y) * 0.9))
y_shrink = y * 0.8
y_scaled = y * 1.2

# %%
plt.plot(y, label="original", zorder=4, c='k')
plt.plot(y_stretch, label="stretch", alpha=0.4)
plt.plot(y_squeeze, label="squeeze", alpha=0.4)
plt.plot(y_shrink, label="shrink", alpha=0.4)
plt.plot(y_scaled, label="amplified", alpha=0.4)
plt.legend()
plt.show()

# %%
# Build dataset, dataloader, collate function

# %%
class FoGDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        """
        Parameters:
        -----------
        data_x: np.ndarray of shape (num_samples, window_size, num_features) which contains data features for FoG
        data_y: np.ndarray of shape (num_samples, window_size) which contains binary labels for FoG
        """
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, index):
        """Returns tuple of (data, label) at index"""
        inputs = self.data_x[index, :, :]
        labels = self.data_y[index, :]
        return inputs, labels


class FoGDataAugment:
    """
    Collate function to apply random time stretch/squeeze and signal shrink/scale
    Apply stretch/squeeze to the time dimension by resampling and truncating to out_samples
        The lower bound of stretch must satisfy (lb * len(input)) > out_samples
    """
    def __init__(self, out_samples=512, p=0.5, stretch=(0.8, 1.2), scale=(0.8, 1.2)):
        """
        Parameters:
        -----------
        p: float between [0, 1], probability of applying each the stretch and scale transform independently
        strech: tuple of float, upper and lower bound on time stretch factor
        scale: tuple of float, upper and lower bound on signal scale factor
        """
        self.p = p
        self.stretch = stretch
        self.scale = scale
        self.out_samples = out_samples
    
    def _reduce_labels(self, y):
        """If there is a 1 in the label, then return 1"""
        return np.any(y == 1, axis=-1).astype(int)

    def _random_crop(self, inputs, labels):
        """Apply a random crop of the signal of length self.out_samples to both inputs and labels"""
        n, d = inputs.shape
        max_offset = n - self.out_samples
        offset = np.random.choice(max_offset)
        inds = slice(offset, offset + self.out_samples)
        return inputs[inds, :], labels[inds]

    def __call__(self, data):
        """
        Parameters:
        -----------
        data: list of tuple of (inputs, labels) of length batch_size
            inputs: np.ndarray, dimensions (n_samples, n_channels), signal data
            labels: np.ndarray, dimensions (n_samples,), binary label vector for the signal data

        Returns:
        --------
        (inputs, labels): augmented signal data, reduced labels
        """
        x = []
        y = []
        for (inputs, labels) in data:
            n, d = inputs.shape
            assert (self.stretch[0] * n) >= self.out_samples, f"input size {n} must be greater than {int(self.out_samples / self.stretch[0])} to apply augmentation"

            # Randomly apply time stretch
            if np.random.binomial(1, self.p) != 0:
                lb, ub = self.stretch
                stretch = np.random.uniform(lb, ub)
                inputs = signal.resample(inputs, int(n * stretch), axis=0)  # Resample the time (n_samples) axis
            if np.random.binomial(1, self.p) != 0:
                lb, ub = self.scale
                scale = np.random.uniform(lb, ub)
                inputs = scale * inputs  # Scale all channels equally
            
            # Apply random crop to self.out_size on both inputs and labels
            inputs, labels = self._random_crop(inputs, labels)

            labels = self._reduce_labels(labels)
            x.append(inputs)
            y.append(labels)
        collated_inputs = torch.tensor(x, dtype=torch.float32)
        collated_labels = torch.tensor(y, dtype=torch.float32)
        return collated_inputs, collated_labels

# %%
train_dataset = FoGDataset(train_X, train_Y)
test_dataset = FoGDataset(test_X, test_Y)

augment_fn = FoGDataAugment(out_samples=512)
test_augment = FoGDataAugment(out_samples=512, p=0)

BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=augment_fn, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=test_augment)

# %%
del train_X, train_Y, test_X, test_Y, train_dataset, test_dataset

# %% [markdown]
# # Define model

# %%
class TransformerModel(nn.Module):
    def __init__(self, n_layers=3, in_features=30, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.input_transform = nn.Linear(in_features, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=4*d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # We'll just make a prediction on the features of the first and last sequence point
        self.classifier = nn.Linear(d_model * 2, 1)
    
    def forward(self, x):
        x = self.input_transform(x)
        x = self.encoder(x)
        x = self.layer_norm(x)

        x = x[:, [0, -1], :].reshape(-1, 2 * self.d_model)
        x = self.classifier(x)
        return x

# %%
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, fc_dim, bidirectional):
        super(LSTMModel, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=bidirection
        )

        # Fully connected layer
        self.fc = nn.Linear(fc_dim, output_dim)

    def forward(self, x):

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x)
        
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out.flatten(start_dim=-2))

        return out

# %%
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, fc_dim, bidirection):
        super(CNNLSTMModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=30, out_channels=64, kernel_size=4)
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=32,kernel_size=4)


          # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=bidirection
        )

        # Fully connected layer
        self.fc1 = nn.Linear(fc_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
       
        out = self.conv1(x.transpose(1,2))
        out = self.mp(out)
        #out = self.conv2(out)
        #out = self.mp(out)
        
        #out = self.conv2(out)
        #out = self.mp(out)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(out.transpose(1,2))
        
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc1(out.flatten(start_dim=-2))
        out = self.fc2(out)
        #out = self.fc3(out)

        return out

# %%
# class CNNLSTM(nn.Module):
#     def __init__(self, input_features=30, dropout_prob=0.5, padding = 'valid', mp=2, pool='max', input_dim=1, hidden_dim=1, layer_dim=1, output_dim=1, fc_dim=1, bidirection=True):
#         super(CNNLSTM, self).__init__()

#         self.conv1 = nn.Conv1d(input_features, 32, kernel_size=32, stride=2)
#         self.mp = nn.MaxPool1d(mp, stride=mp)

#         self.conv2 = nn.Conv1d(32, 64, kernel_size=16)

#         self.conv3 = nn.Conv1d(64, 128, kernel_size=8)

#         self.conv4 = nn.Conv1d(128, 1, kernel_size=4)
#         self.dropout = nn.Dropout(dropout_prob)

#         self.lstm = nn.LSTM(
#             input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=bidirection
#         )

#         if bidirection:
#             fc_dim = 2*hidden_dim
#         else:
#             fc_dim = hidden_dim

#         # Fully connected layer
#         self.fc = nn.Linear(fc_dim, output_dim)


#     def forward(self, x):
#         x = self.conv1(x.transpose(1,2))
#         # print(x.shape)
#         x = F.relu(x)
#         x = self.mp(x)
#         #print(x.shape)
#         x = self.conv2(x)
#         # print(x.shape)
#         x = F.relu(x)
#         x = self.mp(x)

#         x = self.conv3(x)
#         # print(x.shape)
#         x = F.relu(x)
        
#         x = self.conv4(x)
#         # print(x.shape)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
        
#         x, _ = self.lstm(x)
        
#         x = self.fc(x)
#         return x

class CNNLSTM(nn.Module):
    def __init__(self, input_features=30, dropout_prob=0.5, padding = 'valid', mp=2, pool='max', input_dim=1, hidden_dim=1, layer_dim=1, output_dim=1, fc_dim=1, bidirection=True):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=32, stride=2)
        self.mp = nn.MaxPool1d(mp, stride=mp)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=16)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=8)

        #self.conv4 = nn.Conv1d(128, 1, kernel_size=4)
        self.conv4 = nn.Conv1d(128, 1, kernel_size=4)
        self.dropout = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=bidirection
        )

        if bidirection:
            fc_dim = 2*hidden_dim
        else:
            fc_dim = hidden_dim

        # Fully connected layer
        self.fc = nn.Linear(fc_dim, output_dim)


    def forward(self, x):
        x = self.conv1(x.transpose(1,2))
        # print(x.shape)
        x = F.relu(x)
        x = self.mp(x)
        #print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.mp(x)

        x = self.conv3(x)
        # print(x.shape)
        x = F.relu(x)
        
        x = self.conv4(x)
        
        
        x = self.dropout(x)
        #x = x.transpose(1,2)
        x, _ = self.lstm(x)
        
        x = self.fc(torch.flatten(x,1))
        return x

# %% [markdown]
# # Training

# %%
def train(model, dataloader, optimizer, criterion, epoch, logging_steps=20, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    total_loss = 0
    correct = 0
    loss_history = []
    for i, (inputs, targets) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        outputs = outputs.flatten()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss
        loss_history.append(loss.item())
        preds = torch.round(torch.sigmoid(outputs))
        correct += torch.sum(preds == targets).detach().cpu()

        # if i % logging_steps == 0:
        #     print(f'Epoch: {epoch} ({i}/{len(dataloader)}) Training loss: {loss}')

    accuracy = correct / (dataloader.batch_size * len(dataloader))
    print(f'Epoch {epoch} done. Training loss: {total_loss/len(dataloader)} Training accuracy: {accuracy}')
    return accuracy, total_loss/len(dataloader), loss_history

# %%
def evaluate(model, dataloader, criterion, epoch, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            outputs = outputs.flatten()
            loss = criterion(outputs, targets)

            total_loss += loss
            preds = torch.round(torch.sigmoid(outputs))
            
            correct += torch.sum(preds == targets).detach().cpu()

        accuracy = correct / (dataloader.batch_size * len(dataloader))
        print(f'Epoch {epoch} done. Eval loss: {total_loss/len(dataloader)} Eval accuracy: {accuracy}')
    return accuracy, total_loss/len(dataloader)

# %%
# input_dim = 42
# hidden_dim = 4 
# layer_dim = 1
# output_dim = 1 
# dropout_prob = 0.5
# fc_dim = 8
# bidirection = True
# model = CNNLSTM(input_features=30, dropout_prob=0.5, padding = 'valid', mp=2, pool='max', input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim, fc_dim=fc_dim, bidirection=bidirection)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
# criterion = nn.BCEWithLogitsLoss()
# EPOCHS = 10

# loss_history = []
# train_acc_history = []
# eval_acc_history = []
# eval_loss_history = []
# for epoch in range(EPOCHS):
#     train_acc, train_loss, history = train(model, train_loader, optimizer, criterion, epoch)
#     eval_acc, eval_loss = evaluate(model, test_loader, criterion, epoch)
    
#     loss_history.extend(history)
#     eval_loss_history.append(eval_loss)
#     train_acc_history.append(train_acc)
#     eval_acc_history.append(eval_acc)
    

# %%
# made running the model a function to work with optuna

def run(params=None):
    if params['pool_size'] == 1:
        input_dim = 216
    elif params['pool_size'] == 2:
        input_dim = 42
    else:
        input_dim = 11
    if params['bidirectional'] == 'True':
        bidirection = True
    else:
        bidirection = False
    hidden_dim = 4 
    layer_dim = 1
    output_dim = 1 
    dropout_prob = 0.5
    fc_dim = 8
    model = CNNLSTM(input_features=30, dropout_prob=params['dropout'], padding=params['padding'], mp=params['pool_size'], pool=params['pooling'], input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim, fc_dim=fc_dim, bidirection=bidirection)
    if params:
        print(params)
        optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()
    EPOCHS = params['epochs']

    loss_history = []
    train_acc_history = []
    eval_acc_history = []
    eval_loss_history = []
    for epoch in range(EPOCHS):
        train_acc, train_loss, history = train(model, train_loader, optimizer, criterion, epoch)
        eval_acc, eval_loss = evaluate(model, test_loader, criterion, epoch)
        
        loss_history.extend(history)
        eval_loss_history.append(eval_loss)
        train_acc_history.append(train_acc)
        eval_acc_history.append(eval_acc)
    
    return eval_acc_history[-1]

# run()

# %%
# hyperparameter optimization using optuna

import optuna
from optuna import trial

def objective(trial):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              'padding': trial.suggest_categorical("padding", ["valid", "same"]),
              'pooling': trial.suggest_categorical("pooling", ["max", "avg"]),
              'pool_size': trial.suggest_int("pool_size", 1, 3),
              'dropout': trial.suggest_float("dropout", 0.1, 0.6),
              'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-2),
              'epochs': trial.suggest_int("epochs", 3, 15),
              'bidirectional': trial.suggest_categorical("bidirectional", ["True", "False"]),
              # 'batch_size': trial.suggest_int("batch_size", 1, 101, step=10), 
              # 'num_channel': trial.suggest_int("num_channel", 10, 100),
              # 'kernel_size': trial.suggest_int("kernel_size", 1, 3),
              }
    # batch_size excluded so we don't need to process data every time
    # num_channel and kernel_size excluded to make data sizes easier to deal with/nicer in the
    # convolution channels. Can definitely be added, just let me know if we want this.

    accuracy = run(params)

    return accuracy

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
best_trial = study.best_trial

# print("  Value: {}".format(trial.value))

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

# %%
best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

# %%
import matplotlib.pyplot as plt

# %%
# xs = np.arange(len(train_loader), len(loss_history) + len(train_loader), len(train_loader))

# plt.plot(loss_history, label="loss")
# plt.ylabel("BCE loss")
# plt.legend()
# plt.twinx()
# plt.plot(xs, train_acc_history, label="accuracy", c="C1")
# plt.ylabel("accuracy")
# plt.legend()
# plt.show()

# # %%
# eval_loss_history = [x.cpu() for x in eval_loss_history]
# plt.plot(eval_loss_history, label="loss")
# plt.legend()
# plt.ylabel("BCE loss")
# plt.twinx()
# plt.plot(eval_acc_history, c="C1", label="accuracy")
# plt.ylabel("accuracy")
# plt.legend()
# plt.show()


