import torch
import torch.nn as nn
from datasets.dataloader import make_train_dataloader
from models.model import ExampleCNN

import os
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
      
parser = argparse.ArgumentParser()
parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'bce', 'focal'], help='Loss function: ce / bce / focal')
args = parser.parse_args()

# training parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 20
learning_rate = 0.01
batch_size = 32
name_suffix = f"loss_{args.loss}"

# data path and weight path
base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "data", "train")
weight_path = os.path.join(base_path, "weights", f"weight_{name_suffix}.pth")

# make dataloader for train data
train_loader, valid_loader = make_train_dataloader(train_data_path, batch_size)

# set cnn model
num_classes = 2 if args.loss == 'ce' else 1
model = ExampleCNN(num_classes)
model = model.to(device)

# set optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
if args.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'bce':
    criterion = nn.BCEWithLogitsLoss()
elif args.loss == 'focal':
    criterion = FocalLoss()

# train
train_loss_list = list()
valid_loss_list = list()
train_accuracy_list = list()
valid_accuracy_list = list()
best = 100
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(epochs):
    print(f'\nEpoch: {epoch+1}/{epochs}')
    print('-' * len(f'Epoch: {epoch+1}/{epochs}'))
    train_loss, valid_loss = 0.0, 0.0
    train_correct, valid_correct = 0, 0
    train_accuracy, valid_accuracy = 0.0, 0.0

    model.train()
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)

        # forward + backward + optimize
        output = model(data)

        if args.loss == 'ce':
            _, preds = torch.max(output.data, 1)
            loss = criterion(output, target)
        else:
            output = output.squeeze(1)         # for BCE/Focal
            target = target.float()            # convert to float for BCE
            pred_labels = (torch.sigmoid(output) > 0.5).long()
            preds = pred_labels
            loss = criterion(output, target)
            
        optimizer.zero_grad()   # zero the parameter gradients
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        train_correct += torch.sum(preds == target.data)
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = float(train_correct) / len(train_loader.dataset)
    train_accuracy_list.append((train_accuracy))

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(valid_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            output = model(data)

            if args.loss == 'ce':
                _, preds = torch.max(output.data, 1)
                loss = criterion(output, target)
            else:
                output = output.squeeze(1)
                target = target.float()
                pred_labels = (torch.sigmoid(output) > 0.5).long()
                preds = pred_labels
                loss = criterion(output, target)

            valid_loss += loss.item() * data.size(0)
            valid_correct += torch.sum(preds == target.data)
        valid_loss /= len(valid_loader.dataset)
        valid_loss_list.append(valid_loss)
        valid_accuracy = float(valid_correct) / len(valid_loader.dataset)
        valid_accuracy_list.append((valid_accuracy))
    
    # print loss and accuracy in one epoch
    print(f'Training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}')
    print(f'Training accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}')

    # record best weight so far
    if valid_loss < best :
        best = valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())
# save the best weight
torch.save(best_model_wts, weight_path)

# plot the loss curve for training and validation
print("\nFinished Training")


pd.DataFrame({
    "train-loss": train_loss_list,
    "valid-loss": valid_loss_list
}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1,epoch+1)
plt.xlabel("Epoch"),plt.ylabel("Loss")
plt.savefig(os.path.join(base_path, "result", f"Loss_curve_{name_suffix}.png"))

# plot the accuracy curve for training and validation
pd.DataFrame({
    "train-accuracy": train_accuracy_list,
    "valid-accuracy": valid_accuracy_list
}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1,epoch+1)
plt.xlabel("Epoch"),plt.ylabel("Accuracy")
plt.savefig(os.path.join(base_path, "result", f"Training_accuracy_{name_suffix}.png"))

