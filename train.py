import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import wandb

from functions import (
    UCF101VideoDataset,
    WaveVideoClassifier,
)


# ---------------------- path ----------------------
data_root = "./"
train_csv = "./train.csv"
val_csv   = "./val.csv"
save_model_path = "./Conv3D_ckpt/"
os.makedirs(save_model_path, exist_ok=True)

# ---------------------- hyper-parameter ----------------------
epochs        = 40
batch_size    = 48
learning_rate = 1e-4
log_interval  = 10
img_x, img_y  = 112,112





# ---------------------- train ----------------------
def train(log_interval, model, device, train_loader, optimizer, epoch):

    model.train()

    losses = []
    scores = []
    N_count = 0   

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()


        # ✅ 第一个batch，第一个epoch，立刻检查梯度
        if batch_idx == 0 and epoch == 0:
            print("\n===== 梯度检查 =====")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad_norm={param.grad.norm():.8f}, param_norm={param.data.norm():.8f}")
                else:
                    print(f"{name}: 无梯度！")
            print("====================\n")



        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores



# ---------------------- validation func ----------------------
def validation(model, device, optimizer, test_loader, epoch):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_latest.pth'))
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score



# ---------------------- devices ----------------------
# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 8, 'pin_memory': True} if use_cuda else {}



# ---------------------- data ----------------------
print("data pending ...")
train_set = UCF101VideoDataset(
    csv_path=train_csv,
    data_root=data_root,
    frames_per_clip=28,
    img_x=img_x,
    img_y=img_y
)
valid_set = UCF101VideoDataset(
    csv_path=val_csv,
    data_root=data_root,
    frames_per_clip=28,
    img_x=img_x,
    img_y=img_y
)

k = len(train_set.classes)
print(f"num of classes by dectect: {k}")
print(f"the size of train set: {len(train_set)}, the size of valid set: {len(valid_set)}")

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)


print("\n===== data sample check =====")
for i in range(5):
    x, y = train_set[i]
    print(f"sample {i}: label={y}, tensor_sum={x.sum():.4f}, tensor_max={x.max():.4f}")
print("========================\n")






# ---------------------- model ----------------------
# create model
myModel = WaveVideoClassifier(in_channels=3, hidden_dim=128, num_classes=k).to(device)


# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    myModel = nn.DataParallel(myModel)






# ---------------------- optimizer ----------------------
optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)   # optimize all cnn parameters




# ---------------------- WandB ----------------------
print("Connecting WandB ...")
wandb.init(
    project="3DWaveformer-ucf101",
    name="classifier-1",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "architecture": "WaveVideoClassifier",
        "dataset": "UCF101",
        "num_classes": k
    }
)




# ---------------------- train loop ----------------------

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, myModel, device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation(myModel, device, optimizer, valid_loader, epoch)

    # ✅ 加在这里，每个epoch结束后打印
    print(f"\n当前lr: {optimizer.param_groups[0]['lr']}")
    for name, param in myModel.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm():.6f}")

    # ✅ 数据检查，只在第一个epoch结束后跑一次
    if epoch == 0:
        print("\n===== 数据样本检查 =====")
        for i in range(5):
            x, y = train_set[i]
            print(f"样本{i}: label={y}, tensor_sum={x.sum():.4f}, tensor_max={x.max():.4f}")
        print("========================\n")

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./3Dwaveformer_epoch_training_losses.npy', A)
    np.save('./3Dwaveformer_epoch_training_scores.npy', B)
    np.save('./3Dwaveformer_epoch_test_loss.npy', C)
    np.save('./3Dwaveformer_epoch_test_score.npy', D)

    wandb.log({
        "epoch": epoch + 1,
        "Train/Loss": train_losses[-1],
        "Train/Accuracy": train_scores[-1],
        "Val/Loss": epoch_test_loss,
        "Val/Accuracy": epoch_test_score
    })








# ---------------------- plot ----------------------
fig = plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
plt.savefig("./fig_3Dwaveformer.png", dpi=600)
# plt.close(fig)
plt.show()

wandb.finish()