import time
import torch
import copy
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from main import VisionTransformer
import torch.nn as nn
import pandas as pd
import os

def train_val_data():
    path = r'dataset\train'
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_data = ImageFolder(path, transform=train_transform)

    train_data, val_data = data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])

    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_data, batch_size=32, shuffle=True, num_workers=2)

    return train_loader, val_loader

def train_val_model(model, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-"*10)

        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()
            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1) #下标
            loss = criterion(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("train:{} loss:{:.4f} acc:{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("val:{} loss:{:.4f} acc:{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model = copy.deepcopy(model.state_dict())

        time_used = time.time() - since
        print("{:.0f}m{:.0f}s".format(time_used // 60, time_used % 60))

    torch.save(best_model, 'C:/Users/Eddie/Desktop/tran/best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_loss_all": val_loss_all,
                                       "val_acc_all": val_acc_all})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

def load_pretrained_weights(model, ckpt_path, verbose=True):
    assert os.path.isfile(ckpt_path), f"找不到权重文件: {ckpt_path}"

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            state_dict = obj["model"]
        elif isinstance(obj, dict) and "state_dict" in obj:
            state_dict = obj["state_dict"]
        else:
            state_dict = obj

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if all(k.startswith("vit.") for k in state_dict.keys()):
        state_dict = {k[len("vit."):]: v for k, v in state_dict.items()}

    model_dict = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if verbose:
        print("预训练权重加载完成")
        if skipped:
            print("因 shape 不匹配/不存在而跳过（正常，通常只有 head）:")
            for k in skipped:
                print("   -", k)
        if missing:
            print("模型中缺失的 key（会被随机初始化）:")
            for k in missing:
                print("   -", k)
        if unexpected:
            print("多余 key:")
            for k in unexpected:
                print("   -", k)
    return model

if __name__ == '__main__':
    model = VisionTransformer()

    model = load_pretrained_weights(
        model,
        r"C:\Users\Eddie\Downloads\model.safetensors",
    )

    train_loader, val_loader = train_val_data()
    train_process = train_val_model(model, train_loader, val_loader, 5)
    matplot_acc_loss(train_process)