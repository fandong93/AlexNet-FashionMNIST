
import os
import torch
import torch.nn as nn
from model import AlexNet
import torch.optim as optim
import matplotlib.pyplot as plt
from load_data import Load
import time
import train_module
import valid_module


def main():
    # device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    # 创建模型，部署 gpu
    model = AlexNet(num_class=10, init_weights=True).to(device)
    # 交叉熵损失
    loss_function = nn.CrossEntropyLoss().to(device)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 载入数据
    load = Load()
    train_loader, valid_loader = load.load_data(64, 1000, "./datasets")
    # 调用
    epochs = 50
    best_acc = 0.0
    max_loss = 0.0
    min_loss = 1.0
    max_acc = 0.0
    min_acc = 1.0

    Loss = []
    Accuracy = []

    train = train_module.Train()
    valid = valid_module.Valid()

    save_path = './model/AlexNet-FashionMNIST.pth'
    if not os.path.exists("./model"):
        os.mkdir("./model")

    img_path = './img/AlexNet-FashionMNIST.jpg'
    if not os.path.exists("./img"):
        os.mkdir("./img")

    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    for epoch in range(0, epochs + 1):
        loss, train_acc = train.train_method(model, device, train_loader, loss_function, optimizer, epoch)

        if loss > max_loss:
            max_loss = loss
        if loss < min_loss:
            min_loss = loss

        if train_acc > max_acc:
            max_acc = train_acc
        if train_acc < min_acc:
            min_acc = train_acc

        valid_acc = valid.valid_method(model, device, valid_loader, epoch)
        Loss.append(loss)
        Accuracy.append(train_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), save_path)  # pytorch 中的 state_dict 是一个简单的 python 的字典对象，将每一层与它的对应参数建立映射关系。

    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    print('Finished Training')

    plt.subplot(2, 1, 1)
    plt.plot(Loss)
    plt.title('Loss')
    x_ticks = torch.arange(0, epochs + 1, 10)
    plt.xticks(x_ticks)
    y_ticks = torch.arange(min_loss, max_loss + 0.1, 0.2)
    plt.yticks(y_ticks)

    plt.subplot(2, 1, 2)
    plt.plot(Accuracy)
    plt.title('Accuracy')
    x_ticks = torch.arange(0, epochs + 1, 10)
    plt.xticks(x_ticks)
    y_ticks = torch.arange(min_acc, max_acc, 0.03)
    plt.yticks(y_ticks)

    plt.subplots_adjust(hspace=0.3)  # 调整子图间距
    plt.savefig(img_path)
    plt.show()


if __name__ == '__main__':
    main()
