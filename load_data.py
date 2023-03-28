
import sys
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt


class Load:
    def load_data(self, train_size, valid_size, path):
        if sys.platform.startswith('win'):
            num_workers = 0
        else:
            num_workers = 4

        # 数据增广方法
        dataset_transforms = transforms.Compose([transforms.Resize(224),     # 随机裁剪至 224 x 224
                                                 transforms.ToTensor(),                 # 转换至 Tensor
                                                 transforms.Normalize(0.5, 0.5)])       # 归一化

        # 50000 张训练图片
        train_set = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=dataset_transforms)
        # print("train_set.data.shape", train_set.data.shape)

        train_loader = data.DataLoader(train_set, batch_size=train_size, shuffle=True, num_workers=num_workers)
        # batch = next(iter(train_loader))
        # print(len(batch))
        # images, labels = batch
        # print("images.shape: ", images.shape)
        # print("labels.shape: ", labels.shape)

        # 10000 张验证图片
        test_set = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=dataset_transforms)
        test_loader = data.DataLoader(test_set, batch_size=valid_size, shuffle=True, num_workers=num_workers)

        # 分类
        classes = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

        matplotlib.use('qt5agg')
        # 查看数据
        figure = plt.figure(figsize=(7, 5))
        cols, rows = 6, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(train_set), size=(1,)).item()  # 从数据集中随机采样
            img, label = train_set[sample_idx]  # 取得数据集的图和标签
            figure.add_subplot(rows, cols, i)   # 画子图，也可以 plt.subplot(rows, cols, i)
            img = img.numpy().transpose((1, 2, 0))
            plt.title(classes[label])
            plt.axis("off")
            plt.imshow((img + 1) / 2)  # (img + 1) / 2 是为了还原被归一化的数据
        plt.show()

        return train_loader, test_loader

# # 测试
# load = Load()
# train_loader, valid_loader = load.load_data(64, 1000, "./datasets")
