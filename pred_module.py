
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from model import AlexNet
import matplotlib.pyplot as plt
import torch.nn.functional as F


def main():
    # 数据增广方法
    dataset_transforms = transforms.Compose([transforms.Resize(224),  # 随机裁剪至 224 x 224
                                             transforms.ToTensor(),  # 转换至 Tensor
                                             transforms.Normalize(0.5, 0.5)])

    # 50000 张训练图片
    train_set = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=dataset_transforms)
    # 分类
    classes = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
    # 从数据集中随机采样
    sample_idx = torch.randint(len(train_set), size=(1,)).item()
    img, label = train_set[sample_idx]
    # 显示图片
    plt.figure("Image")
    image = img.numpy().transpose((1, 2, 0))
    plt.imshow(image)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title(classes[label])
    plt.show()

    # dataset_transforms(img)
    # print(img.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet().to(device)
    model.load_state_dict(torch.load('./model/AlexNet-FashionMNIST.pth'))  # 加载模型
    model.eval()  # 把模型转为 valid 模式

    # 预测
    output = model(img.to(device))
    predict = output.argmax(dim=1)
    pred_class = classes[predict.item()]
    print("预测类别：", pred_class)
    print(classes[label] == pred_class)


if __name__ == '__main__':
    main()
