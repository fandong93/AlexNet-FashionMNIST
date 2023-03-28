
import torch
import sys
from tqdm import tqdm


class Valid:
    def valid_method(self, model, device, valid_loader, epoch):
        # 模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
        # 因为调用 eval() 将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
        model.eval()
        # 统计模型正确率, 设置初始值
        total = 0
        correct = 0.0
        # torch.no_grad 将不会计算梯度, 也不会进行反向传播
        with torch.no_grad():
            val_bar = tqdm(valid_loader, file=sys.stdout)  # file=sys.stdout 输出到控制台
            for step, data in enumerate(val_bar):
                image, label = data
                image, label = image.to(device), label.to(device)
                output = model(image)
                predict = output.argmax(dim=1)
                # 计算正确数量
                total += label.size(0)
                correct += torch.eq(predict, label).sum().item()
                val_bar.desc = 'Valid Epoch {}, Acc {:.3f}%'.format(epoch, 100 * (correct / total))

        return correct / total
