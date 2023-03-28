
import sys
import torch
from tqdm import tqdm


class Train:
    def train_method(self, model, device, train_loader, loss_func, optimizer, epoch):

        model.train()
        model = model.to(device)

        num = 0
        total = 0
        correct = 0.0
        sum_loss = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)  # file=sys.stdout 输出到控制台
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            predict = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += torch.eq(predict, labels).sum().item()
            sum_loss += loss.item()
            num = step
            loss.backward()
            optimizer.step()
            train_bar.desc = 'Train Epoch {:d}, Loss {:.4f}, Acc {:.3f}%'.format(epoch, loss.item(), 100 * (correct / total))

        return sum_loss / num, correct / total
