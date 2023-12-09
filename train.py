import torch
from torch.utils.data import DataLoader
import math
import numpy as np

from data.dataset import VOCDataset
from models.YOLOV4 import YOLOV4
from utils.loss import YOLOV4Loss
from utils.utils import load_darknet_pretrain_weights
from utils.transforms import DataAugment


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    base_lr = 0.01
    lrf = 0.01
    momentum = 0.937
    weight_decay = 0.0005
    warmup_epochs = 3.0
    nbs = 64  # nominal batch size
    warmup_momentum = 0.8  # warmup initial momentum
    warmup_bias_lr = 0.1  # warmup initial bias lr

    num_epochs = 150
    batch_size = 16
    B, C = 3, 20
    net_size = 416

    jitter = 0.3
    hue = 0.1
    sat = 1.5
    exp = 1.5

    anchors = [[12, 16], [19, 36], [40, 28],
               [36, 75], [76, 55], [72, 146],
               [142, 110], [192, 243], [459, 401]]
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    pretrain = 'pretrain/yolov4.conv.137'
    train_label_list = 'data/voc0712/train.txt'

    print_freq = 5
    save_freq = 5

    # def model
    yolov4 = YOLOV4(B=B, C=C)
    load_darknet_pretrain_weights(yolov4, pretrain)
    yolov4 = yolov4.to(device)

    # def loss
    criterion = YOLOV4Loss(B, C, anchors=anchors, masks=masks, device=device)

    # def optimizer
    optimizer = torch.optim.SGD(yolov4.parameters(), lr=base_lr, momentum=momentum, nesterov=True)
    lf = one_cycle(1, lrf, num_epochs)  # cosine 1->hyp['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # def dataset
    data_transform = DataAugment(jitter=jitter, hue=hue, sat=sat, exp=exp, net_size=net_size)
    train_dataset = VOCDataset(train_label_list, transform=data_transform, net_size=net_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print('Number of training images: ', len(train_dataset))

    nb = len(train_loader)
    nw = max(round(warmup_epochs * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)

    # train
    for epoch in range(num_epochs):
        yolov4.train()
        total_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            # Warmup
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])

            current_lr = get_lr(optimizer)

            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = yolov4(inputs)

            loss = 0.
            for idx, pred in enumerate(preds):
                loss += criterion(pred, targets, idx, inputs.size(2))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print current loss.
            if i % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f, Weight: %.4f'
                      % (epoch, num_epochs, i, len(train_loader), current_lr, loss.item(), total_loss / (i+1), yolov4.conv1.conv.weight[0,0]))

        scheduler.step()
        if epoch % save_freq == 0:
            torch.save(yolov4.state_dict(), 'weights/yolov4_' + str(epoch) + '.pth')

    torch.save(yolov4.state_dict(), 'weights/yolov4_final.pth')