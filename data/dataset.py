import torch
import numpy as np
import cv2
import os
import random
from torch.utils.data import Dataset

from utils.utils import xywh2xyxy, xyxy2xywh, xywhn2xyxy, random_perspective


class VOCDataset(Dataset):
    def __init__(self, label_list, transform=None, net_size=416):
        super(VOCDataset, self).__init__()
        self.transform = transform
        self.mosaic = True
        self.max_objects = 30
        self.net_size = net_size
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]

        with open(label_list, 'r') as f:
            image_path_lines = f.readlines()

        self.images_path = []
        self.labels = []
        for image_path_line in image_path_lines:
            image_path = image_path_line.strip().split()[0]
            label_path = image_path.replace('JPEGImages', 'labels').replace('jpg', 'txt')
            if not os.path.exists(label_path):
                continue

            self.images_path.append(image_path)
            with open(label_path, 'r') as f:
                label_lines = f.readlines()

            labels_tmp = np.empty((len(label_lines), 5), dtype=np.float32)
            for i, label_line in enumerate(label_lines):
                labels_tmp[i] = [float(x) for x in label_line.strip().split()]
            self.labels.append(labels_tmp)

        assert len(self.images_path) == len(self.labels), 'images_path\'s length dont match labels\'s length'
        self.indices = range(len(self.images_path))

    def __getitem__(self, idx):
        # mosaic data augment
        image, labels = None, None
        have_labels = False
        while not have_labels:
            if self.mosaic and random.random() < 0.5:
                image, labels = self.load_mosaic(idx)
            else:
                image, labels = self.load_origin(idx)
            have_labels = True if labels.size else False

        img_h, img_w, _ = image.shape
        labels = xyxy2xywh(labels, img_w, img_h)

        # to torch
        inputs = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)

        targets = torch.zeros((self.max_objects, 5), dtype=torch.float32)
        if len(labels) < self.max_objects:
            targets[:len(labels), :] = torch.from_numpy(labels)
        else:
            targets = torch.from_numpy(labels)[:self.max_objects, :]

        return inputs, targets

    def __len__(self):
        return len(self.images_path)

    def load_image(self, index, i_mixup, mosaic):
        image = cv2.cvtColor(cv2.imread(self.images_path[index]), cv2.COLOR_BGR2RGB)
        labels = self.labels[index]
        img_h, img_w, _ = image.shape
        if self.transform:
            labels = xywh2xyxy(labels, img_w, img_h)
            image, labels = self.transform(image, labels, i_mixup, mosaic)
            img_h, img_w, _ = image.shape
            labels = xyxy2xywh(labels, img_w, img_h)

        # resize
        image = cv2.resize(image, (self.net_size, self.net_size), interpolation=random.choice(self.interps))
        return image, image.shape[:2], labels

    def load_mosaic(self, index):
        labels4 = []
        mosaic_border = [-self.net_size // 2, -self.net_size // 2]
        yc, xc = [int(random.uniform(-x, 2 * self.net_size + x)) for x in mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)
        # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, (h, w), labels = self.load_image(index, i_mixup=i, mosaic=True)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((self.net_size * 2, self.net_size * 2, img.shape[2]), 114,
                               dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.net_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.net_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.net_size * 2), min(self.net_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if labels.size:
                labels = xywhn2xyxy(labels, w, h, padw, padh)
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in labels4[:, 1:]:
            np.clip(x, 0, 2 * self.net_size, out=x)

        # Augment
        img4, labels4 = random_perspective(img4, labels4,
                                           degrees=0,
                                           translate=0.2,
                                           scale=0.5,
                                           shear=0,
                                           perspective=0,
                                           border=mosaic_border)  # border to remove

        return img4, labels4

    def load_origin(self, index):
        img, (h, w), labels = self.load_image(index, i_mixup=-1, mosaic=False)
        if labels.size:  # normalized xywh to pixel xyxy format
            labels = xywh2xyxy(labels, w, h)

        return img, labels