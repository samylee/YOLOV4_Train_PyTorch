import torch
import cv2
import numpy as np

from models.YOLOV4 import YOLOV4
from utils.utils import nms


def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def postprocess(outputs, B, num_classes, anchors, masks, conf_thresh, net_size, scale_x_y, img_w, img_h):
    anchors = torch.from_numpy(np.asarray(anchors, dtype=np.float32))
    masks = torch.from_numpy(np.asarray(masks, dtype=np.int))

    preds = []
    for i, x in enumerate(outputs):
        bs, _, gs, _ =  x.shape
        x = x.view(bs, B, num_classes+5, gs, gs).permute(0, 1, 3, 4, 2).contiguous()
        grid = make_grid(gs, gs)
        anchor_grid = anchors[masks[i]].view(1, -1, 1, 1, 2)
        x[..., 0:2] = (x[..., 0:2].sigmoid() * scale_x_y[i] - 0.5 * (scale_x_y[i] - 1) + grid) / gs
        x[..., 2:4] = torch.exp(x[..., 2:4]) * anchor_grid / net_size
        x[..., 4:] = x[..., 4:].sigmoid()
        x = x.view(bs, -1, num_classes+5)
        preds.append(x)
    predictions = torch.cat(preds, dim=1)

    # for num_samples = 1, ..., (C+5)
    predictions = predictions.squeeze(0)

    # Filter out confidence scores below conf_thresh
    detections = predictions[predictions[:, 4] >= conf_thresh].clone()
    if not detections.size(0):
        return detections

    # conf * classes
    class_confs, class_id = detections[:, 5:].max(1, keepdim=True)
    class_confs *= detections[:, 4].unsqueeze(-1)

    # xywh to xyxy
    detections_cp = detections[:, :4].clone()
    detections[:, 0] = (detections_cp[:, 0] - detections_cp[:, 2] / 2.) * img_w
    detections[:, 1] = (detections_cp[:, 1] - detections_cp[:, 3] / 2.) * img_h
    detections[:, 2] = (detections_cp[:, 0] + detections_cp[:, 2] / 2.) * img_w
    detections[:, 3] = (detections_cp[:, 1] + detections_cp[:, 3] / 2.) * img_h

    return torch.cat((detections[:, :4], class_confs.float(), class_id.float()), 1)


def preprocess(img, net_size):
    # img bgr2rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize img
    img_resize = cv2.resize(img_rgb, (net_size, net_size))

    # norm img
    img_resize = torch.from_numpy(img_resize.transpose((2, 0, 1)))
    img_norm = img_resize.float().div(255).unsqueeze(0)
    return img_norm


def model_init(model_path, B=3, C=20):
    # load moel
    model = YOLOV4(B=B, C=C)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


if __name__ == '__main__':
    # load moel
    checkpoint_path = 'weights/yolov4_140.pth'
    B, C = 3, 20
    model = model_init(checkpoint_path, B, C)

    # params init
    net_size = 416
    anchors = [[12, 16], [19, 36], [40, 28],
               [36, 75], [76, 55], [72, 146],
               [142, 110], [192, 243], [459, 401]]
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    thresh = 0.25
    iou_thresh = 0.45
    scale_x_y = [1.2, 1.1, 1.05]

    # voc
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # load img
    img = cv2.imread('demo/horses.jpg')
    img_h, img_w, _ = img.shape

    # preprocess
    img_norm = preprocess(img, net_size)

    # forward
    outputs = model(img_norm)

    # postprocess
    results = postprocess(outputs, B, C, anchors, masks, thresh, net_size, scale_x_y, img_w, img_h)

    if results.size(0) > 0:
        # nms
        results = nms(results.data.cpu().numpy(), iou_thresh)

        # show
        for i in range(results.shape[0]):
            cv2.rectangle(img, (int(results[i][0]), int(results[i][1])), (int(results[i][2]), int(results[i][3])),
                          (0, 255, 0), 2)
            cv2.putText(img, classes[int(results[i][5])] + '-' + str(round(results[i][4], 4)),
                        (int(results[i][0]), int(results[i][1])), 0, 0.6, (0, 255, 255), 2)

    # cv2.imwrite('assets/result3.jpg', img)
    cv2.imshow('demo', img)
    cv2.waitKey(0)
