import torch
import torch.nn as nn
import numpy as np
from utils.utils import bbox_iou


class YOLOV4Loss(nn.Module):
    def __init__(self, B=3, C=20, anchors=None, masks=None, device=None):
        super(YOLOV4Loss, self).__init__()
        self.device = device
        self.B, self.C = B, C
        self.anchors = torch.from_numpy(np.asarray(anchors, dtype=np.float32)).to(self.device)
        self.masks = torch.from_numpy(np.asarray(masks, dtype=np.int)).to(self.device)
        self.ignore_thresh = 0.7
        self.iou_thresh = 0.213
        self.class_thresh = 0.25
        self.scale_x_y = [1.2, 1.1, 1.05]
        self.iou_normalizer = 0.07
        self.obj_normalizer = 1.0
        self.cls_normalizer = 1.0

        self.class_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.noobj_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.obj_criterion = nn.MSELoss(reduction='sum').to(self.device)

    def make_grid(self, nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, preds, targets, idx, img_size):
        batch_size, _, grid_size, _ = preds.shape

        # num_samples, 3(anchors), 13(grid), 13(grid), 25 (tx, ty, tw, th, conf, classes)
        preds_permute = (
            preds.view(batch_size, self.B, self.C+5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # anchors, grids
        grid = self.make_grid(grid_size, grid_size).to(self.device)
        anchor_grid = self.anchors[self.masks[idx]].view(1, -1, 1, 1, 2)

        # tx, ty, tw, th
        preds_txty = (torch.sigmoid(preds_permute[..., 0:2]) * self.scale_x_y[idx] - 0.5 * (self.scale_x_y[idx] - 1)  + grid) / grid_size
        preds_twth = torch.exp(preds_permute[..., 2:4]) * anchor_grid / img_size
        preds_bbox = torch.cat((preds_txty, preds_twth), dim=-1)

        # conf, class
        preds_conf = torch.sigmoid(preds_permute[..., 4])
        preds_class = torch.sigmoid(preds_permute[..., 5:])

        # get noobj mask
        noobj_mask = self.get_noobj_mask(preds_bbox, targets)
        noobj_mask = self.compare_yolo_class(noobj_mask, preds_class)

        # get targets
        noobj_mask, obj_mask, targets_class, targets_bbox = self.build_targets(noobj_mask, preds_bbox, preds_conf, preds_class, targets, idx, img_size)

        # 1. noobj loss
        preds_conf_noobj_mask = preds_conf[noobj_mask]
        loss_noobj = self.noobj_criterion(preds_conf_noobj_mask, torch.zeros_like(preds_conf_noobj_mask))

        # 2. obj loss
        preds_conf_obj_mask = preds_conf[obj_mask]
        loss_obj = self.obj_criterion(preds_conf_obj_mask, torch.ones_like(preds_conf_obj_mask))

        # 3. class loss
        class_mask = obj_mask.unsqueeze(-1).expand_as(preds_class)
        preds_class_mask = preds_class[class_mask]
        targets_class_mask = targets_class[class_mask]
        loss_class = self.class_criterion(preds_class_mask, targets_class_mask)

        # 4. bbox loss
        bbox_mask = obj_mask.unsqueeze(-1).expand_as(preds_bbox)
        preds_bbox_mask = preds_bbox[bbox_mask].view(-1, 4)
        targets_bbox_mask = targets_bbox[bbox_mask].view(-1, 4)
        loss_bbox = 1. - bbox_iou(preds_bbox_mask, targets_bbox_mask, CIoU=True)

        # 5. average classes in one box
        targets_class_mask_ave = targets_class_mask.view(-1, self.C).sum(dim=-1)
        assert loss_bbox.shape == targets_class_mask_ave.shape, 'loss_bbox shape must equal to targets_class_mask_ave shape'
        loss_bbox = (loss_bbox / targets_class_mask_ave).sum()

        loss = self.cls_normalizer * loss_class + self.obj_normalizer * (loss_obj + loss_noobj) + self.iou_normalizer * loss_bbox

        return loss / batch_size

    def get_noobj_mask(self, preds_bbox, targets):
        batch_size, _, grid_size, _, _ = preds_bbox.shape
        noobj_mask_list = []
        for b in range(batch_size):
            preds_bbox_batch = preds_bbox[b]
            targets_bbox_batch = targets[b, :, 1:]
            preds_ious_list = []
            for target_bbox_batch in targets_bbox_batch:
                if target_bbox_batch[0] == 0:
                    break
                preds_ious_noobj = bbox_iou(preds_bbox_batch.view(-1, 4), target_bbox_batch).view(self.B, grid_size, grid_size)
                preds_ious_list.append(preds_ious_noobj)
            preds_ious_tensor = torch.stack(preds_ious_list, dim=0)
            preds_ious_max = torch.max(preds_ious_tensor, dim=0)[0]

            noobj_mask_batch = preds_ious_max <= self.ignore_thresh
            noobj_mask_list.append(noobj_mask_batch)
        noobj_mask = torch.stack(noobj_mask_list)
        return noobj_mask

    def compare_yolo_class(self, noobj_mask, preds_class):
        yolo_class = torch.any(preds_class > self.class_thresh, dim=-1)
        noobj_mask = ~((~noobj_mask) & yolo_class)
        return noobj_mask

    def build_targets(self, noobj_mask, preds_bbox, preds_conf, preds_class, targets, idx, img_size):
        batch_size, _, grid_size, _, _ = preds_bbox.shape

        obj_mask = torch.empty_like(preds_conf, dtype=torch.bool, requires_grad=False).fill_(False)
        targets_class = torch.zeros_like(preds_class, requires_grad=False)
        targets_bbox = torch.zeros_like(preds_bbox, requires_grad=False)
        for b in range(batch_size):
            targets_batch = targets[b]
            for target_batch in targets_batch:
                target_class_batch = int(target_batch[0])
                assert target_class_batch < self.C, 'oh shit'
                target_bbox_batch = target_batch[1:]
                if target_bbox_batch[0] == 0:
                    break
                i = int(target_bbox_batch[0] * grid_size)
                j = int(target_bbox_batch[1] * grid_size)

                target_bbox_batch_shift = torch.zeros((1, 4), dtype=torch.float32).to(self.device)
                target_bbox_batch_shift[0, 2:] = target_bbox_batch[2:]

                anchors_match_batch_shift = torch.zeros((self.anchors.size(0), 4), dtype=torch.float32).to(self.device)
                anchors_match_batch_shift[:, 2:] = self.anchors / img_size

                anchors_ious = bbox_iou(anchors_match_batch_shift, target_bbox_batch_shift)

                # get best obj index
                anchors_ious_index = torch.max(anchors_ious, dim=0)[1]
                if anchors_ious_index in self.masks[idx]:
                    find_obj_index = torch.nonzero(self.masks[idx] == anchors_ious_index)[0]

                    noobj_mask[b, find_obj_index, j, i] = False
                    obj_mask[b, find_obj_index, j, i] = True
                    targets_class[b, find_obj_index, j, i, target_class_batch] = 1.0
                    targets_bbox[b, find_obj_index, j, i] = target_bbox_batch

                # get other obj index for yolov4 legend
                for n in self.masks[idx]:
                    if n != anchors_ious_index and anchors_ious[n] > self.iou_thresh:
                        find_obj_index = torch.nonzero(self.masks[idx] == n)[0]

                        noobj_mask[b, find_obj_index, j, i] = False
                        obj_mask[b, find_obj_index, j, i] = True
                        targets_class[b, find_obj_index, j, i, target_class_batch] = 1.0
                        targets_bbox[b, find_obj_index, j, i] = target_bbox_batch

        return noobj_mask, obj_mask, targets_class, targets_bbox