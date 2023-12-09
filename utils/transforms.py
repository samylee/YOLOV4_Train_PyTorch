import cv2
import numpy as np
import random


class DataAugment:
    def __init__(self, jitter=0.3, hue=0.1, sat=1.5, exp=1.5, net_size=416):
        self.jitter = jitter
        self.lowest_w, self.lowest_h = 1.0 / net_size, 1.0 / net_size
        self.hue, self.sat, self.exp = hue, sat, exp

    def __call__(self, image, labels, i_mixup, mosaic):
        oh, ow, _ = image.shape
        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        r3 = random.uniform(0, 1)
        r4 = random.uniform(0, 1)

        pleft = self.rand_precalc_random(-dw, dw, r1)
        pright = self.rand_precalc_random(-dw, dw, r2)
        ptop = self.rand_precalc_random(-dh, dh, r3)
        pbot = self.rand_precalc_random(-dh, dh, r4)

        # for mosaic
        if mosaic:
            if i_mixup == 0:
                pleft += pright; pright = 0; ptop += pbot; pbot = 0
            elif i_mixup == 1:
                pright += pleft; pleft = 0; ptop += pbot; pbot = 0
            elif i_mixup == 2:
                pleft += pright; pright = 0; pbot += ptop; ptop = 0
            else:
                pright += pleft; pleft = 0; pbot += ptop; ptop = 0

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot

        # flip
        if random.random() < 0.5:
            image, labels = self.flip(image, labels)

        # crop+hsv
        cropped = self.crop_image(image, pleft, ptop, swidth, sheight)
        cropped = self.hsv(cropped)

        crop_h, crop_w, _ = cropped.shape

        # limit labels
        labels_out = labels.copy()
        shift = np.array([pleft, ptop, pleft, ptop])  # [m, 4]
        labels_out[:, 1:] = labels_out[:, 1:] - shift
        labels_out[:, 1] = labels_out[:, 1].clip(min=0, max=crop_w)
        labels_out[:, 2] = labels_out[:, 2].clip(min=0, max=crop_h)
        labels_out[:, 3] = labels_out[:, 3].clip(min=0, max=crop_w)
        labels_out[:, 4] = labels_out[:, 4].clip(min=0, max=crop_h)

        mask_w = ((labels_out[:, 3] - labels_out[:, 1]) / crop_w > self.lowest_w)
        mask_h = ((labels_out[:, 4] - labels_out[:, 2]) / crop_h > self.lowest_h)
        labels_out = labels_out[mask_w & mask_h]

        return cropped, labels_out

    def rand_precalc_random(self, min, max, random_part):
        if max < min:
            swap = min
            min = max
            max = swap
        return int((random_part * (max - min)) + min)

    def intersect_rect(self, rect1, rect2):
        x = max(rect1[0], rect2[0])
        y = max(rect1[1], rect2[1])
        width = min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - x
        height = min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - y
        return [x, y, width, height]

    def random_scale(self):
        dhue = random.uniform(-self.hue, self.hue)
        scale = random.uniform(1., self.sat)
        if random.randrange(2):
            dsat = scale
        else:
            dsat = 1. / scale
        scale = random.uniform(1., self.exp)
        if random.randrange(2):
            dexp = scale
        else:
            dexp = 1. / scale
        return dhue, dsat, dexp

    def crop_image(self, img, pleft, ptop, swidth, sheight):
        oh, ow, _ = img.shape
        if pleft == 0 and ptop == 0 and swidth == ow and sheight == oh:
            return img
        else:
            src_rect = [pleft, ptop, swidth, sheight]
            img_rect = [0, 0, ow, oh]
            new_src_rect = self.intersect_rect(src_rect, img_rect)
            assert new_src_rect[2] > 0 and new_src_rect[3] > 0, 'no intersect'
            dst_rect = [max(0, -pleft), max(0, -ptop), new_src_rect[2], new_src_rect[3]]

            img_mean = cv2.mean(img)
            cropped_img = np.empty((sheight, swidth, 3), dtype=np.uint8)
            cropped_img[:, :] = img_mean[:3]
            cropped_img[dst_rect[1] : dst_rect[1] + dst_rect[3], dst_rect[0] : dst_rect[0] + dst_rect[2], :] = \
                img[new_src_rect[1] : new_src_rect[1] + new_src_rect[3], new_src_rect[0] : new_src_rect[0] + new_src_rect[2], :]

            return cropped_img

    def flip(self, image, labels):
        _, width, _ = image.shape
        image = np.ascontiguousarray(image[:, ::-1])
        labels_cp = labels.copy()
        labels[:, 1::2] = width - labels_cp[:, 3::-2]
        return image, labels

    def hsv(self, image):
        dhue, dsat, dexp = self.random_scale()
        if dsat != 1 or dexp != 1 or dhue != 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            h = h + 179 * dhue
            h = np.clip(h, 0, 255).astype(hsv.dtype)

            s = s * dsat
            s = np.clip(s, 0, 255).astype(hsv.dtype)

            v = v * dexp
            v = np.clip(v, 0, 255).astype(hsv.dtype)

            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return image
