import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional
from torchvision.models.detection.image_list import ImageList
class ODTransformer(nn.Module):
    def __init__(self, min_size, max_size, means, stds):
        super(ODTransformer, self).__init__()
        self.minsize = min_size
        self.maxsize = max_size
        self.means = means
        self.stds = stds

    def forward(self, images, targets=None):
        for i in range(len(images)):
            if targets == None:
                img, target = self.rescale(images[i])
            else:
                img, target = self.rescale(images[i], targets[i])
            img = self.normalize(img)
            images[i] = img
            if targets:
                targets[i] = target
        images_batched = self.batchimages(images)
        imgs_size = [img.shape[-2:] for img in images]
        imgs_list = ImageList(images_batched, imgs_size)
        if targets:
            return imgs_list, targets
        return imgs_list

    def rescale(self, image, target=None):
        h, w = image.shape[-2:]
        mins = min(h, w)
        maxs = max(h, w)
        scale_factor = self.minsize / mins
        if maxs * scale_factor > self.maxsize:
            scale_factor = self.maxsize / maxs
        image = functional.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)
        if target != None:
            target['boxes'] = self.rescale_box(target['boxes'], (h, w), image.shape[-2:])
        return image, target

    def rescale_box(self, boxes, srchw, trghw):
        ratio_h, ratio_w = trghw[0]/srchw[0], trghw[1]/srchw[1]
        x1, y1, x2, y2 = boxes.unbind(1)
        x1, x2 = x1 * ratio_w, x2 * ratio_w
        y1, y2 = y1 * ratio_h, y2 * ratio_h
        return torch.stack((x1, y1, x2, y2), 1)

    def normalize(self, img):
        dtype, device = img.dtype, img.device
        means = torch.as_tensor(self.means, dtype=dtype, device=device).view(3, 1, 1)
        stds = torch.as_tensor(self.stds, dtype=dtype, device=device).view(3, 1, 1)
        img = (img - means) / stds
        return img

    def batchimages(self, imgs, divisible=32):
        imgs_size = [img.shape[-2:] for img in imgs]
        h, w = zip(*imgs_size)
        max_h = int(math.ceil(max(h) / divisible) * divisible)
        max_w = int(math.ceil(max(w) / divisible) * divisible)
        batched_imgs = torch.zeros((len(imgs), 3, max_h, max_w), dtype=imgs[0].dtype, device=imgs[0].device)
        for i in range(len(imgs)):
            batched_imgs[i, :, :imgs_size[i][0], :imgs_size[i][1]] = imgs[i]
        return batched_imgs

    def postprocess(self, targets, srchws, trghws):
        for i in range(len(targets)):
            targets[i]['boxes'] = self.rescale_box(targets[i]['boxes'], srchws[i], trghws[i])
        return targets
            # for dict in boxes[i]:
            #     srcx1, srcy1, srcx2, srcy2 = dict['boxes']
            #     trgx1, trgy1, trgx2, trgy2 = srcx1 * ratio_w, srcy1 * ratio_h, srcx2 * ratio_w, srcy2 * ratio_h
            #     dict['boxes']