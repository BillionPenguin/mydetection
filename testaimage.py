from torchvision.datasets.coco import CocoDetection
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

import pdetection.utils as util

def getODtarget(source):
    target = []
    for t in source:
        this = {}
        this['image_id'] = t['image_id']
        this['bbox'] = t['bbox']
        this['category_id'] = t['category_id']
        this['id'] = t['id']
        target.append(this)
    return target

path_data = '/home/peng/Documents/srch/Object Detection/dataset/coco/val2017'
path_anno = '/home/peng/Documents/srch/Object Detection/dataset/coco/annotations_trainval2017/annotations/instances_val2017.json'

coco = COCO(path_anno)
cats = coco.loadCats(coco.getCatIds())

coco_dset = CocoDetection(root=path_data, annFile=path_anno)
# img, target = coco_dset[2]
img = Image.open('/home/peng/Downloads/joker.jpg')
npimg = np.asarray(img)
tensorimg = transforms.ToTensor()(npimg)
print(type(img), npimg.shape, tensorimg.shape)
testmod = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91)
testmod.eval()
out = testmod([tensorimg])
img.show()
resimg = img.copy()
catsname = util.idtocname(out[0]['labels'], coco)
print(catsname)
util.drawRects(resimg, out[0]['boxes'], catsname, out[0]['scores'])
resimg.show()
print(len(out[0]['boxes']), len(out[0]['labels']), len(out[0]['scores']))

# print(len(cats))