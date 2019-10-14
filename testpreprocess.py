from matplotlib import image, pyplot
from os import listdir
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection.transform import GeneralizedRCNNTransform
path = '/home/peng/Documents/srch/Object Detection/dataset/coco/train2017'
transformer = GeneralizedRCNNTransform(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
picnum = 0
for filename in listdir(path):
    img = image.imread(path + '/' + filename)
    print('img:',img.dtype ,',' ,img.shape)
    torchimg = transforms.ToTensor()(img)
    print('torchimg:', torchimg.dtype, ',', torchimg.shape)
    post = transformer([torchimg])[0].tensors[0]
    postimg = transforms.ToPILImage()(post).convert('RGB')
    print('postimg:', img.dtype)
    pyplot.subplot(211)
    pyplot.imshow(img)
    pyplot.subplot(212)
    pyplot.imshow(postimg)
    pyplot.show()
    input('next')
    pyplot.cla()
