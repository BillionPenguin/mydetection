from torchvision.datasets.coco import CocoDetection
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import transforms
from pdetection import transform
from torch.utils.data.dataloader import DataLoader
import torch

from pdetection import utils

path_data = '/home/peng/Documents/srch/Object Detection/dataset/coco/val2017'
path_anno = '/home/peng/Documents/srch/Object Detection/dataset/coco/annotations_trainval2017/annotations/instances_val2017.json'

coco_dset = CocoDetection(root=path_data, annFile=path_anno)
trans = transform.ODTransformer(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
trans_compare = GeneralizedRCNNTransform(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# trans = transform.ODTransformer(800, 1333, [0.4, 0.5, 0.6], [0.5, 1, 2])

cocoloader = DataLoader(coco_dset, batch_size=3, collate_fn=utils.collate_fn)
for data in cocoloader:
    images, targets = data
    inputimgs = []
    totensor = transforms.ToTensor()
    for img in images:
        inputimgs.append(totensor(img))
    src_img_sizes = [img.shape[-2:] for img in inputimgs]
    # print([img.shape[-2:] for img in inputimgs])
    targets1 = utils.totargets(targets)
    targets2 = utils.totargets(targets)
    print(targets2[2]['boxes'])
    inputimgs_bk = [torch.zeros_like(img).copy_(img) for img in inputimgs]
    # print([img.shape for img in inputimgs])
    outimg, outtrg = trans(inputimgs, targets1)
    # print([img.shape for img in inputimgs_bk])
    outimg2, outtrg2 = trans_compare(inputimgs_bk, targets2)
    # print(outimg.image_sizes)
    # print(outimg.tensors.shape)
    # print(outimg2.image_sizes)
    # print(outimg2.tensors.shape)
    #
    # print(torch.sum(outimg2.tensors - outimg.tensors) / outimg.tensors.numel())
    break