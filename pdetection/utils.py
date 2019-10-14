from PIL import Image, ImageFont, ImageDraw
import torch
import random

def whto2pts(x, y, w, h):
    return [x, y,  x + w, y + h]

def tosingletarget(src):
    nlen = len(src)
    boxes = torch.zeros(nlen, 4)
    labels = torch.zeros(nlen)
    for i in range(nlen):
        boxes[i] = torch.tensor(whto2pts(*src[i]['bbox']))
        labels[i] = torch.tensor(src[i]['category_id'])
    return boxes, labels

def totargets(srcs):
    ret = []
    for src in srcs:
        this = {}
        this['boxes'], this['labels'] = tosingletarget(src)
        ret.append(this)
    return ret

def collate_fn(batch):
    return tuple(zip(*batch))

def idtocname(idxs, coco):
    if isinstance(idxs, torch.Tensor):
        if idxs.dim() == 1:
            idxs = [int(t) for t in idxs]
        else:
            raise ValueError('idxs\'s dim should be 1 if it\'s a Tensor')
    biblo = coco.loadCats(ids=idxs)
    cnames = [t['name'] for t in biblo]
    return cnames

def drawsingleRect(img, box, label, score):
    x, y, x1, y1 = box
    string = '%s\n%.2f' % (label, score)
    draw = ImageDraw.Draw(img)
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    draw.rectangle(((x, y), (x1, y1)), outline=(r, g, b))
    font_ = ImageFont.truetype('arial', size=10)
    text_size = font_.getsize(string)
    draw.text((x, y), string, font=font_, fill=(r, g, b))

def drawRects(img, boxes, labels, scores):
    for i in range(len(boxes)):
        drawsingleRect(img, boxes[i], labels[i], scores[i])
    return