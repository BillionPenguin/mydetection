import torch
import torch.nn as nn

class fastRCNN(nn.module):
    def __init__(self, backbone, RPN, head, transformer=None):
        super(fastRCNN, self).__init__()
        self.transformer = transformer
        self.backbone = backbone
        self.rpn = RPN
        self.head = head

    def forward(self, images, targets=None):
        input_img_sizes = [img.shape[-2:] for img in images]
        losses = {}
        if self.traning and targets == None:
            raise ValueError("targets not found in traning mode")
        if self.transformer != None:
            x, targets = self.transformer(images, targets)
        features = self.backbone(x)
        rpn_box_cls, rpn_loss = self.RPN(features, targets)
        head_box_cls, head_loss = self.head(features, rpn_box_cls,  targets)

        losses['rpn'] = rpn_loss
        losses['head'] = head_loss

        if self.transformer != None:
            self.transformer.postprocess(head_box_cls, x.image_sizes, input_img_sizes)

        if self.traning:
            return losses
        return head_box_cls