import torch
import pdetection.box as box
import torchvision.models.detection._utils as util
import random
coder_ref = util.BoxCoder([2, 2, 2, 2])
device = torch.device('cuda')
coder = box.BoxCoder([2, 2, 2, 2])
out = [torch.randint(0, 100, (4, 4), dtype=torch.float32, device=device)]
inp = [torch.randint(0, 100, (4, 4), dtype=torch.float32, device=device)]
print(coder_ref.decode(out, inp).squeeze(), '\n', coder.decode(out, inp)[0])
print(coder_ref.decode(out, inp).squeeze().shape, '\n', coder.decode(out, inp)[0].shape)
print(torch.sum((coder_ref.decode(out, inp).squeeze() - coder.decode(out, inp)[0])))
print((coder_ref.decode(out, inp).squeeze() - coder.decode(out, inp)[0]).clamp(min=-999, max=999))