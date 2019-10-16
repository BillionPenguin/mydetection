import torch
a = torch.tensor([[1, -2, -1, 8],
                  [1, 8, -1, 4],
                  [8, -2, -1, 4]])
v, i= torch.max(a, dim=0)
a[a < 0] = torch.arange(a[a < 0].numel())

