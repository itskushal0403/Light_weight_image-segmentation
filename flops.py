import torch
from model import LiteSeg
from thop import profile

model = LiteSeg(num_classes=21)

dummy_input = torch.randn(1, 3, 300, 300)

flops, params = profile(model, inputs=(dummy_input, ))

print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
print(f"Params: {params / 1e6:.4f} M")