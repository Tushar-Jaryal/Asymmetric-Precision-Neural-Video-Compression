import torch
from compressai.zoo import cheng2020_attn

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = cheng2020_attn(quality=4, pretrained=True).to(device)
model.eval()

x=torch.rand(1, 3, 256, 256).to(device)
with torch.no_grad():
    out = model(x)

bits = 0
for l in out["likelihoods"].values():
    bits += torch.sum(-torch.log2(l)).item()

print("Bits:", out["likelihoods"])
print("Recon shape:", out["x_hat"].shape)