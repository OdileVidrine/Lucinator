import torch

print("cuda available", torch.cuda.is_available())
print("hip version", torch.version.hip)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
x = torch.randn(1, device="cuda")
print("ok", x.device)
