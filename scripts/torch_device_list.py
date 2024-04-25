import torch

available = []
for name in torch.backends.__dict__.keys():
    try:
        if getattr(torch.backends, name).is_available():
            available.append(name)
    except:
        continue

print(f'› Available PyTorch backends: {", ".join(available)}')