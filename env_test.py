import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(1)) # If available, shows the GPU name

device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu") # Use CUDA:2 if available
print(torch.cuda.get_device_name(device))