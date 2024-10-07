import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Check cuDNN version
print("cuDNN Version:", torch.backends.cudnn.version())