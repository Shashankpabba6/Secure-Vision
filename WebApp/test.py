import torch
if torch.cuda.is_available():
    print("CUDA is available!")
    print("GPU device name:", torch.cuda.get_device_name(0))  # Assuming you have at least one GPU
else:
    print("CUDA is not available. You may want to install GPU drivers and CUDA toolkit.")