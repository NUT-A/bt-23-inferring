import torch
import oneflow

def print_stats():
    print(f"PyTorch version: {torch.__version__}")
    print(f"OneFlow version: {oneflow.__version__}")
    print(f"cudnn version: {torch.backends.cudnn.version()}")
    print(f"CUDA version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise SystemError("GPU device not found. PyTorch requires a GPU to run this code.")
    else:
        print(f"Found GPU device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")