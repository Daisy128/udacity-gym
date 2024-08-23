import torch

if __name__ == '__main__':

    print("CUDA Available:", torch.cuda.is_available())
    print("PyTorch version:", torch.__version__)

    # Get the current CUDA device
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("No CUDA device found.")
