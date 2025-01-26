import torch

# TBD - supply a number of iterations beyond which train/eval loops are preempted (for debugging purposes only)
MAX_SAMPLES = None

def device():
    # return torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")