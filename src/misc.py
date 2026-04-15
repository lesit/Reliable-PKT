import random
import numpy as np
import time

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_elapsed_time(elapsed_time:float):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours>0:
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    elif minutes>0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.1f} sec"

def format_elapse_from(start_time:float):
    return format_elapsed_time(time.time() - start_time)
