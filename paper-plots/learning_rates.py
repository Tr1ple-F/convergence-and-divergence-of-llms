import math
import sys
import pandas as pd
from utils import styled_plot

S_1 = "Model"
S_2 = "Training Step"
S_3 = "Learning Rate"

def get_lr(num_iters, start_lr, min_lr, warmup_iter = (0.01*143000), end_iter = 143000): # Code from gpt-neo-x
    """Learning rate decay functions from:
    https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

    num_iters_ = num_iters
    # Warmup.
    if warmup_iter > 0 and num_iters <= warmup_iter:
        return float(start_lr) * num_iters_ / warmup_iter

    end_iter_ = end_iter - warmup_iter
    lr = min_lr + (
            (start_lr - min_lr)
            / 2.0
            * (math.cos(math.pi * num_iters_ / end_iter_) + 1)
    )
    return max(lr, min_lr)

model_configs = [("14m", 0.001, 0.0001), ("31m", 0.001, 0.0001), ("70m", 0.001, 0.0001), ("160m", 0.0006, 0.00006), ("410m", 0.0003, 0.00003)] # From the pythia files

arr = []
path = f'../working_dir/{sys.argv[1]}/output'

for model_config in model_configs:
    for i in ([0,1,2,4,8,16,32,64,128,256,512,1000] + [1000*v for v in [2,4,8,16,32,64,128,143]]):
        m, start, min_lr = model_config
        arr.append({
            S_1: m,
            S_2: i,
            S_3: get_lr(i, start, min_lr)
        })

df = pd.DataFrame(arr)
df.to_csv(f'{path}/learning_rates.csv')

styled_plot(df, S_2, S_3, "Model", "Model", S_2, S_3, f'{path}/learning_rates.png')