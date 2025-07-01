import sys
import pandas as pd
from utils import styled_plot

path = f'../working_dir/{sys.argv[1]}'
df = pd.read_csv(f'{path}/seeds_icl.csv')
df['Training Step'] = df['Training Step'].str.replace('step', '', regex=False).astype(int)
df['Model'] = df['Model'].str.replace('EleutherAI/pythia-', '', regex=False).astype(str)

styled_plot(
    df,
    'Training Step',
    'ICL',
    'Model',
    'Model',
    'Training Step',
    'ICL',
    f'{path}/icl.png',
    y_log = False,
    order_legend = False
)