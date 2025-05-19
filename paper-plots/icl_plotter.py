import sys
import pandas as pd
from utils import styled_plot

path = f'../working_dir/{sys.argv[1]}/output'
df = pd.read_csv(f'{path}/seeds_icl.csv')

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