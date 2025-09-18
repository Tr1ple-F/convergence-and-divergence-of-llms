import sys
import pandas as pd
from utils import styled_plot

path = f'../working_dir/{sys.argv[1]}'
df = pd.read_csv(f'{path}/blimp.csv')
df['Training Step'] = df['Training Step'].str.replace('step', '', regex=False).astype(int)
df['Model'] = df['Model'].str.replace('EleutherAI/pythia-', '', regex=False).astype(str)
# Create a numeric version of the model size for sorting
df['Model_Size'] = df['Model'].str.replace('m', '', regex=False).astype(int)

# Sort by that size
df = df.sort_values(by='Model_Size')

# (Optional) drop helper column if you don't need it
df = df.drop(columns=['Model_Size']).reset_index(drop=True)

styled_plot(
    df,
    'Training Step',
    'KL',
    'Model',
    'Model',
    'Training Step',
    r'Downstream convergence',
    f'{path}/blimp.png',
    y_log = False,
    order_legend = False,
    y_scale=5,
    y_label_loc="top"
)