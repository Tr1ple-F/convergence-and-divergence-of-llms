import pandas as pd
import sys
from utils import styled_plot

pos_index = sys.argv[2]

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_pos_{pos_index}_dataframe.csv')

def plot_no_filters():
    df_plot = df.copy()
    df_plot = df_plot[
        (df_plot['Model'] == df_plot['Model 2']) &
        (df_plot['Training Step'] == df_plot['Training Step 2']) &
        (df_plot['Seed 1'] != df_plot['Seed 2'])
        ]

    y_label = r'Expected Convergence ($\mathbb{E}[\mathrm{conv}]$)'
    save_loc = f'../working_dir/{sys.argv[1]}/output/multiberts.png'

    styled_plot(
        df_plot, 'Training Step', 'KL Average', 'Model', 'Model', 'Training Step',
        y_label, save_loc, order_legend=False, legend_include=False, y_scale=5.5, y_label_loc="top", vertical_lines=[]
    )

plot_no_filters()