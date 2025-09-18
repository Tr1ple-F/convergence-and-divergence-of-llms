import ipdb
import pandas as pd
import sys
from utils import styled_plot

df_plot = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv')
df_plot = df_plot.drop(columns=["Frequency", "POS", "POS Context"])
df_plot = df_plot[
    (df_plot['Model'] == df_plot['Model 2']) &
    (df_plot['Training Step'] == df_plot['Training Step 2']) &
    (df_plot['Seed 1'] != df_plot['Seed 2'])
    ]
df_plot = df_plot.groupby(['Model', 'Training Step', 'ID'], as_index=False)['KL'].mean()
model_order = ['14m', '31m', '70m', '160m', '410m']
df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=model_order, ordered=True)
df_plot = df_plot.sort_values(by=['Model', 'Training Step'], ascending=[True, True])

y_label = r'Expected Convergence ($\mathbb{E}[\mathrm{conv}]$)'
save_loc = f'../working_dir/{sys.argv[1]}/output/seeds_context_variance.png'
styled_plot(
    df_plot,
    'Training Step',
    'KL',
    'Model',
    'Model',
    'Training Step',
    y_label,
    save_loc,
    order_legend=False,
    y_scale=5,
    y_label_loc="top"
)
