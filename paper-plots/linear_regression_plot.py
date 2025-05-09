import pandas as pd
import sys
from utils import styled_plot

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_pos_linear_regression_transpose.csv')

models = ['Model 1_14m', 'Model 1_31m', 'Model 1_70m', 'Model 1_160m', 'Model 1_410m']
models_short = [x.replace('Model 1_', '') for x in models]
rename_dict = {models[i]: models_short[i] for i in range(len(models))}
df.rename(columns=rename_dict, inplace=True)
cols_to_drop = [col for col in df.columns if col.startswith('POS')]
df = df.drop(columns=cols_to_drop)

df1 = df.copy()
df2 = df.copy()
df_long = pd.melt(
    df2,
    id_vars=['Unnamed: 0', 'Revision', 'Frequency'],
    var_name='Model',
    value_name='Value'
)

freq = 'Frequency'
save_loc1 = f'../working_dir/{sys.argv[1]}/output/seeds_linear_regression_by_Model.png'
styled_plot(df_long, 'Revision', 'Value', 'Model', 'Model', 'Training Step', 'δ', save_loc1, order_legend=False)
save_loc2 = f'../working_dir/{sys.argv[1]}/output/seeds_linear_regression_by_Freq.png'
styled_plot(df_long, 'Revision', freq, None, None, 'Training Step', 'α', save_loc2, order_legend=False)