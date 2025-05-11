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
df = df[['Unnamed: 0', 'Revision', 'Frequency'] + models_short]

df1 = df.copy()
df2 = df.copy()
df_l_1 = pd.melt(
    df1,
    id_vars=['Unnamed: 0', 'Revision', 'Frequency'],
    var_name='Model',
    value_name='Value'
)
df_l_2 = pd.melt(
    df2,
    id_vars=['Unnamed: 0', 'Revision'] + models_short,
    var_name='Factor',
    value_name='Value'
)

save_loc1 = f'../working_dir/{sys.argv[1]}/output/seeds_linear_regression_by_Model.png'
styled_plot(df_l_1, 'Revision', 'Value', 'Model', 'Model', 'Training Step', 'δ', save_loc1, order_legend=False)
save_loc2 = f'../working_dir/{sys.argv[1]}/output/seeds_linear_regression_by_Freq.png'
styled_plot(df_l_2, 'Revision', 'Value', 'Factor', 'Factor', 'Training Step', 'α', save_loc2, order_legend=False)