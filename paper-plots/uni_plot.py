import pandas as pd
import sys
from utils import styled_plot

S1 = 'Training Step'

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_uni_dataframe.csv')
df_uniform = df[[S1, 'Model', 'Uniform KL']].rename(columns={'Uniform KL': 'KL'})
df_uniform['Distribution'] = 'Uniform'
df_unigram = df[[S1, 'Model', 'Unigram KL']].rename(columns={'Unigram KL': 'KL'})
df_unigram['Distribution'] = 'Unigram'
df_out = pd.concat([df_uniform, df_unigram])

path = f'../working_dir/{sys.argv[1]}/output/seeds_against_dist.png'
y_label = r'Expected KL'
styled_plot(df_out, S1, 'KL', 'Model', 'Distribution', S1, y_label, path, order_legend=False, y_scale=3, legend_include=(7,9))