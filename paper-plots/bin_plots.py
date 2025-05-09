import pandas as pd
import sys
from utils import styled_plot

T_S = 'Training Step'

def bin_freq_10(x):
    if x < 10:
        return '0-10'
    elif x < 100:
        return '10-100'
    elif x < 1000:
        return '10^3-10^4'
    elif x < 10000:
        return '10^4-10^5'
    elif x < 100000:
        return '10^5-10^6'
    else:
        return '10^6+'

def bin_surprisal(x):
    if x < 1:
        return '0-1'
    elif x < 2:
        return '1-2'
    elif x < 3:
        return '2-3'
    elif x < 4:
        return '3-4'
    elif x < 5:
        return '4-5'
    elif x < 6:
        return '5-6'
    elif x < 7:
        return '6-7'
    elif x < 8:
        return '7-8'
    elif x < 9:
        return '8-9'
    elif x < 10:
        return '9-10'
    else:
        return '10+'


# Load the dataframe
def get_df(path, binner, column, title):
    df = pd.read_csv(path)
    if column == "Frequency":
        df = df.drop(columns=["POS","POS Context"])
    print("Loaded dataframe")
    df[title] = df[column].apply(binner)
    print("Binned")
    df_average = df.groupby(['Model', T_S, 'Seed 1', 'Model 2', 'Training Step 2', 'Seed 2', title]).mean().reset_index()
    print("Grouped")

    # Filter data
    df_plot = df_average[df_average['Model'].isin(["14m", "410m"])]
    df_plot = df_plot[df_plot['Model 2'] == df_plot['Model']]
    df_plot = df_plot[df_plot[T_S] == df_plot['Training Step 2']]
    df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
    return df_plot

# Plotting
y_label = r'Expected convergence ($\mathbb{E}[\mathrm{conv}]$)'
save_loc1 = f'../working_dir/{sys.argv[1]}/output/frequency_result.png'
save_loc2 = f'../working_dir/{sys.argv[1]}/output/surprisal_result.png'
path1 = f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv'
path2 = f'../working_dir/{sys.argv[1]}/output/seeds_surprisal_by_token.csv'
styled_plot(get_df(path1, bin_freq_10, 'Frequency', 'Binned Freq'), T_S, 'KL', 'Binned Freq', 'Model', T_S, y_label, save_loc1)
styled_plot(get_df(path2, bin_surprisal, 'Surprisal', 'Binned Cross Entropy'), T_S, 'KL', 'Binned Freq', 'Model', T_S, y_label, save_loc2)