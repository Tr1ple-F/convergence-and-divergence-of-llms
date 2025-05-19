import ipdb
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
    elif x < 4:
        return '2-4'
    elif x < 8:
        return '4-8'
    else:
        return '8+'


# Load the dataframe
def get_df(path, binner, column, title, is_kl = True):
    df = pd.read_csv(path)

    df = df.drop(columns=["POS","POS Context", "ID"])

    if is_kl:
        df = df.drop(columns=['Surprisal'])
    else:
        df = df.drop(columns = ['KL'])

    if column == 'Final Surprisal':
        df = df.drop(columns=['Frequency'])
    else:
        df = df.drop(columns=['Final Surprisal'])

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
y_label1 = r'Expected convergence ($\mathbb{E}[\mathrm{conv}]$)'
y_label2 = r'Cross Entropy ($H$)'
h_label = r'Binned $H$'
save_loc1 = f'../working_dir/{sys.argv[1]}/output/frequency_result.png'
save_loc2 = f'../working_dir/{sys.argv[1]}/output/surprisal_result.png'
save_loc3 = f'../working_dir/{sys.argv[1]}/output/frequency_result_ce.png'
save_loc4 = f'../working_dir/{sys.argv[1]}/output/surprisal_result_ce.png'
path1 = f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv'

freq_df_kl = get_df(path1, bin_freq_10, 'Frequency', 'Binned Freq', is_kl=True)
styled_plot(freq_df_kl, T_S, 'KL', 'Binned Freq', 'Model', T_S, y_label1, save_loc1, order_legend=False)

cross_entropy_df_kl = get_df(path1, bin_surprisal, 'Final Surprisal', h_label, is_kl=True)
styled_plot(cross_entropy_df_kl, T_S, 'KL', h_label, 'Model', T_S, y_label1, save_loc2, order_legend=False)

freq_df_ce = get_df(path1, bin_freq_10, 'Frequency', 'Binned Freq', is_kl=False)
styled_plot(freq_df_ce, T_S, 'Surprisal', 'Binned Freq', 'Model', T_S, y_label2, save_loc3, order_legend=False)

cross_entropy_df_ce = get_df(path1, bin_surprisal, 'Final Surprisal', h_label, is_kl=False)
styled_plot(cross_entropy_df_ce, T_S, 'Surprisal', h_label, 'Model', T_S, y_label2, save_loc4, order_legend=False)