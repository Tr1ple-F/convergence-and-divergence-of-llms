import sys

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from utils import deduped_config, strip
from scipy.optimize import curve_fit

##########
# Inputs #
##########
revs = [int(strip(x)) for x in deduped_config()['revisions']]
models = [strip(x) for x in deduped_config()['model_names']]
type = sys.argv[2]

#############
# Dataframe #
#############
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/deduped_pos_{type}_dataframe.csv')

# Revisions
dataframe = df[df['Model 1'] == df['Model 2']]
dataframe = dataframe[dataframe['Model 1'] != "12b"]
dataframe = dataframe[dataframe['Model 1'] != "6.9b"]

dataframe = dataframe[dataframe['Revision 1'] != 143000]
dataframe = dataframe[dataframe['Revision 1'] != 128000] # 160m, 70m exception
dataframe = dataframe[dataframe['Revision 1'] != 64000]
dataframe = dataframe[dataframe.apply(lambda row: row['Revision 2'] == revs[revs.index(row['Revision 1']) + 1], axis=1)]

def plot1(tag_name):
    cp = dataframe.copy()
    cp[f'σ'] = cp[f'KL Average - {tag_name}'] / cp[f'KL Average']
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=cp, x='Revision 1', y='σ', errorbar="ci")
    # Add a horizontal line at 1
    plt.axhline(y=1, color='r', linestyle='--')
    plt.axvline(x=8, color='g', linestyle='--')
    plt.axvline(x=64, color='g', linestyle='--')
    plt.axvline(x=1000, color='g', linestyle='--')
    plt.xscale('log')
    # Fix y-axis bottom at 0
    plt.ylim(bottom=0)
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/ratio_{tag_name}_{type}.png')
    plt.close()

def linear_func(x, alpha, offset):
    return alpha * x + offset

ratio_list = []
diff_list = []

def plot2(tag_name):
    str0 = f'Cross-Entropy Ratio'
    str1 = f'Cross-Entropy Δ'

    melted = dataframe.copy()
    melted[str0] = melted.apply(lambda row: row[f'Surprisal Average - {tag_name}'] / df[(df['Model 1'] == row['Model 1']) & (df['Revision 1'] == revs[revs.index(row['Revision 1']) + 1])][f'Surprisal Average - {tag_name}'].values[0], axis=1)
    melted[str1] = melted.apply(lambda row: row[f'Surprisal Average - {tag_name}'] - df[(df['Model 1'] == row['Model 1']) & (df['Revision 1'] == revs[revs.index(row['Revision 1']) + 1])][f'Surprisal Average - {tag_name}'].values[0], axis=1)

    xdata0 = melted[str0]
    xdata1 = melted[str1]
    ydata = melted[f'KL Average - {tag_name}']
    popt0, pcov0 = curve_fit(linear_func, xdata0, ydata)
    alpha0, offset0 = popt0
    popt1, pcov1 = curve_fit(linear_func, xdata1, ydata)
    alpha1, offset1 = popt1

    melted[f'Ratio Fitted KL Average - {tag_name}'] = linear_func(melted[str0], *popt0)
    melted[f'Diff Fitted KL Average - {tag_name}'] = linear_func(melted[str1], *popt1)

    ratio_list.append({'alpha': alpha0, 'tag': tag_name})
    diff_list.append({'alpha': alpha1, 'tag': tag_name})

    plt.figure(figsize=(10, 6))
    df0 = melted.melt(id_vars=['Model 1', 'Revision 1', 'Model 2', 'Revision 2'], value_vars=[f'KL Average - {tag_name}', str0, f'Ratio Fitted KL Average - {tag_name}'], var_name='Metric', value_name='Value')
    sns.lineplot(data=df0, x='Revision 1', y='Value', hue='Metric', errorbar="ci", legend=None).set(ylabel='Value')
    plt.xscale('log')
    plt.axvline(x=8, color='r', linestyle='--')
    plt.axvline(x=64, color='r', linestyle='--')
    plt.axvline(x=1000, color='r', linestyle='--')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/curve_fitted_{tag_name}_ratio_{type}.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    df1 = melted.melt(id_vars=['Model 1', 'Revision 1', 'Model 2', 'Revision 2'], value_vars=[f'KL Average - {tag_name}', str1, f'Diff Fitted KL Average - {tag_name}'], var_name='Metric', value_name='Value')
    sns.lineplot(data=df1, x='Revision 1', y='Value', hue='Metric', errorbar="ci", legend=None).set(ylabel='Value')
    plt.xscale('log')
    plt.axvline(x=8, color='r', linestyle='--')
    plt.axvline(x=64, color='r', linestyle='--')
    plt.axvline(x=1000, color='r', linestyle='--')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/curve_fitted_{tag_name}_diff_{type}.png')
    plt.close()

def plot(tag_name):
    plot1(tag_name)
    plot2(tag_name)

def general():
    str0 = f'Cross-Entropy Ratio'
    str1 = f'Cross-Entropy Δ'
    melted = dataframe.copy()
    melted[str0] = melted.apply(lambda row: row[f'Surprisal Average'] / df[(df['Model 1'] == row['Model 1']) & (df['Revision 1'] == revs[revs.index(row['Revision 1']) + 1])][f'Surprisal Average'].values[0], axis=1)
    melted[str1] = melted.apply(lambda row: row[f'Surprisal Average'] - df[(df['Model 1'] == row['Model 1']) & (df['Revision 1'] == revs[revs.index(row['Revision 1']) + 1])][f'Surprisal Average'].values[0], axis=1)
    xdata0 = melted[str0]
    xdata1 = melted[str1]
    ydata = melted[f'KL Average']
    popt0, pcov0 = curve_fit(linear_func, xdata0, ydata)
    alpha0, offset0 = popt0
    popt1, pcov1 = curve_fit(linear_func, xdata1, ydata)
    alpha1, offset1 = popt1

    print(f"Alpha 1: {alpha0}")
    print(f"Alpha 2: {alpha1}")

general()
plot("TO")
plot("MD")
plot("VBN")
plot("VBZ")
plot("JJ")
plot("NNP")
plot("PRP$")
plot("NN")
plot("NNS")
plot("DT")
plot("VB")
plot("VBG")
plot("VBD")
plot("CC")
plot("CD")
plot("PRP")
plot("IN")
plot("RB")

# Sort alpha list by alpha
print("---- Ratios ----")
ratio_list.sort(key=lambda x: x['alpha'])
for alpha in ratio_list:
    print(f"{alpha['tag']} & {alpha['alpha'] * 100:.2f}\\% \\\\ \\hline")
print("---- Diffs ----")
diff_list.sort(key=lambda x: x['alpha'])
for alpha in diff_list:
    print(f"{alpha['tag']} & {alpha['alpha'] * 100:.2f}\\% \\\\ \\hline")
