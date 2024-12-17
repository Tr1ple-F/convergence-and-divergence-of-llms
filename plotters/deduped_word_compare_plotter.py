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
cutoff = float(sys.argv[2])
c0 = cutoff
c1 = 1/cutoff
cutoff_surprisal = float(sys.argv[3])

revs = [int(strip(x)) for x in deduped_config()['revisions']]
models = [strip(x) for x in deduped_config()['model_names']]

interesting_tags = input("Enter tags separated by commas (default: Nouns, Verbs, IN, JJ, DT, CC, RB, NN, NNP, NNS, VB, VBD, VBG, VBN, VBP, VBZ, PRP, PRP$, TO, CD, MD): ").split(",")
if interesting_tags == [""]:
    interesting_tags = ["Nouns", "Verbs", "IN", "JJ", "DT", "CC", "RB", "NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "PRP", "PRP$", "TO", "CD", "MD"]
case_type = input("Enter case type (high, low, both): ").strip().lower() or "both"

show_high = case_type == "high" or case_type == "both"
show_low = case_type == "low" or case_type == "both"

print("Implicit assumption is that the surprisal gets better")

#############
# Dataframe #
#############
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe_neg1.csv')

# Models
frame1 = df[df['Model 1'] != "12b"]
frame1 = frame1[frame1['Model 1'] != "6.9b"]
frame1 = frame1[frame1['Model 1'] != "2.8b"]
frame1 = frame1[frame1.apply(lambda row: row['Model 2'] == models[models.index(row['Model 1']) + 1], axis=1)]
frame1 = frame1[frame1['Revision 1'] == frame1['Revision 2']]
# Revisions
frame2 = df[df['Model 1'] == df['Model 2']]
frame2 = frame2[frame2['Model 1'] != "12b"]
frame2 = frame2[frame2['Model 1'] != "6.9b"]

frame2 = frame2[frame2['Revision 1'] != 143000]
frame2 = frame2[frame2['Revision 1'] != 128000] # Comment out this
frame2 = frame2[frame2['Revision 1'] != 64000] # Comment out this
frame2 = frame2[frame2.apply(lambda row: row['Revision 2'] == revs[revs.index(row['Revision 1']) + 1], axis=1)]

#############
# Functions #
#############
def check_surprisal(row, tag):
    m1 = row['Model 1']
    m2 = row['Model 2']
    r1 = row['Revision 1']
    r2 = row['Revision 2']
    s1 = df[(df['Model 1'] == m1) & (df['Revision 1'] == r1)]['Surprisal Average - ' + tag].values[0]
    s2 = df[(df['Model 1'] == m2) & (df['Revision 1'] == r2)]['Surprisal Average - ' + tag].values[0]
    if s1/s2 < cutoff_surprisal:
        print(f"ALARM! % {s1/s2*100}")

def analyze(row):
    for tag in interesting_tags:
        if row['KL Average'] > 0.001:
            if show_high and row[f'KL Average - {tag}'] > c1*row[f'KL Average']:
                print(f"! Revision: {row['Revision 1']}, Model: {row['Model 1']} -> {row['Model 2']}, Tag: {tag}, %: {row[f'KL Average - {tag}']/row[f'KL Average'] * 100}")
                check_surprisal(row, tag)
            if show_low and row[f'KL Average - {tag}'] < c0*row[f'KL Average']:
                print(f"# Revision: {row['Revision 1']}, Model: {row['Model 1']} -> {row['Model 2']}, Tag: {tag}, %: {row[f'KL Average - {tag}']/row[f'KL Average'] * 100}")
                check_surprisal(row, tag)

###############
# Main script #
###############

def print_out():
    print("------------ Models    ------------")
    for i, row in frame1.iterrows():
        analyze(row)
    print("------------ Revisions ------------")
    for i, row in frame2.iterrows():
        analyze(row)

def plot1(tag_name):
    # melted = frame2.melt(id_vars=['Model 1', 'Revision 1', 'Model 2', 'Revision 2'], var_name='Metric', value_vars=[f'KL Average', f'KL Average - {tag_name}'], value_name='KL')
    # Add column ratio which divides KL Average - tag_name by KL Average
    melted = frame2.copy()
    melted[f'σ'] = melted[f'KL Average - {tag_name}'] / melted[f'KL Average']
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted, x='Revision 1', y='σ', errorbar="sd")
    # Add a horizontal line at 1
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xscale('log')
    # Fix y-axis bottom at 0
    plt.ylim(bottom=0)
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/word_compare_{tag_name}.png')
    plt.close()

def linear_func(x, alpha, offset):
    return alpha * x + offset

alpha_list = []
end_alpha_fit = []

def plot2(tag_name):
    # melted = frame2.melt(id_vars=['Model 1', 'Revision 1', 'Model 2', 'Revision 2'], var_name='Metric', value_vars=[f'KL Average', f'KL Average - {tag_name}'], value_name='KL')
    # Add column ratio which divides KL Average - tag_name by KL Average
    melted = frame2.copy()
    melted[f'Ratio'] = melted.apply(lambda row: row[f'Surprisal Average - {tag_name}'] / df[(df['Model 1'] == row['Model 1']) & (df['Revision 1'] == revs[revs.index(row['Revision 1']) + 1])][f'Surprisal Average - {tag_name}'].values[0], axis=1)
    melted[f'Cross-Entropy Δ'] = melted.apply(lambda row: row[f'Surprisal Average - {tag_name}'] - df[(df['Model 1'] == row['Model 1']) & (df['Revision 1'] == revs[revs.index(row['Revision 1']) + 1])][f'Surprisal Average - {tag_name}'].values[0], axis=1)

    choice = f'Cross-Entropy Δ'
    xdata = melted[choice]
    ydata = melted[f'KL Average - {tag_name}']
    popt, pcov = curve_fit(linear_func, xdata, ydata) # On whole data
    perr = np.sqrt(np.diag(pcov)) # Calculate the standard deviation errors
    alpha, offset = popt

    melted[f'Fitted KL Average - {tag_name}'] = linear_func(melted[choice], *popt)

    print(f"Fitted for {tag_name}: alpha = {alpha}, offset = {offset}, error = {perr}")
    alpha_list.append({'alpha': alpha, 'tag': tag_name})

    plt.figure(figsize=(10, 6))
    melted = melted.melt(id_vars=['Model 1', 'Revision 1', 'Model 2', 'Revision 2'], value_vars=[f'KL Average - {tag_name}', choice, f'Fitted KL Average - {tag_name}'], var_name='Metric', value_name='Value')
    sns.lineplot(data=melted, x='Revision 1', y='Value', hue='Metric', errorbar="sd").set(ylabel='Value')
    # Please add an annotation on the plot adding alpha and offset for alpha*ratio + offset
    plt.annotate(f'α = {alpha:.2f}, offset = {offset:.2f}', xy=(0.1, 0.9), xycoords='axes fraction')
    plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/word_compare_fitted_{tag_name}.png')
    plt.close()


def plot_general():
    melted = frame2.copy()
    melted[f'Cross-Entropy Δ'] = melted.apply(lambda row: row[f'Surprisal Average'] - df[(df['Model 1'] == row['Model 1']) & (df['Revision 1'] == revs[revs.index(row['Revision 1']) + 1])][f'Surprisal Average'].values[0], axis=1)

    choice = f'Cross-Entropy Δ'
    xdata = melted[choice]
    ydata = melted[f'KL Average']
    popt, pcov = curve_fit(linear_func, xdata, ydata) # On whole data
    alpha, offset = popt

    melted[f'Fitted KL Average'] = linear_func(melted[choice], *popt)

    print(f"Fitted for: alpha = {alpha}, offset = {offset}, error = {np.sqrt(np.diag(pcov))}")

    plt.figure(figsize=(10, 6))
    melted = melted.melt(id_vars=['Model 1', 'Revision 1', 'Model 2', 'Revision 2'], value_vars=[f'KL Average', choice, f'Fitted KL Average'], var_name='Metric', value_name='Value')
    sns.lineplot(data=melted, x='Revision 1', y='Value', hue='Metric', errorbar="sd").set(ylabel='Value')
    # Please add an annotation on the plot adding alpha and offset for alpha*ratio + offset
    plt.annotate(f'α = {alpha:.2f}, offset = {offset:.2f}', xy=(0.1, 0.9), xycoords='axes fraction')
    plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/word_compare_fitted_{choice}_all.png')
    plt.close()

def plot(tag_name):
    plot1(tag_name)
    plot2(tag_name)


plot_general()

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
alpha_list.sort(key=lambda x: x['alpha'])
for alpha in alpha_list:
    print(f"{alpha['tag']} & {alpha['alpha'] * 100:.2f}\\% \\ \\hline")
