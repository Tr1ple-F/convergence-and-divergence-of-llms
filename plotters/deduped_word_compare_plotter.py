import sys

import pandas as pd

from utils import deduped_config, strip

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
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe.csv')

# Models
frame1 = df[df['Model 1'] != "12b"]
frame1 = frame1[frame1['Model 1'] != "6.9b"]
frame1 = frame1[frame1['Model 1'] != "2.8b"]
frame1 = frame1[frame1.apply(lambda row: row['Model 2'] == models[models.index(row['Model 1']) + 1], axis=1)]
frame1 = frame1[frame1['Revision 1'] == frame1['Revision 2']]
# Revisions
frame2 = df[df['Model 1'] == df['Model 2']]
frame2 = frame2[frame2['Revision 1'] != 143000]
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

print("------------ Models    ------------")
for i, row in frame1.iterrows():
    analyze(row)
print("------------ Revisions ------------")
for i, row in frame2.iterrows():
    analyze(row)
