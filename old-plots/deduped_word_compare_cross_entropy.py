import sys

import pandas as pd

from utils import deduped_config, strip

##########
# Inputs #
##########
cutoff = float(sys.argv[2])
c0 = cutoff
c1 = 1/cutoff

revs = [int(strip(x)) for x in deduped_config()['revisions']]
models = [strip(x) for x in deduped_config()['model_names']]

interesting_tags = input("Enter tags separated by commas (default: Nouns, Verbs, IN, JJ, DT, CC, RB, NN, NNP, NNS, VB, VBD, VBG, VBN, VBP, VBZ, PRP, PRP$, TO, CD, MD): ").split(",")
if interesting_tags == [""]:
    interesting_tags = ["Nouns", "Verbs", "IN", "JJ", "DT", "CC", "RB", "NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "PRP", "PRP$", "TO", "CD", "MD"]
case_type = input("Enter case type (high, low, both): ").strip().lower() or "both"

show_high = case_type == "high" or case_type == "both"
show_low = case_type == "low" or case_type == "both"

check_models = sys.argv[3] == "models" or sys.argv[3] == "both"
check_revisions = sys.argv[3] == "revisions" or sys.argv[3] == "both"

print("Implicit assumption is that the surprisal gets better")

#############
# Functions #
#############
def calc_surprisal_row(row):
    m1 = row['Model 1']
    m2 = row['Model 2']
    r1 = row['Revision 1']
    r2 = row['Revision 2']
    s1 = df[(df['Model 1'] == m1) & (df['Revision 1'] == r1)]['Surprisal Average'].values[0]
    s2 = df[(df['Model 1'] == m2) & (df['Revision 1'] == r2)]['Surprisal Average'].values[0]
    row['Surprisal Diff'] = s1 - s2

def calc_surprisal_diff(row, tag):
    m1 = row['Model 1']
    m2 = row['Model 2']
    r1 = row['Revision 1']
    r2 = row['Revision 2']
    s1 = df[(df['Model 1'] == m1) & (df['Revision 1'] == r1)]['Surprisal Average - ' + tag].values[0]
    s2 = df[(df['Model 1'] == m2) & (df['Revision 1'] == r2)]['Surprisal Average - ' + tag].values[0]
    row['Surprisal Diff - ' + tag] = s1 - s2

def analyze(row):
    for tag in interesting_tags:
        if row['Surprisal Diff'] > 0.001:
            if show_high and row[f'Surprisal Diff - {tag}'] > c1*row[f'Surprisal Diff']:
                print(f"! Revision: {row['Revision 1']}, Model: {row['Model 1']} -> {row['Model 2']}, Tag: {tag}, %: {row[f'Surprisal Diff - {tag}']/row[f'Surprisal Diff'] * 100}")
            if show_low and row[f'Surprisal Diff - {tag}'] < c0*row[f'Surprisal Diff']:
                print(f"# Revision: {row['Revision 1']}, Model: {row['Model 1']} -> {row['Model 2']}, Tag: {tag}, %: {row[f'Surprisal Diff - {tag}']/row[f'Surprisal Diff'] * 100}")

#############
# Dataframe #
#############
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe.csv')

###############
# Main script #
###############

# Models
if check_models:
    frame1 = df[df['Model 1'] != "12b"]
    frame1 = frame1[frame1['Model 1'] != "6.9b"]
    frame1 = frame1[frame1['Model 1'] != "2.8b"]
    frame1 = frame1[frame1.apply(lambda row: row['Model 2'] == models[models.index(row['Model 1']) + 1], axis=1)]
    frame1 = frame1[frame1['Revision 1'] == frame1['Revision 2']]
    frame1 = frame1[['Model 1', 'Model 2', 'Revision 1', 'Revision 2']]
    frame1_rows = []
    for i, row in frame1.iterrows():
        row = row.copy()  # Create a copy of the row to allow modifications
        calc_surprisal_row(row)
        for tag in interesting_tags:
            calc_surprisal_diff(row, tag)
        frame1_rows.append(row)

    frame1_new = pd.DataFrame(frame1_rows)
    print(frame1_new)
    print("------------ Models    ------------")
    for i, row in frame1_new.iterrows():
        analyze(row)

# Revisions
if check_revisions:
    frame2 = df[df['Model 1'] == df['Model 2']]
    frame2 = frame2[frame2['Model 1'] == '2.8b'] # COMMENT OUT LATER!!!
    print("Reduced models")
    frame2 = frame2[frame2['Revision 1'] != 143000]
    frame2 = frame2[frame2.apply(lambda row: row['Revision 2'] == revs[revs.index(row['Revision 1']) + 1], axis=1)]
    frame2 = frame2[['Model 1', 'Model 2', 'Revision 1', 'Revision 2']]
    frame2_rows = []
    for i, row in frame2.iterrows():
        row = row.copy()  # Create a copy of the row to allow modifications
        calc_surprisal_row(row)
        for tag in interesting_tags:
            calc_surprisal_diff(row, tag)
        frame2_rows.append(row)

    frame2_new = pd.DataFrame(frame2_rows)
    print(frame2_new)
    print("------------ Revisions ------------")
    for i, row in frame2_new.iterrows():
        analyze(row)
