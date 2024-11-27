import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from datetime import datetime

def now():
    return datetime.now().strftime("%Y-%m-%d%H%M%S")

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe.csv')

revision1_start = int(input("Enter the start value for Revision 1 (default 0): ") or 0)
revision1_end = int(input("Enter the end value for Revision 1 (default 143k): ") or 143000)
revision2_start = int(input("Enter the start value for Revision 2 (default 0): ") or 0)
revision2_end = int(input("Enter the end value for Revision 2 (default 143k): ") or 143000)

df = df[df["Revision 1"].between(left=revision1_start, right=revision1_end, inclusive="both")]
df = df[df["Revision 2"].between(left=revision2_start, right=revision2_end, inclusive="both")]
df = df[df["Model 1"] == '12b']
df = df[df["Model 2"] == '6.9b']

note = f" (Revisions {revision1_start}-{revision1_end} and {revision2_start}-{revision2_end})"
note_short = f"_rev1_{revision1_start}_{revision1_end}_rev2_{revision2_start}_{revision2_end}"

tag = input("Enter the tag (default $): ") or '$'
now = now()

sns.scatterplot(data=df, x=f"KL Average - {tag}", y=f"Surprisal Average - {tag}", hue="Model 1")
plt.title("KL vs Surprisal for " + tag + note)
plt.savefig(f'../working_dir/{sys.argv[1]}/output/temp/out_{now}{note_short}_m1.png')
plt.close()

sns.scatterplot(data=df, x=f"KL Average - {tag}", y=f"Surprisal Average - {tag}", hue="Model 2")
plt.title("KL vs Surprisal for " + tag + note)
plt.savefig(f'../working_dir/{sys.argv[1]}/output/temp/out_{now}{note_short}_m2.png')
plt.close()

sns.scatterplot(data=df, x=f"KL Average - {tag}", y=f"Surprisal Average - {tag}", hue="Revision 1")
plt.title("KL vs Surprisal for " + tag + note)
plt.savefig(f'../working_dir/{sys.argv[1]}/output/temp/out_{now}{note_short}_r1.png')
plt.close()

sns.scatterplot(data=df, x=f"KL Average - {tag}", y=f"Surprisal Average - {tag}", hue="Revision 2")
plt.title("KL vs Surprisal for " + tag + note)
plt.savefig(f'../working_dir/{sys.argv[1]}/output/temp/out_{now}{note_short}_r2.png')
plt.close()
