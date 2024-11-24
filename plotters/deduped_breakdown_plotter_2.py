import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from datetime import datetime

def now():
    return datetime.now().strftime("%Y-%m-%d%H%M%S")

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe.csv')
df = df[df["Revision 1"].between(left=4000, right = 143000, inclusive="both")]
df = df[df["Revision 2"].between(left= 0, right = 512, inclusive="both")]

tag = 'CC'
sns.scatterplot(data=df, x=f"KL Average - {tag}", y=f"Surprisal Average - {tag}", hue="Revision 2")
plt.title("Scatter Plot of KL Average vs Surprisal Average for " + tag)
plt.savefig(f'../working_dir/{sys.argv[1]}/output/temp/out_{now()}.png')
plt.close()
