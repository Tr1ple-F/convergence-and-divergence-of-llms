import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import math

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv')
df = df[df['Seed 1'] != df['Seed 2']]
df = df[df['Model 1'] == df['Model 2']]
df = df[df['Revision 1'] == df['Revision 2']]
df['Frequency'] = df['Frequency'].apply(lambda x: math.log10(x))

data_transpose = []

def one_revision_LG(df):
    # Convert categorical variables to numeric (if POS and Prev_POS are strings)
    df_encoded = pd.get_dummies(df, columns=['POS', 'POS Context', 'Model 1'])
    print("Get encoded dataframe")

    # Prepare X and y
    X = df_encoded.drop(columns=['KL', 'Seed 1', 'Seed 2', 'Revision 1', 'Revision 2', 'Model 2', 'Unnamed: 0'])
    y = df_encoded['KL']
    # print("X and y prepared")

    # import ipdb; ipdb.set_trace()

    # Fit the linear regression model
    model = LinearRegression()
    # print("Model created")
    model.fit(X, y)
    # print("Model fitted")

    # Print the importance (coefficients)
    coefficients = pd.Series(model.coef_, index=X.columns)
    # print("Feature Contributions to KL Divergence:")
    # print(coefficients.sort_values(ascending=False))

    obj = {}
    for index, row in coefficients.items():
        obj[index] = row

    data_transpose.append(obj)

for revision in df['Revision 1'].unique():
    print(revision)
    one_revision_LG(df[df['Revision 1'] == revision])

df_transpose = pd.DataFrame(data_transpose)
df_transpose.to_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_pos_linear_regression_transpose.csv')
