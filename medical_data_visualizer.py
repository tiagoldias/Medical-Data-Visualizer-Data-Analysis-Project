import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
def bmi_classification(row):
    bmi = row['weight'] / (row['height'] / 100)**2
    if bmi > 25:
        return 1
    return 0
df['overweight'] = df.apply(bmi_classification, axis=1)

# 3
def normalize_chol(value):
    if value > 1:
        return 1
    return 0

df['cholesterol'] = df['cholesterol'].apply(normalize_chol)

df['gluc'] = df['gluc'].apply(normalize_chol)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable'])['value'].value_counts().reset_index(name='total')
    

    # 7
    cardio_plot = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat)    


    # 8
    fig = cardio_plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
             (df['height'] >= df['height'].quantile(0.025)) & 
             (df['height'] <= df['height'].quantile(0.975)) &
             (df['weight'] >= df['weight'].quantile(0.025)) &
             (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones(corr.shape, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, linewidths=0.5, ax=ax)



    # 16
    fig.savefig('heatmap.png')
    return fig
