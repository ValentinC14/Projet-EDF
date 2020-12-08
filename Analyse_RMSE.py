## LIBRAIRIES

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

## CODE

st.title('Visualisation des données météociel en lien avec le projet EDF')

st.header('''**Dataframe d'étude (random sample)**''')




## CHARGEMENT DU DATA (! CHANGER LE PATH DU CSV)
def load_data():
    #df1 = pd.read_excel('/Users/valentincatherine/Desktop/Projet-EDF/temp_dist_rmse_part1.xlsx', index_col=0)
    df2 = pd.read_excel('/Users/valentincatherine/Desktop/Projet-EDF/temp_dist_rmse_part2.xlsx', index_col=0)
    #df = pd.concat([df1,df2])
    return df2

num_bins = st.sidebar.slider('''Nombre de bins de l'historigramme''', 1,200)
plt.style.use('ggplot')

st.header('**Historigramme des biais**')

def plot_hist(num_bins, hist=True, kde=False):
    fig,ax = plt.subplots(figsize=(18,10))
    df = load_data()
    plt.xlabel('Biais', fontweight='bold')
    plt.ylabel('Counts', fontweight='bold')
    mu = np.mean(df.RMSE)
    sigma = np.std(df.RMSE)
    plt.xlim(mu - 5*sigma, mu + 5*sigma)
    ax = sns.distplot(df.RMSE, bins=num_bins, hist = True, kde = False, color = 'green')
    plt.title(r'''$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$''' %(mu, sigma), fontweight="bold")
    return st.pyplot(fig)

plot_hist(num_bins)

def plot_alt_vs_biais():
    fig, ax = plt.subplots()
    df = load_data()
    df = df[df['dist_km'] < 500]
    ax = sns.scatterplot(x = 'dist_km', y = 'RMSE', sizes = (20,200), legend = 'full',alpha = 0.75, data = df)
    st.header('''**Affichage de l'altitude selon le biais**''')
    st.write('La corrélation est de %.3f' %(df['dist_km'].corr(df['RMSE'])))

    return st.pyplot(fig)

plot_alt_vs_biais()
