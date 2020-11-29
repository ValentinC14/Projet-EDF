import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import seaborn as sns


st.title('Visualisation des données météociel en lien avec le projet EDF')

st.header('''**Dataframe d'étude (random sample)**''')
#

nrows = st.sidebar.slider('Nombre de lignes du dataframe', min_value = 1, max_value = 1300000)

def load_data(nrows):
    df = pd.read_csv('/Users/valentincatherine/Desktop/Projet-EDF/liste_biais.csv', index_col=0)
    #df = df.dropna()
    return df.sample(nrows)

df = load_data(nrows)
st.dataframe(df.head())

num_bins = st.sidebar.slider('''Nombre de bins de l'historigramme''', 1,200)
plt.style.use('ggplot')

st.header('**Historigramme des biais**')

def plot_hist(num_bins, nrows, hist, kde):
    fig,ax = plt.subplots()
    df = load_data(nrows)
    plt.xlabel('Biais', fontweight='bold')
    plt.ylabel('Counts', fontweight='bold')
    mu = np.mean(df.biais)
    sigma = np.std(df.biais)
    plt.xlim(mu - 5*sigma, mu + 5*sigma)
    ax = sns.distplot(df.biais, bins=num_bins, hist = hist, kde = kde, color = 'green')
    plt.title(r'''$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$''' %(mu, sigma), fontweight="bold")
    return st.pyplot(fig)

selector = st.selectbox('TYPE DE GRAPHE', ['HISTORIGRAMME SEULEMENT','KDE SEULEMENT','HISTORIGRAMME ET KDE'])
st.write(selector)
if selector == 'HISTORIGRAMME SEULEMENT':
    plot_hist(num_bins, nrows, hist =True, kde = False)
if selector == 'KDE SEULEMENT':
    plot_hist(num_bins, nrows, hist =False, kde = True)
if selector == 'HISTORIGRAMME ET KDE':
    plot_hist(num_bins, nrows, hist =True, kde = True)
