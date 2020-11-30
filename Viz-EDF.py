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

def create_range_dist():
    df_un_point_par_station = df.groupby(by = 'numero_station').mean()

    bins = [0, 5, 10, 20, 30, 40, 50]
    names = ['<5km', '5-10', '10-20', '20-30', '30-40', '40-50', '+50']

    d = dict(enumerate(names, 1))

    df_un_point_par_station['range_dist'] = np.vectorize(d.get)(np.digitize(df_un_point_par_station['distance_station'], bins))
    return df_un_point_par_station

df_un_point_par_station = create_range_dist()

def plot_alt_vs_biais():
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x = 'altitude', y = 'biais', hue = 'range_dist', size='range_dist', sizes = (20,200), legend = 'full',alpha = 0.75, data = df_un_point_par_station)
    return st.pyplot(fig)


st.header('''**Affichage de l'altitude selon le biais**''')
st.write('La corrélation est de %.3f' %(df_un_point_par_station['altitude'].corr(df_un_point_par_station['biais'])))

plot_alt_vs_biais()


# AFFICHAGE 3D Altitude/Distance/Biais
def plot_3D_alt_dist_biais():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = df_un_point_par_station['biais'].to_list()
    y = df_un_point_par_station['distance_station'].to_list()
    z = df_un_point_par_station['altitude'].to_list()
    ax.scatter(x,y,z)
    ax.set_xlabel("Biais")
    ax.set_ylabel("Distance avec la station synop")
    ax.set_zlabel("Différence d'altitude")
    return st.pyplot(fig)

st.header('''**Affichage 3D - biais VS distance VS altitude **''')

st.write(df_un_point_par_station['altitude'].corr(df_un_point_par_station['biais']))
plot_3D_alt_dist_biais()
