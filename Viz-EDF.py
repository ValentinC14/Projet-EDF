## LIBRAIRIES

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import re
import plotly.express as px

streamlit cheet sheat
## CODE

st.title('Visualisation des données météociel en lien avec le projet EDF')

st.header('''**Dataframe d'étude (random sample)**''')


nrows = st.sidebar.slider('Nombre de lignes du dataframe', min_value = 1, max_value = 1300000)

## CHARGEMENT DU DATA (! CHANGER LE PATH DU CSV)
def load_data(nrows):
    df = pd.read_csv('/Users/valentincatherine/Desktop/Projet-EDF/liste_biais.csv', index_col=0)
    #df = df.dropna()
    return df.sample(nrows)

## CRÉATION DE CATÉGORIES POUR LES FEATURES DISTANCE ET ALTITUDE
def create_range_dist_alt(df):
    bins_dist = [0, 5, 10, 20, 30, 40, 50]
    names_dist = ['<5km', '5-10km', '10-20km', '20-30km', '30-40km', '40-50km', '+50km']
    bins_alt = [0, 10, 50, 100, 500]
    names_alt = ['<10m', '10-50m','50-100m','100-500m','+500m']
    d_dist = dict(enumerate(names_dist, 1))
    d_alt = dict(enumerate(names_alt, 1))
    df['range_dist'] = np.vectorize(d_dist.get)(np.digitize(df['distance_station'], bins_dist))
    df['range_alt'] = np.vectorize(d_alt.get)(np.digitize(df['altitude'], bins_alt))
    return df

df = load_data(nrows)
df = create_range_dist_alt(df)
st.dataframe(df.head()) ## AFFICHER LE DATAFRAME SUR STREAMLIT


## AFFICHAGE DISTRIBUTION HISTORIGRAMME + KDE

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

## CRÉATION D'UN NOUVEAU DATAFRAME AVEC SEULEMENT UN POINT PAR STATION
df_un_point_par_station = df.drop(columns = ['range_dist','range_alt']).groupby(by = 'numero_station').mean()
df_un_point_par_station = create_range_dist_alt(df_un_point_par_station)
st.dataframe(df_un_point_par_station) ## AFFICHAGE DU DF POUR VOIR SI C'EST CORRECT

## GRAPH DE L'ALTITUDE EN FONCTION DU BIAIS
def plot_alt_vs_biais():
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x = 'altitude', y = 'biais', hue = 'range_dist', size='range_dist', sizes = (20,200), legend = 'full',alpha = 0.75, data = df_un_point_par_station)
    return st.pyplot(fig)


st.header('''**Affichage de l'altitude selon le biais**''')
st.write('La corrélation est de %.3f' %(df_un_point_par_station['altitude'].corr(df_un_point_par_station['biais'])))

plot_alt_vs_biais()

# AFFICHAGE 3D Altitude/Distance/Biais

st.header('''**Affichage 3D - biais VS distance VS altitude **''')

def plot_3D_alt_dist_biais():
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    fig = px.scatter_3d(df_un_point_par_station, x='biais', y='distance_station', z="altitude", opacity=0.7)
    return st.write(fig)

plot_3D_alt_dist_biais()

## AFFICHAGE DISTRIBUTION DES BIAIS SELON LA DISTANCE

st.header('''**Distribution du biais selon leur distance**''')

def multi_plot_kde_dist(var):

    def ax_settings(ax, var_name,x_min, x_max, mu, sigma):
        ax.set_xlim(x_min,x_max)
        ax.set_yticks([])

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.spines['bottom'].set_edgecolor('#444444')
        ax.spines['bottom'].set_linewidth(2)

        ax.text(0.02, 0.05, var_name, fontsize=15, fontweight="bold", transform = ax.transAxes)
        ax.text(0.65,0.9, r'''$\mu=%.3f,\ \sigma=%.3f$''' %(mu, sigma), fontweight="bold", fontsize=12, transform = ax.transAxes, color='red')

        return None

    freq = df[var].value_counts(normalize=True)*100
    NB_GROUP = len(freq)
    #st.write(freq)
    fig = plt.figure(figsize=(12,17))
    gs = gridspec.GridSpec(nrows = NB_GROUP, ncols=2, figure=fig, width_ratios=[3,1], height_ratios = [1]*NB_GROUP, wspace=0.2, hspace=0.2)
    ax = [None]*(NB_GROUP + 1)
    features = freq.index.to_list()

    for i in range(NB_GROUP):
        ax[i] = fig.add_subplot(gs[i,0])
        mu = np.mean(df[df[var] == features[i]].biais)
        sigma = np.std(df[df[var] == features[i]].biais)

        if var == 'range_dist':
            ax_settings(ax[i], 'Dist: ' + features[i], -10, 10, mu, sigma)
        else:
            ax_settings(ax[i], 'Alt: ' + features[i], -10, 10, mu, sigma)

        sns.kdeplot(data=df[df[var] == features[i]].biais,
        ax=ax[i], shade=True, color="blue", legend=False)

    ax[NB_GROUP] = fig.add_subplot(gs[:, 1])
    ax[NB_GROUP].spines['right'].set_visible(False)
    ax[NB_GROUP].spines['top'].set_visible(False)
    ax[NB_GROUP].barh(features, freq, color='#004c99', height=0.4)
    ax[NB_GROUP].set_xlim(0,100)
    ax[NB_GROUP].invert_yaxis()
    ax[NB_GROUP].text(1.09, -0.04, '(%)', fontsize=10, transform = ax[NB_GROUP].transAxes)
    ax[NB_GROUP].tick_params(axis='y', labelsize = 14)
    return st.pyplot(fig)

selector = st.selectbox('Select', ['range_dist','range_alt'])
multi_plot_kde_dist(selector)

##
