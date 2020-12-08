'''
Ce programme permet de créer un dataframe avec les colonnes suivantes :
temperature du synop, temperature de la meteociel, leur RMSE
et enfin, le plus important, la distance les séparant en km

Pour ceci, on part de deux base de données déjà existantes :
- 1ère : un dossier nommé 'liste_tem_synop_mc' qui contient plusieurs CSV où
  chaque csv est une station météociel spécifique avec sa température à chaque
  pas (1 heure), la date mais également, la température synop associé du plus
  proche
  CSV : colonne = [Date, température synop, température météociel]
 - 2ème : Une base de donnée 'Station Météociel Synop'
   colonnes : les id du meteociel et du synop associé, et leur distance en km

 Finalement, on 'join' ces deux base de données sur l'id météociel
 permettant d'avoir un base de données avec seulement les températures et la
 distance séparant les deux stations

 L'objectif est d'ensuite, réaliser des figures de corrélations des figures
 selon la distance en km

'''

import os
import numpy as np
import pandas as pd

URL_ALL =[]
for path, subdirs, files in os.walk('/Users/valentincatherine/Desktop/Projet-EDF'):
    for name in files:
        if name[-3:] == 'csv' :
            print(os.path.join(path, name))

import pandas as pd
##PATH DU DOSSIER
PATH = '/Users/valentincatherine/Desktop/Projet-EDF/liste_temp_synop_mc'
URL_ALL = []
liste_id = []
for name in os.listdir(PATH):
    liste_id.append(name.split('_')[1].split('.')[0])
    URL_ALL.append(os.path.join(PATH, name))

liste_df = []
for index, url in enumerate(URL_ALL):
    df = pd.read_csv(url, sep=';', )
    id = liste_id[index]
    if id != 'Store':
        df['id'] = int(id)
    df = df.drop(['Unnamed: 0'], axis = 1).dropna()
    liste_df.append(df)

df_tot = pd.concat([df for df in liste_df])

df_meteo_synop = pd.read_excel('/Users/valentincatherine/Desktop/Projet-EDF/Station_meteociel_synop.xlsx').rename(columns = {'id_mc':'id'}).drop(['Unnamed: 0'], axis=1)

columns_to_keep = ['temperature_synop', 'temperature_meteociel','difference_temperature','dist_km']

df_merge = pd.merge(df_tot, df_meteo_synop, how = 'left', on='id')[columns_to_keep]

df_merge['RMSE'] = np.sqrt((df_merge['temperature_synop'] - df_merge['temperature_meteociel'])**2)

df_merge_part1 = df_merge.loc[:1000000,:]
df_merge_part2 = df_merge.loc[1000000:,:]

df_merge_part1.to_excel(PATH + '/temp_dist_rmse_part1.xlsx')
df_merge_part2.to_excel(PATH + '/temp_dist_rmse_part2.xlsx')
