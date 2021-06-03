# Projet-EDF

- - - -

<img align="right" width="400" height="150" src="https://github.com/valentincthrn/projet-edf/blob/main/images/logo_edf.png">

This project is part of the Centrale educational curriculum. It lasts 18 months and is carried out in a team of 12 people in collaboration with EDF. Here, I will detail the technical aspect undertaken by the team and myself. This project is part of the Centrale educational curriculum. It lasts 18 months and is carried out in a team of 12 people in collaboration with EDF. Here, I will detail the technical aspect undertaken by the team and myself. Chronologically, the work was divided as follows: 

- [step 1 : search for reliable meteorological websites](#s1)
- [step 2 : production of 300+ datasheets for each of the connected objects throughout France](#s2)
- [step 3 : conclusion on the data selection criteria](#s3)
- [step 4 : development of an AI algorithm to predict the load factor of French wind farms](#s4)

<img align="left" width="50" height="50" src="https://github.com/valentincthrn/projet-edf/blob/main/images/attention_logo.png">

 I will only provide the Python codes for the last step (Machine Learning). Indeed, the preceding codes were carried out in the group and can be confidential for our customer. Furthermore, The notebook available in this Github is an old version for confidentiality reasons.

<a name="s1"></a>
## Step 1 : Search for reliable meteorological websites

Since the project was largely exploratory by the client, a first step was to gather and classify the different weather websites. The objective was to provide the extraction team with the sites to be extracted. This classification was done according to different criteria, mainly the number of stations, the type of data, and the historical depth. This part is essential in all data-driven projects. Indeed, the quality of the data is an indispensable component. I learned a lot during this period, which lasted 4 months, and understood the importance of good data research. 

Here you can see an excel extract of our research : 

<p align="center">
  <img width="800" height="300" src="https://github.com/valentincthrn/projet-edf/blob/main/images/excel_recherche.png">
</p>

<a name="s2"></a>
## Step 2 : Analysis sheets

Once the data had been extracted, the idea was to carry out graphic displays and statistical analyzes on each of the connected objects collected. Next, the deliverable was to provide EDF with selection criteria on the quality of the data. But how can we qualify the quality of the latter?

Amid all this collected data, we have stations that transmit highly reliable data and serve as a benchmark. These are data from 63 stations ([available here in this CSV](https://github.com/valentincthrn/projet-edf/blob/main/postesSynop.csv)) called "synop" and they are distributed homogeneously over all French territories.

<img align="right" width="300" height="300" src="https://github.com/valentincthrn/projet-edf/blob/main/images/map-station.png">

<br>
<br>
<br>
<br>
Thus, each extracted meteorological station was associated with the closest reference synop station. We used 4 statistical indicators to judge the similarity or difference between a station and its reference (= its synop). The criteria were: bias, standard deviation, MAE, and MAPE. On the right, the red dots are the weather stations to which we must know their reliability. In blue, these are the reference stations, or also called "synop". 

<br>
<br>
<br>
<br>
<br>
<br>

One problem encountered during this step is the length of time it takes to complete a sheet. Indeed, all the graphs returned by the python code had to be copied and pasted on a word document. Furthermore, the execution time of the code was long, and more than 15 graphics were present. So, I decided to automate the process by creating directly a local folder with all the pre-filled images and texts for each station. 

Here is one of our 650 Analysis Sheet provided to EDF : [EDF Sheet Fontannes](https://github.com/valentincthrn/projet-edf/blob/main/images/exemple-fiche-analyse.pdf)

<a name="s3"></a>
## Step 3 : Conclusion on criteria

We must not forget that the goal was to provide quality criteria to EDF. The characteristics of each meteociel station relative to its reference station are the following (averaged at different time steps): MAE, RMSE, MAPE, altitude, distance. After extracting and specifying all these indicators, the goal was to choose the most qualitative stations. 

<img align="right" width="300" height="300" src="https://github.com/valentincthrn/projet-edf/blob/main/images/exemple_plot_3D.png">

To do this, we performed an unsupervised learning algorithm (KMeans). We plotted in a 3D plane different indicators between them and then identified the indicators most likely to play on the quality. Here is an example of a 3D plot where each color corresponds to a group.

Finally, after some reflexions and analysis,results on criteria are summarized in the following table : 

<p align="center">
  <img width="225" height="125" src="https://github.com/valentincthrn/projet-edf/blob/main/images/tableau_resultat_critere.png">
</p>


Thus, a station is considered reliable if it is located less than 70km from its reference station (its synop) and at an altitude difference of less than 400m.

I also made some graphical displays on Streamlit where an extract of the code is available [here](https://github.com/valentincthrn/projet-edf/blob/main/Viz-EDF.py).

<a name="s4"></a>
## Step 4 : Machine Learning

This last step was optional but it was the most interesting point for me. In the last 2 months, we formed a team of 3 people to deal with the Machine Learning part. I mainly led and carried out this part. 

After a lot of discussions with the client, the goal was to predict the load factors of the wind turbines between the time t + 1h and t + 4h from the historical wind data before time t. The load factor is defined as the number of hours of wind power per hour. 


The load factor is defined as the ratio between the output of the wind turbines and their installed power. Thus a load factor of 0% corresponds to a wind farm at standstill (no production). A factor of 100% means that the production is at its maximum. 

In contrast to the method used by EDF, and due to the limited resources I have, I have adopted an end-to-end approach as illustrated below. 

<p align="center">
  <img width="800" height="350" src="https://github.com/valentincthrn/projet-edf/blob/main/images/method.jpg">
</p>

To predict the load factors, EDF's method is to use their supercomputer to get an estimate of the wind in the next 24 hours. These supercomputers are based on extremely complex weather models. Then, for each time step (in this case hourly), a model is used to predict the load factor at that same time step. (LF stands for "Load Factor").
As I do not have the computing power, I was forced to remove this block from the pipeline. So I created a model that takes ALL the wind data of the previous 24 hours and predicts the load factor of the next 4 hours. This is an approach called "end to end". 

**Now it's time for some code !**

The data from the Extraction division are several Excels. Each excel corresponds to ONLY one weather station. Within these excels, the indexes are the time step (here hourly step) and the columns are our features (date, meteociel temperature, synop temperature, meteociel wind, synop wind). So, the first step is to concatenate all the excels together in a dataframe. 

- `liste_url` is a list of all the paths to our different excels. 

```python 
def concatenation_dfcomp(liste_url):
    liste_df = [] ### Initialisation de la liste qui permettra de réaliser la concatenation à la fin
    
    columns_to_drop = ['date','temperature_synop','ind','annee','mois','jour','heure','difference_temperature','difference_vent'] ### les colonnes à enlever
    compt_na = 0 ### Un compteur qui va comptabiliser les stations qui ne seront pas concaténer car présentant des valeurs manquantes
    liste_sans_na_id = [] ### Liste qui retriendra tout les identifiants des df qui seront concaténés
    
    for index, url in enumerate(liste_url):
        
        ### Mise en forme du df en accord avec les points précédents
        df = pd.read_csv(url, sep=';', index_col = 'Unnamed: 0')
        id = int(liste_id[index])
        df = df.rename(columns = {'Unnamed: 0':'Datetime'}).drop(columns_to_drop, axis=1)
        df.index = pd.to_datetime(df.index)
        df['vent_meteociel'] = df['vent_meteociel'].apply(lambda x : round(x,2))
        
        ### Variable contenant le nombre de valeurs manquantes sur la colonne température météociel et température vent
        isna = df.isna().sum().tolist()
        
        ### On ajoute pour chaque dataframe une colonne de son identifiant, ce qui permettra de le repérer par la suite
        id_serie = [id] * len(df)
        df['id'] = id_serie

        ### On regarde si isna == [0, 0] en d'autres termes, si le df présente aucune valeurs manquantes sur les données météociel
        if isna != [0, 0]:
            compt_na = compt_na + 1
        
        else:
            liste_sans_na_id.append(id)
            liste_df.append(df)
    
    ### Concaténation des df
    df_tot = pd.concat([df for df in liste_df]).reset_index() ### On reset_index car l'index est notre pas de temps et par la suite, c'est mieux d'éviter d'avoir notre pas de temps en index
    
    return [df_tot, compt_na, liste_sans_na_id]
```

Then a pivot is applied to index each time step and the columns correspond to the data of each station. 

```python
def ajout_colonne_nom(df, liste_id):
    ### date unique est une variable qui sauvegarde toutes les dates différentes du df
    date_unique = df_tot['index'].unique()
    
    liste_nom_colonne_temp = []
    liste_nom_colonne_vent = []
    
    for ident in liste_id:
        N = len(df[df['id']== int(ident)])
        
        for index in range(N):
            liste_nom_colonne_temp.append('temp_{}'.format(ident))
            liste_nom_colonne_vent.append('vent_{}'.format(ident))
    
    df['nom_temp'] = liste_nom_colonne_temp
    df['nom_vent'] = liste_nom_colonne_vent
    
    return df

df = ajout_colonne_nom(df_tot, liste_id_sans_na)
df_vent = df.pivot(index = 'index', columns = 'nom_vent', values = 'vent_meteociel')
```

The purpose of the two previous codes can be summarised in the following illustration.


<p align="center">
  <img width="800" height="350" src="https://github.com/valentincthrn/projet-edf/blob/main/images/resume-preprocess.png">
</p>

However, during the concatenation, missing values appeared because of the join on all the existing time steps. The following code allows you to make this very useful representation. Here, each white "dot" represents a missing value. 

```python 

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)

```

<p align="center">
  <img width="800" height="350" src="https://github.com/valentincthrn/projet-edf/blob/main/images/heatmap.png">
</p>


After analysis, I first deleted all stations with more than 6% missing values. For the other stations, I applied an experimental class from Scikit-learn called "Iterative Imputer" which replaced the missing values very well. 
```python 
from sklearn.impute import IterativeImputer

def impute_na(df):
    ### Création du nouveau dataframe avec les valeurs qui ont été remplacées
    imputer = IterativeImputer(random_state = 42)
    array_df = df.values ### Converti le dataframe en tableau 
    new_array = imputer.fit_transform(array_df)
    df_sans_na = pd.DataFrame(new_array) 
    
    ### Mettre les dates et nom de colonnes de l'ancien sur le nouveau
    liste_col = list(df.columns)
    liste_date = list(df.index)
    df_sans_na.columns = liste_col
    df_sans_na['date'] = liste_date
    df_final = df_sans_na.set_index('date')
    
    return df_final

X = impute_na(df)
```
A heatmap like the one above would show that no missing values remain. (no white spots)

After some pre-processing of the labels, it is time to tackle the implementation of a Machine Learning model. As a reminder, the input data contains 24 points (wind data over the last 24 hours) and the data to be predicted are the 4 instants after t.
The first step which I think is essential is to start simple via linear regression. More generally, I used a class called "LazyPredict" which allows making predictions with all models compatible with our data types. 

```python
from lazypredict.Supervised import LazyRegressor
reg = LazyRegressor(verbose = 2, ignore_warnings = False, custom_metric =None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
```

The results show a better efficiency of the regularised linear regression models (ElasticNet, Lasso). 

<p align="center">
  <img width="300" height="400" src="https://github.com/valentincthrn/projet-edf/blob/main/images/lazy_result.png">
</p>

Then, with GridSearchCV, I found the optimal hyperparameter of ElasticNet: the l1_ratio equal to 0.1. 


Then the results of the model `ElasticNet(l1_ratio = 0.1)`on the test data are the following: 

<p align="center">
  <img width="800" height="350" src="https://github.com/valentincthrn/projet-edf/blob/main/images/result_test_elastic.png">
</p>

On average, our model predicts a -1.3% difference in load factor. Our client was satisfied with the result. 

Then, out of curiosity, I developed a simple neural network structured in a relatively simple way. 

```python
def create_model_simple_avec_dropout(dropout_rate = 0.2):

    inputs = keras.Input(shape=(8208,))


    x = layers.Dense(1024, activation='relu',kernel_initializer="he_normal")(inputs)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, activation='relu',kernel_initializer="he_normal")(x)
    x = layers.Dropout(dropout_rate)(x)
       
    x = layers.Dense(64, activation='relu',kernel_initializer="he_normal")(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(8, activation='relu',kernel_initializer="he_normal")(x)

    outputs = layers.Dense(2)(x)
    
    model = keras.Model(inputs = inputs, outputs = outputs, name='model_simple_avec_dropout')
    
    return model
```

I obtained less satisfactory results with an average bias of -3.31% on the test data. 


<p align="center">
  <img width="800" height="350" src="https://github.com/valentincthrn/projet-edf/blob/main/images/result_test_NN.png">
</p>

- - - -

## Conclusion

The results are satisfactory. Indeed, the limited historical depth of our data (6 months) plays on the performance of our results, especially for the neural network. The ElasticNet regression model seems to be much better than this simple neural network. However, due to lack of time, several extensions that I would have liked to work on were not addressed (setting up an RNN, optimizing the hyperparameters of the neural network, refining the quality of the data...)







