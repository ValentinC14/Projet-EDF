# Projet-EDF

<img align="right" width="400" height="150" src="https://github.com/valentincthrn/projet-edf/blob/main/images/logo_edf.png">

This project is part of the Centralien educational curriculum. It lasts 18 months and is carried out in a team of 12 people in collaboration with EDF. Here, I will detail the technical aspect undertaken by the team and myself. This project is part of the Centralien educational curriculum. It lasts 18 months and is carried out in a team of 12 people in collaboration with EDF. Here, I will detail the technical aspect undertaken by the team and myself. Chronologically, the work was divided as follows: 

- [step 1 : search for reliable meteorological websites](#s1)
- [step 2 : production of 300+ data sheets for each of the connected objects throughout France](#s2)
- [step 3 : conclusion on the data selection criteria](#s3)
- [step 4 : development of an AI algorithm to predict the load factor of French wind farms](#s4)

/!\ I will only provide the Python codes for the last step of Machine Learning. Indeed, the preceding codes were carried out in group and can be confidential for our customer. 

<a name="s1"></a>
## Step 1 : Search for reliable meteorological websites

Since the project was largely exploratory by the client, a first step was to gather and classify the different weather websites. The objective was to provide the extraction team with the sites to be extracted. This classification was done according to different criteria, mainly the number of stations, the type of data and the historical depth. This part is essential in all data driven projects. Indeed, the quality of the data is an indispensable component. I learned a lot during this period, which lasted 4 months, and understood the importance of good data research. 

Here you can see an excel extract of our research : 

(mettre photo excel)

<a name="s2"></a>
## Step 2 : Analysis sheets

Once the data had been extracted, the idea was to carry out graphic displays and statistical analyzes on each of the connected objects collected. Next, the deliverable was to provide EDF with selection criteria on the quality of the data. But how can we qualify the quality of the latter?

In the midst of all this collected data, we have stations that transmit highly reliable data and serve as a benchmark. These are data from 63 stations called "synop" and they are distributed homogeneously over all French territories.

Thus, each extracted meteorological station was associated with the closest reference synop station. We used 4 statistical indicators to judge the similarity or difference between a station and its reference (= its synop). The criteria were: bias, standard deviation, MAE and MAPE. On the right, the red dots are the weather stations to which we must know their reliability. In blue, these are the reference stations, or also called "synop". 

One problem encountered during this step is the length of time it takes to complete a form. Indeed, all the graphs returned by the python code had to be copied and pasted on a word document. Furthermore, the execution time of the code was long and more than 15 graphics were present. So, I decided to automate the process by creating directly a local folder with all the pre-filled images and texts for each station. 

<a name="s3"></a>
## Step 3 : Conclusion on criteria

We must not forget that the goal was to provide quality criteria to EDF. The characteristics of each meteoscience station relative to its reference station are the following (averaged at different time steps): MAE, RMSE, MAPE, altitude, distance. 
After extracting and specifying all these indicators, the goal was to choose the most qualitative stations. 

To do this, we performed an unsupervised learning algorithm (KMeans).We plotted in a 3D plane different indicators between them and then identified the indicators most likely to play on the quality. 

The results are summarized in the following table: (table du google slide)

<a name="s4"></a>
## Step 4 : Machine Learning

This last step was optional but it was the most interesting point for me. In the last 2 months, we formed a team of 3 people to deal with the Machine Learning part. I mainly led and carried out this part. 

After a lot of discussions with the client, the goal was to predict the load factors of the wind turbines between the time t + 1h and t + 4h from the historical wind data before time t. The load factor is defined as the number of hours of wind power per hour. 


The load factor is defined as the ratio between the output of the wind turbines and their installed power. Thus a load factor of 0% corresponds to a wind farm at standstill (no production). A factor of 100% means that the production is at its maximum. 

In contrast to the method used by EDF, and due to the limited resources I have, I have adopted an end-to-end approach as illustrated below. 

(put method)

The data from the Extraction division are several Excels. Each excel corresponds to ONLY one weather station. Within these excels, the indexes are the time step (here hourly step) and the columns are our features (date, meteoscel temperature, synop temperature, meteoscel wind, synop wind). So, the first step is to concatenate all the excels together in a dataframe. 

(put code)

Then a pivot is applied to index each time step and the columns correspond to the data of each station. 

(put code)

However, during the concatenation, missing values appeared because of the join on all the existing time steps. After analysis, I first deleted all stations with more than 6% missing values. For the other stations, I applied an experimental class from Scikit-learn called "Iterative Imputer" which replaced the missing values very well. 

(put missing values code)

The data is now ready and now it is time to implement AI algorithms. The first step which I think is essential is to start simple via linear regression. More generally, I used a class called "LazyPredict" which allows to make predictions with all models compatible with our data types. 

(put Lazy Predict code)

The results show a better efficiency of the regularised linear regression models (ElasticNet, Lasso). 

Then, with GridSearchCV, I found the optimal hyperparameter of ElasticNet: the l1_ratio equal to 0.1. 

(gridsearch code)

Then the results of the model on the test data are the following: 

(put picture)

On average, our model predicts a -1.3% difference in load factor. Our client was satisfied with the result. 

Then, out of curiosity, I developed a simple neural network structured in a relatively simple way. 

(code NN)

I obtained less satisfactory results with an average bias of -3.31% on the test data. 





