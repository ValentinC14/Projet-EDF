# Projet-EDF

This project is part of the Centralien educational curriculum. It lasts 18 months and is carried out in a team of 12 people in collaboration with EDF. Here, I will detail the technical aspect undertaken by the team and myself. This project is part of the Centralien educational curriculum. It lasts 18 months and is carried out in a team of 12 people in collaboration with EDF. Here, I will detail the technical aspect undertaken by the team and myself. Chronologically, the work was divided as follows: 
- step 1 : search for reliable meteorological websites
- step 2 : production of 300+ data sheets for each of the connected objects throughout France
- step 3 : conclusion on the data selection criteria
- step 4 : development of an AI algorithm pour prédire le facteur de charge des parc éoliens français

## Step 1 : Search for reliable meteorological websites

Since the project was largely exploratory by the client, a first step was to gather and classify the different weather websites. The objective was to provide the extraction team with the sites to be extracted. This classification was done according to different criteria, mainly the number of stations, the type of data and the historical depth. This part is essential in all projects where data is entered. Indeed, the quality of the data is an indispensable component. I learned a lot during this period, which lasted a few months, and understood the importance of good data research. 

Here you can see an excel extract of our results : 

(mettre photo excel)

# Projet-EDF

This project is part of the Centralien educational curriculum. It lasts 18 months and is carried out in a team of 12 people in collaboration with EDF. Here, I will detail the technical aspect undertaken by the team and myself. This project is part of the Centralien educational curriculum. It lasts 18 months and is carried out in a team of 12 people in collaboration with EDF. Here, I will detail the technical aspect undertaken by the team and myself. Chronologically, the work was divided as follows: 
- step 1 : search for reliable meteorological websites
- step 2 : production of 300+ data sheets for each of the connected objects throughout France
- step 3 : conclusion on the data selection criteria
- step 4 : development of an AI algorithm to predict the load factor of French wind farms

## Step 1 : Search for reliable meteorological websites

Since the project was largely exploratory by the client, a first step was to gather and classify the different weather websites. The objective was to provide the extraction team with the sites to be extracted. This classification was done according to different criteria, mainly the number of stations, the type of data and the historical depth. This part is essential in all projects where data is entered. Indeed, the quality of the data is an indispensable component. I learned a lot during this period, which lasted a few months, and understood the importance of good data research. 

Here you can see an excel extract of our results : 

(mettre photo excel)

## Step 2 : Analysis sheets

Once the data had been extracted, the idea was to carry out graphic displays and statistical analyzes on each of the connected objects collected. Next, the deliverable was to provide EDF with selection criteria on the quality of the data. But how can we qualify the quality of the latter?

In the midst of all this collected data, we have stations that transmit highly reliable data and serve as a benchmark. These are data from 279 stations called "synop" and they are distributed homogeneously over all French territories.

Thus, each extracted meteorological station was associated with the closest reference synop station. We used 4 statistical indicators to judge the similarity or difference between a station and its reference (= its synop). The criteria were: bias, standard deviation, MAE and MAPE. 

One problem encountered during this step is the length of time it takes to complete a form. Indeed, all the graphs returned by the python code had to be copied and pasted on a word document. However, the execution time of the code was long and more than 15 graphics were present. So, I decided to automate the process by creating a local folder with all the pre-filled images and texts. 

## Step 3 : Conclusion on criteria

We must not forget that the goal was to provide quality criteria to EDF. The characteristics of each meteoscience station relative to its reference station are the following (averaged at different time steps): MAE, RMSE, MAPE, altitude, distance. 
After extracting and specifying all these indicators, the goal was to choose the most qualitative stations. 

To do this, we performed an unsupervised learning algorithm (KMeans).We plotted in a 3D plane different indicators between them and then identified the indicators most likely to play on the quality. 

The results are summarized in the following table: (table du google slide)






