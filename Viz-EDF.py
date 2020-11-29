import streamlit as st
import numpy as np
import pandas as import pd
import matplotlib.pyplot as plt


st.title('Visualisation des données météociel en lien avec le projet EDF')

df = pd.read_csv('Projet-EDF/liste_biais.csv')

st.DataFrame(df)
