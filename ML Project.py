#!/usr/bin/env python
# coding: utf-8

# In[262]:


# Étape 1 : Importation des bibliothèques
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[284]:


# 2. Charger les données"
df = pd.read_csv("C:\\Users\\A\\Desktop\\IA MASTER2\\House Price prediction\\house_prices.csv")

df.head()


# In[285]:


# Étape 3 : Prétraitement simple
# Garder les colonnes numériques pour simplifier
#df = df.select_dtypes(include=["int64", "float64"])
#df = df.dropna(axis=1, how="any")  # Supprimer les colonnes avec valeurs manquantes

df.info()
df.describe()
df.isnull().sum()


# In[286]:


df.info()


# In[287]:


#df.fillna(df.mean(numeric_only=True), inplace=True)
#df.head()


# In[288]:


print(df.columns.tolist())


# In[289]:


# Sélectionner les colonnes de type object avec peu de modalités (ex: < 100)
cat_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 100]

print("Colonnes catégorielles à encoder :", cat_cols)

# Puis encoder uniquement celles-ci
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# In[267]:


# Séparer variables indépendantes et cible
#X = df.drop("Price (in rupees)", axis=1)
#y = df["Price (in rupees)"]


# In[290]:


print("df shape:", df.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
print(df.head())


# In[291]:


df.fillna(0, inplace=True)  # Replace NaNs with 0

# Now redo X and y
X = df.drop(['Index', 'Price (in rupees)'], axis=1)
y = df['Price (in rupees)']

print("X shape:", X.shape)
print("y shape:", y.shape)


# In[292]:


print(df.shape)
print(X.shape)


# In[293]:


df = df.dropna()  # ou df.dropna()


# In[294]:


if df.empty:
    print("Le DataFrame est vide.")
else:
    print("Le DataFrame contient :", df.shape[0], "lignes")


# In[295]:


# Garder uniquement les colonnes numériques mais pas les lignes vides
df = df.select_dtypes(include=["int64", "float64"])
df = df.dropna()


# In[296]:


print(len(X), len(y))  # Les deux doivent avoir plus que 0 lignes et même taille


# In[297]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Masquer les FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Colonnes numériques
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Histogrammes
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
    plt.title(f"Histogramme de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()



# In[313]:


# Filtrer les colonnes numériques
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Matrice de corrélation
plt.figure(figsize=(12, 8))
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()


# In[316]:


# Boîtes à moustaches pour chaque colonne numérique
for col in numerical_df.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=numerical_df[col], color="skyblue")
    plt.title(f"Boxplot de {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


# In[298]:


# Define X and y
X = df.drop(['Index', 'Price (in rupees)'], axis=1)
y = df['Price (in rupees)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[299]:


# Étape 5 : Entraînement du modèle
from sklearn.ensemble import RandomForestRegressor
import joblib

model = RandomForestRegressor()
model.fit(X_train, y_train)


# In[300]:


# Étape 7 : Application Streamlit
# Enregistrer le modèle
joblib.dump(model, "house_price_model.pkl")


# In[303]:


import joblib

obj = joblib.load("house_price_model.pkl")
print(type(obj))


# In[305]:


print(obj)


# In[306]:


joblib.dump({'model': model, 'features': features}, "house_price_model.pkl")


# In[308]:


data = joblib.load("house_price_model.pkl")
model = data['model']
features = data['features']


# In[309]:


obj = joblib.load("house_price_model.pkl")
print(obj)


# In[317]:


# Charger modèle et features (même si features semble incorrect dans le fichier)
obj = joblib.load("house_price_model.pkl")
model = obj['model']



# In[318]:


# Afficher les noms exacts de colonnes attendues par le modèle
print("Colonnes exactes utilisées à l'entraînement :", model.feature_names_in_)



# In[320]:


# Créer les données d'entrée dans le bon ordre avec les bons noms
X_input = pd.DataFrame([[850, 1000]], columns=model.feature_names_in_)



# In[321]:


# Prédire
prediction = model.predict(X_input)
print("Prix estimé :", prediction[0])



# In[ ]:




