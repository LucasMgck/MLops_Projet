#!/usr/bin/env python
# coding: utf-8

# # Projet MLops

# #### Groupe : Lucas - Klervi - Elena - Thomas R
# vasseur.corentin@gmail.com

# # Mettre dans le requirement.txt

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y


# In[89]:


get_ipython().system('pip freeze > requirements.txt')


# In[3]:


################################################


# ## Import des données

# In[71]:


def load_and_describe_data(file_path):
    """
    Charge un fichier CSV et fournit un aperçu général des données.
    :param file_path: Chemin du fichier CSV
    :return: DataFrame Pandas
    """
    df = pd.read_csv(file_path)


    return df


# In[73]:


file_path = "data/revenus.csv"
df = load_and_describe_data(file_path)
print("Aperçu des premières lignes du jeu de données :")
print(df.head())

print("\nInformations générales sur le dataset :")
print(df.info())


# # Statistiques descriptives

# In[12]:


def descriptive_statistics(df):
    """
    Affiche les statistiques descriptives des variables numériques
    et les fréquences/proportions pour chaque variable catégorielle.
    
    :param df: DataFrame Pandas contenant les données à analyser
    """
    # Statistiques descriptives pour les variables numériques
    print("Statistiques descriptives des variables numériques :")
    print(df.describe())
    
    # Compter le nombre de valeurs uniques pour chaque variable catégorielle
    print("\nNombre de valeurs uniques pour chaque variable catégorielle :")
    unique_counts = df.select_dtypes(include=['object']).nunique()
    print(unique_counts)
    
    # Fréquence et proportion des valeurs pour chaque variable catégorielle
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\nFréquence et proportion des modalités pour la variable '{col}':")
        
        # Calcul des fréquences et des proportions
        value_counts = df[col].value_counts()
        proportions = (value_counts / len(df)) * 100  # Proportion en pourcentage
        
        # Créer un DataFrame pour afficher fréquence et proportion côte à côte
        freq_prop_df = pd.DataFrame({'Fréquence': value_counts, 'Proportion (%)': proportions})
        
        print(freq_prop_df)


# In[14]:


descriptive_statistics(df)


# ## Distribution des variables numériques

# In[17]:


def plot_numerical_distributions(df):
    """
    Affiche des histogrammes pour les variables numériques.
    :param df: DataFrame Pandas
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f"Distribution de la variable '{col}'")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.show()

# Exemple d'utilisation
plot_numerical_distributions(df)


# ## Distribution des valeurs catégorielles

# In[19]:


def plot_categorical_distributions(df):
    """
    Affiche des diagrammes en barres pour les variables catégorielles.
    :param df: DataFrame Pandas
    """
    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df[col], palette="viridis", order=df[col].value_counts().index)
        plt.title(f"Distribution de la variable '{col}'")
        plt.xlabel("Nombre d'observations")
        plt.ylabel(col)
        plt.show()

# Exemple d'utilisation
plot_categorical_distributions(df)


# ## Matrice de corrélation

# In[21]:


def plot_correlation_matrix(df):
    """
    Affiche la matrice de corrélation pour les variables numériques.
    :param df: DataFrame Pandas
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice de corrélation")
    plt.show()

# Exemple d'utilisation
plot_correlation_matrix(df)


# On remarque la presence de ? dans le jeu de donnée, on decide de les supprimer. Et de regrouper les modalités rares

# # Recodage des variables

# In[75]:


# Fonction pour supprimer les "?"

def drop_data(df):
    """
    Supprime les lignes contenant un '?' dans un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame à traiter.
    
    Returns:
        pd.DataFrame: Le DataFrame nettoyé sans les lignes contenant '?'.
    """
    # Supprimer les lignes où au moins un élément est '?'
    df_cleaned = df[~df.isin(['?']).any(axis=1)]
    return df_cleaned


# In[77]:


df=drop_data(df)


# In[79]:


# Compter le nombre de '?' dans chaque colonne
question_marks = (df == '?').sum()

# Afficher le résultat
print(question_marks)

# Nombre total de '?' dans tout le DataFrame
total_question_marks = question_marks.sum()
print(f"Nombre total de '?': {total_question_marks}")


# In[81]:


# Fonction de recodage des variables

def numeriser_toutes_colonnes(df):
    """
    Cette fonction numérise plusieurs colonnes spécifiées dans le DataFrame 
    avec des mappings prédéfinis directement dans la fonction.

    :param df: DataFrame contenant les données à numériser
    :return: DataFrame avec les colonnes numérisées
    """




    #Suppression de la variable education : c'est la même que educational num

    if 'education' in df.columns:
        df = df.drop(columns=['education'])
        
    # Mapping pour 'workclass'
    workclass_mapping = {
        'Private': 1,
        'Self-emp-not-inc': 2,
        'Local-gov': 3,
        'State-gov': 4,
        'Self-emp-inc': 5,
        'Federal-gov': 6,
        'Without-pay': 7,
        'Never-worked': 8
    }
    
    # Mapping pour 'marital-status'
    marital_status_mapping = {
        'Never-married': 1,
        'Married-civ-spouse': 2,
        'Widowed': 3,
        'Divorced': 4,
        'Separated': 5,
        'Married-spouse-absent': 6,
        'Married-AF-spouse': 7
    }
    
    # Mapping pour 'occupation'
    occupation_mapping = {
        'Machine-op-inspct': 1,
        'Farming-fishing': 2,
        'Protective-serv': 3,
        'Other-service': 4,
        'Prof-specialty': 5,
        'Craft-repair': 6,
        'Adm-clerical': 7,
        'Exec-managerial': 8,
        'Tech-support': 9,
        'Sales': 10,
        'Priv-house-serv': 11,
        'Transport-moving': 12,
        'Handlers-cleaners': 13,
        'Armed-Forces': 14
    }
    
    # Mapping pour 'relationship'
    relationship_mapping = {
        'Own-child': 1,
        'Husband': 2,
        'Not-in-family': 3,
        'Unmarried': 4,
        'Wife': 5,
        'Other-relative': 6
    }
    
    # Mapping pour 'race'
    race_mapping = {
        'Black': 1,
        'White': 2,
        'Asian-Pac-Islander': 3,
        'Other': 4,
        'Amer-Indian-Eskimo': 5
    }
    
    # Mapping pour 'gender'
    gender_mapping = {
        'Male': 1,
        'Female': 2
    }
    
    # Mapping pour 'native-country'
    native_country_mapping = {
        'United-States': 1,
        'Peru': 2,
        'Guatemala': 3,
        'Mexico': 4,
        'Dominican-Republic': 5,
        'Ireland': 6,
        'Germany': 7,
        'Philippines': 8,
        'Thailand': 9,
        'Haiti': 10,
        'El-Salvador': 11,
        'Puerto-Rico': 12,
        'Vietnam': 13,
        'South': 14,
        'Columbia': 15,
        'Japan': 16,
        'India': 17,
        'Cambodia': 18,
        'Poland': 19,
        'Laos': 20,
        'England': 21,
        'Cuba': 22,
        'Taiwan': 23,
        'Italy': 24,
        'Canada': 25,
        'Portugal': 26,
        'China': 27,
        'Nicaragua': 28,
        'Honduras': 29,
        'Iran': 30,
        'Scotland': 31,
        'Jamaica': 32,
        'Ecuador': 33,
        'Yugoslavia': 34,
        'Hungary': 35,
        'Hong': 36,
        'Greece': 37,
        'Trinadad&Tobago': 38,
        'Outlying-US(Guam-USVI-etc)': 39,
        'France': 40,
        'Holand-Netherlands': 41
    }
    
    # Mapping pour 'income'
    income_mapping = {
        '<=50K': 0,
        '>50K': 1
    }
    
    # List des colonnes à numériser avec leurs mappings
    mappings = {
        'workclass': workclass_mapping,
        'marital-status': marital_status_mapping,
        'occupation': occupation_mapping,
        'relationship': relationship_mapping,
        'race': race_mapping,
        'gender': gender_mapping,
        'native-country': native_country_mapping,
        'income': income_mapping
    }

    # Suppression des espaces dans les colonnes concernées
    cols_to_strip = ['workclass', 'occupation', 'native-country']
    for col in cols_to_strip:
        df[col] = df[col].str.strip()

    # Appliquer les mappings de numérisation à chaque colonne
    for col, mapping in mappings.items():
        df[col] = df[col].replace(mapping)
    
    return df


# In[83]:


df=numeriser_toutes_colonnes(df)
df


# # Analyse Exploratoire des données

# In[54]:


def full_data_analysis(file_path):
    """
    Effectue une analyse complète des données.
    :param file_path: Chemin du fichier CSV
    """

    df = load_and_describe_data(file_path)
    df_cleaned = drop_data(df)

    # Vérifier si le DataFrame nettoyé est vide
    if df_cleaned.empty:
        print("Attention, le DataFrame nettoyé est vide après la suppression des '?'!")
    else:
        # Numériser les colonnes avec des mappings
        df_cleaned = numeriser_toutes_colonnes(df_cleaned)

    print("\n\nDonnées nettoyées et numérisées :")
    print(df_cleaned.info())


    descriptive_statistics(df_cleaned)
    plot_numerical_distributions(df_cleaned)
    plot_categorical_distributions(df_cleaned)
    plot_correlation_matrix(df_cleaned)

# Exemple d'utilisation
full_data_analysis("data/revenus.csv")


# # Modélisation

# In[85]:


def model(df):
    """
    Entraîne et évalue plusieurs modèles de classification sur le DataFrame d'entrée.

    Parameters:
    - df : Le DataFrame contenant les données. 

    Returns:
    - pd.DataFrame: Un DataFrame contenant les résultats de l'évaluation des modèles.

    Cette fonction effectue les étapes suivantes :
    1. Divise les données d'entrée en ensembles d'entraînement et de test.
    2. Définit un ensemble de modèles de classification, y compris DecisionTree, KNeighbors, LogisticRegression et RandomForest.
    3. Spécifie des grilles d'hyperparamètres pour chaque modèle pour l'ajustement des hyperparamètres.
    4. Itère sur chaque modèle, créant un pipeline avec ou sans mise à l'échelle en fonction du type de modèle.
    5. Utilise GridSearchCV pour trouver les meilleurs hyperparamètres pour chaque modèle.
    6. Affiche et enregistre les meilleurs hyperparamètres, la précision, le score et le rapport de classification pour chaque modèle sur l'ensemble de test.

    Remarque : Cette fonction suppose que la variable cible est 'SALAIRE' dans le DataFrame d'entrée.

    Example:
    data = pd.read_csv('votre_data.csv')
    results = model(data)
    """

    y = df.income
    X = df.drop('income', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    models = {
        'DecisionTree': DecisionTreeClassifier(),
        'KNeighbors': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier()
    }

    param_grid = {
        'DecisionTree': {'model__criterion': ['gini', 'entropy'], 'model__max_depth': [None, 5, 10, 15]},
        'KNeighbors': {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']},
        'LogisticRegression': {'model__C': [0.1, 1, 10], 'model__penalty': ['l1', 'l2']},
        'RandomForest': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 5, 10, 15],
                         'model__criterion': ['gini', 'entropy']}
    }

    for model_name, model in models.items():
        if 'KNeighbors' in model_name or 'LogisticRegression' in model_name:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('model', model)
            ])

        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        y_pred = grid_search.predict(X_test)
        print(f"Accuracy for {model_name}: {accuracy_score(y_test, y_pred)}")
        print(f"Score for {model_name}: {grid_search.score(X_test, y_test)}")
        print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
        print("=" * 80)


# In[87]:


# Exemple d'utilisation
model(df)


# In[ ]:




