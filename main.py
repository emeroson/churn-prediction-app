import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

# Informations générales
print(df.info())

# Statistiques
print(df.describe())

# Nombre de clients qui quittent
print(df['Churn'].value_counts())

# Churn par type de contrat
print(df.groupby('Contract')['Churn'].value_counts())

# Churn selon le prix
print(df.groupby('Churn')['MonthlyCharges'].mean())

# Churn selon durée abonnement
print(df.groupby('Churn')['tenure'].mean())

import pandas as pd

# Charger le dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -------------------------
# 🔍 ANALYSE (déjà faite)
# -------------------------

# Transformer Churn en 0 / 1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Supprimer colonne inutile
df = df.drop('customerID', axis=1)

# Transformer les variables catégorielles
df = pd.get_dummies(df)

# -------------------------
# 🤖 MACHINE LEARNING
# -------------------------

# Séparer X et y
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train / Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
from sklearn.metrics import accuracy_score

print("Accuracy :", accuracy_score(y_test, y_pred))