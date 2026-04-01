# Prédiction du Churn Client – Télécommunications


## Présentation

Ce projet vise à prédire le churn client dans le secteur des télécommunications à l’aide du machine learning.
L’objectif est d’identifier les clients à risque de départ et de proposer des actions pour améliorer la fidélisation.


## Objectifs
	•	Analyser le comportement des clients
	•	Identifier les facteurs influençant le churn
	•	Construire un modèle prédictif
	•	Déployer une application interactive avec Streamlit



## Données
	•	Source : Telco Customer Churn (Kaggle)
	•	Nombre de clients : 7043
	•	Variable cible : Churn (Yes/No)



## Analyse exploratoire
	•	Taux de churn : 26,5% (~1 client sur 4 quitte)
	•	Facteurs de risque :
	•	Contrats mensuels (month-to-month)
	•	Charges mensuelles élevées
	•	Faible ancienneté
	•	Absence de services



 ## Modèle
	•	Algorithme : Régression logistique
	•	Séparation : 80% entraînement / 20% test
	•	Précision : ~82%

 ## Matrice de confusion
	•	Bonne détection des clients fidèles
	•	Certains churners non détectés → amélioration possible



 ## Insights business
	•	Les contrats courts augmentent le churn
	•	Les nouveaux clients sont les plus à risque
	•	Les prix élevés influencent le départ
	•	Les services supplémentaires améliorent la fidélité



 ## Architecture

Pipeline :
	1.	Prétraitement des données (encodage, normalisation)
	2.	Entraînement du modèle
	3.	Sauvegarde (model.pkl, scaler.pkl)
	4.	Déploiement avec Streamlit (app.py)



## Application

Une application Streamlit permet :
	•	Saisie des informations client
	•	Prédiction du churn en temps réel
	•	Visualisation des facteurs de risque
	•	Analyse des insights



 ## Valeur métier
	•	Identifier les clients à risque
	•	Réduire le churn
	•	Améliorer la fidélisation
	•	Optimiser les revenus



## Technologies utilisées
	•	Python
	•	Pandas
	•	Scikit-learn
	•	Streamlit
	•	Plotly



 ## Auteur

ANOH AMON FRANCKLIN HEMERSON
Master 1 Data Science – INSEEDS

Supervisé par : Akposso Didier Martial


##  ## Dashbord en ligne 

lien vers l'application :
https://churn-prediction-app-9wu7bp7drfuybuz68scxsg.streamlit.app/
