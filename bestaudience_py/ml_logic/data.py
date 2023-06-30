import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer


def cleaning_data(raw_data):
    data_cleaned = raw_data.copy()

    #Première selection de features, on drop celles qu'on ne veut pas
    data_cleaned = data_cleaned.drop(["Remboursement HT","Remboursement Produits HT"
                                      ,"Remboursement Livraison HT","Discount Produits HT"
                                      ,"CA Net HT","CA HT","CA Livraison HT"
                                      ,"Taux de Solde","CA Produits Net HT"
                                      ,"N Produits Remboursés","N Produits Facturés"
                                      ,"Produit - Coloris","Produit - Référence taille"
                                      ,"Produit - Référence Produit","Commande - Méthode de livraison"
                                      ,"Commande - Type de Commande","Remboursement - ID"
                                      ,"Client - ID.1","Commande - Facturation - Pays",],axis=1)

    #Je transforme na en NaN puis drop les na de la colonne Produit - Forme
    na_mask_cat = data_cleaned['Produit - Forme']=='na'
    data_cleaned[na_mask_cat]=np.nan
    data_cleaned = data_cleaned.dropna()

    #Création d'un Df sans les NA pour calculer la moyenne d'age

    mean_birth=data_cleaned.groupby('Client - ID')['Client - Date de Naissance'].first()
    mean_birth=pd.DataFrame(mean_birth).reset_index()
    na_mask_date2 = mean_birth['Client - Date de Naissance']!='na'
    mean_birth_masked = mean_birth[na_mask_date2]
    mean_birth_masked['Client - Date de Naissance'] = pd.to_datetime(mean_birth_masked['Client - Date de Naissance'])

    mean_birth_masked['age']=mean_birth_masked['Client - Date de Naissance'].dt.year
    annee_naissance_moyenne = round(mean_birth_masked['age'].mean())
    data_cleaned['Client - Date de Naissance'] = data_cleaned['Client - Date de Naissance'].replace('na', pd.to_datetime(f'01/01/{str(int(annee_naissance_moyenne))}'))

    #Idem pour civilité
    na_mask_civ = data_cleaned['Client - Civilité']=='na'
    prop_civ=data_cleaned.groupby('Client - ID')['Client - Civilité'].first()
    prop_civ=pd.DataFrame(prop_civ).reset_index()
    proportion_mme=round(len(prop_civ[prop_civ['Client - Civilité']=='Mme'])/(len(prop_civ[prop_civ['Client - Civilité']=='Mr'])+(len(prop_civ[prop_civ['Client - Civilité']=='Mme']))),2)
    data_cleaned['Client - Civilité'] = np.where(data_cleaned['Client - Civilité']=='na', np.random.choice(['Mme', 'Mr'], p=[proportion_mme, 1-proportion_mme]), data_cleaned['Client - Civilité'])

    #Supression des lignes concernant un client qui semble "faux", surement utilisé pour faire des tests par l'entreprise
    #.index sert a suppr toute la ligne et pas seulement la case
    data_cleaned = data_cleaned.drop(data_cleaned[data_cleaned['Client - ID'] == 'CLT99058'].index)

    #Raccourcir nom cette colonne
    data_cleaned = data_cleaned.rename(columns={'Client - Optin (Oui/Non)': 'Client - Optin'})

    #Conversion en date fr puis en date US
    data_cleaned['DAY'] = pd.to_datetime(data_cleaned['DAY'],format='%d/%m/%Y')
    data_cleaned['DAY'] = data_cleaned['DAY'].dt.strftime('%m/%d/%Y')
    data_cleaned['DAY'] = pd.to_datetime(data_cleaned['DAY'],format='%m/%d/%Y')

    #Passage en float du CA
    data_cleaned['CA Produits HT'] = data_cleaned['CA Produits HT'].str.replace(',', '.').astype(float)

    # Convertir la colonne en type int puis str
    data_cleaned['Client - Mois de Création'] = data_cleaned['Client - Mois de Création'].astype('int64')
    data_cleaned['Client - Mois de Création'] = data_cleaned['Client - Mois de Création'].astype(str)
    # Convertir la colonne en type datetime avec le format '%Y%m'
    data_cleaned['Client - Mois de Création'] = pd.to_datetime(data_cleaned['Client - Mois de Création'], format='%Y%m')

    # Convertir la colonne en type int puis str
    data_cleaned['Client - Mois de la première commande'] = data_cleaned['Client - Mois de la première commande'].astype('int64')
    data_cleaned['Client - Mois de la première commande'] = data_cleaned['Client - Mois de la première commande'].astype(str)
    # Convertir la colonne en type datetime avec le format '%Y%m'
    data_cleaned['Client - Mois de la première commande'] = pd.to_datetime(data_cleaned['Client - Mois de la première commande'], format='%Y%m')

    # Convertir la colonne en type int puis str
    data_cleaned['Client - Mois de la dernière commande'] = data_cleaned['Client - Mois de la dernière commande'].astype('int64')
    data_cleaned['Client - Mois de la dernière commande'] = data_cleaned['Client - Mois de la dernière commande'].astype(str)
    # Convertir la colonne en type datetime avec le format '%Y%m'
    data_cleaned['Client - Mois de la dernière commande'] = pd.to_datetime(data_cleaned['Client - Mois de la dernière commande'], format='%Y%m')

    data_cleaned['Client - Optin'] = data_cleaned['Client - Optin'].astype('int64')
    data_cleaned['N Commandes'] = data_cleaned['N Commandes'].astype('int64')
    data_cleaned['Commande - ID Transaction'] = data_cleaned['Commande - ID Transaction'].astype('int64')

    data_cleaned['Client - Date de Naissance'] = pd.to_datetime(data_cleaned['Client - Date de Naissance'])


    return data_cleaned
