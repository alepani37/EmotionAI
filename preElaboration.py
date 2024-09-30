import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp


#campi del df
#TimeStamp,
#Delta_TP9,Delta_AF7,Delta_AF8,Delta_TP10,Theta_TP9,Theta_AF7,Theta_AF8,Theta_TP10,Alpha_TP9,Alpha_AF7,Alpha_AF8,Alpha_TP10,Beta_TP9,Beta_AF7,Beta_AF8,Beta_TP10,Gamma_TP9,Gamma_AF7,Gamma_AF8,Gamma_TP10,
#RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10,
#Coinvolgimento,Prob_Positivita,label,emozione,sin_ruota,cos_ruota,r_ruota

#df_raw: dataframe da elaborare
#log: se True, stampa i log (default False)
#normalize_method: metodo di normalizzazione da utilizzare (default 'l1')
#return: dataframe elaborato
def preElaboration(df_raw, log = False , normalize_method = 'l1'):

    #elimino le classi non utilizzate
    df_raw.drop(["Coinvolgimento","Prob_Positivita","emozione","sin_ruota","cos_ruota","r_ruota"], axis=1, inplace=True)
    if log:
        print("Dropped labels not needed")

    #elimino le prime 20 colonne (le onde celebrali Theta, Alpha, Beta, Delta, Gamma restituite dal Mind Monitor)
    df_raw.drop(df_raw.iloc[:, 0:21], inplace=True, axis=1)
    if log:
        print("Dropped first 20 columns")

    

    #separo i segnali raccolti dalla classe
    df_x = df_raw.iloc[:, 0:4]
    df_y = df_raw.iloc[:, 4:]

    # elimino le righe in cui mancano i dati raccolti da uno o più elettrodi
    df_raw.dropna(inplace=True)

    if log:
        print("df_raw size: "+str(df_raw.shape))
        print("df_x size: "+str(df_x.shape))
        print("df_y size: "+str(df_y.shape))

    #normalizzazione
    df = normalize(df_x,normalize_method)
    if log:
        print("Normalized")
    
    #converti le etichette in numeri
    for col in df_y.columns:
        df_y[col] = df_y[col].apply(lambda x: label_to_number(x))

    df = pd.concat([df, df_y], axis=1)

    if log:
        print("preelaboration done")

    return df


#df: dataframe da normalizzare
#method: metodo di normalizzazione da utilizzare: l1, l2, mean, minmax, standard
#return: dataframe normalizzato
def normalize(df,method):
    if method == 'l1':
        df_normalized = pd.DataFrame(pp.normalize(df, norm='l1'))
        return df_normalized
    elif method == 'l2':
        df_normalized = pd.DataFrame(pp.normalize(df, norm='l2'))
        return df_normalized
    elif method == "mean":
        df_norm = (df - df.mean()) / (df.max() - df.min())
        return df_norm
    elif method == "minmax":
        scaler = pp.MinMaxScaler()
        df_norm = pd.DataFrame(scaler.fit_transform(df))
        return df_norm
    elif method == "standard":
        scaler = pp.StandardScaler()
        df_norm = pd.DataFrame(scaler.fit_transform(df))
        return df_norm
    else:
        print("Error: method not found")
        raise Exception("Error: method not found")


#converte le label da stringhe a numeri
def label_to_number(label):
    if label == "neutre":
        return 0
    elif label == "positive":
        return 1
    else:
        return -1

def preElaborationRuota(df_raw, log = False , normalize_method = 'l1'):

    #elimino le classi non utilizzate
    df_raw.drop(["label","Coinvolgimento","Prob_Positivita"], axis=1, inplace=True)
    if log:
        print("Dropped labels not needed")

    #elimino le prime 20 colonne (le onde celebrali Theta, Alpha, Beta, Delta, Gamma restituite dal Mind Monitor)
    df_raw.drop(df_raw.iloc[:, 0:21], inplace=True, axis=1)
    if log:
        print("Dropped first 20 columns")

    #separo i segnali raccolti dalla classe
    df_x = df_raw.iloc[:, 0:4]
    df_label = df_raw.iloc[:, 4:5]
    df_ruota = df_raw.iloc[:, 5:]

    #df_angolo = pd.DataFrame()
    #calcolo l'angolo della ruota
    #df_angolo["angolo"] = df_ruota.apply(lambda row: np.arctan2(row["sin_ruota"],row["cos_ruota"]), axis=1)
    df_coordinate = pd.DataFrame()
    df_coordinate["x"] = df_ruota.apply(lambda row: row["r_ruota"]*row["cos_ruota"], axis=1)
    df_coordinate["y"] = df_ruota.apply(lambda row: row["r_ruota"]*row["sin_ruota"], axis=1)


    # elimino le righe in cui mancano i dati raccolti da uno o più elettrodi
    df_raw.dropna(inplace=True)

    

    #normalizzazione
    df = normalize(df_x,normalize_method)
    if log:
        print("Normalized")

    df = pd.concat([df, df_label], axis=1)
    df = pd.concat([df, df_coordinate], axis=1)

    if log:
        print("preelaboration done")

    return df