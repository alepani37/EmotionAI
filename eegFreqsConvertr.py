
import pandas as pd
import numpy as np
import math

'''
Questa funzione converte i dati da una frequenza f1 a una frequenza f2
    f1: frequenza di campionamento dei dati
    f2: frequenza di campionamento desiderata

    return: dataframe con i dati convertiti
'''
def freqsConverter(f1,f2,data):
    print('converting from ' + str(f1) + ' Hz to ' + str(f2) + ' Hz started')
    new_data = pd.DataFrame()
    N = len(data)  # numero di campioni
    tmp_n = math.ceil(f1/f2)
    new_data = data.rolling(tmp_n,step=tmp_n).mean()
    print('conversion from ' + str(f1) + ' Hz to ' + str(f2) + ' Hz done')
    new_data = new_data.reset_index(drop=True)
    return new_data


'''
Questa funzione aggiunge una label alla prima metà dei dati e una label alla seconda metà dei dati
    data: dataframe a cui aggiungere le label

    return: dataframe con le label aggiunte
'''
def addLabels(data):
    print('adding labels started')
    new_data = pd.DataFrame()
    
    N = len(data)  # numero di campioni

    new_data = data

    new_data = new_data.assign(label = np.where(new_data.index < N/2, 'attentive', 'distracted'))
    new_data = new_data.reset_index(drop=True)


    print('adding labels done')
    return new_data

   
