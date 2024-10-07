from fv_generator import *
from eegBandSeparator import *
from preElaboration import *
import os
import pandas as pd
import numpy as np


dir_path = "C:\\Users\\aless\OneDrive - Università di Cagliari\\ML\\dataset tesi\\dataset_ruota\\dataset_elaborati_ruota_standard"
df = pd.DataFrame()


count = 1

for path in os.scandir(dir_path):

    df = pd.read_csv(path)

    #eseguo la pre elaborazione
    #dataset_preelaborato = preElaborationRuota(df, normalize_method= "l2")

    #dataset_preelaborato.dropna(inplace=True)
    #creazione feature vector
    feature_vector = fv_gen_regression(df,sw=0.8)

    #cancello le righe con la label 0
    #feature_vector.drop(feature_vector[feature_vector[140] == 0].index,inplace=True)
    feature_vector.to_csv("C:\\Users\\aless\\OneDrive - Università di Cagliari\ML\\feature vector\\fv_ruota\\fv_08s_standard\\fv"+str(count)+'.csv', index = False) 

    #dataset_preelaborato.to_csv("C:\\Users\\aless\\OneDrive - Università di Cagliari\\ML\\dataset tesi\\dataset_ruota\\dataset_elaborati_ruota_l2\\"+str(count)+".csv",index = False)

    count += 1




