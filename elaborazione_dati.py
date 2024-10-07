from fv_generator import *
from eegBandSeparator import *
from preElaboration import *
import os
import pandas as pd
import numpy as np


dir_path = "path dati"
df = pd.DataFrame()


count = 1

#pre elaborazione
for path in os.scandir(dir_path):

    df = pd.read_csv(path)

    #eseguo la pre elaborazione
    dataset_preelaborato = preElaborationRuota(df, normalize_method= "l2")
    dataset_preelaborato.dropna(inplace=True)
    dataset_preelaborato.to_csv("path dati elaborati"+str(count)+".csv",index = False)

    count += 1

count = 1

dir_path = "path dati elaborati"
for path in os.scandir(dir_path):

    df = pd.read_csv(path)
    feature_vector = fv_gen_regression(df,sw=0.8)

    #cancello le righe con la label 0
    feature_vector.drop(feature_vector[feature_vector[140] == 0].index,inplace=True)
    feature_vector.to_csv("path salvataggio fv"+str(count)+'.csv', index = False) 

    count += 1



