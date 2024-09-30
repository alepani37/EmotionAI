import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor,XGBRFRegressor
import warnings
from postElaboration import postElaborationRegression

warnings.filterwarnings("ignore")

path_feature_vector = "C:\\Users\\luca-\\OneDrive\\Data\\Uni\\Tirocinio\\Datasets\\emozioni"
name_file = "results-all-subjects-reg2.txt"

#letture dei file contenenti i feature vector
array_feature_vector = []
for path in os.scandir(path_feature_vector):
    df = pd.read_csv(path)
    #drop valori NaN
    df = df.dropna()
    array_feature_vector.append(df)


preds = [0 for i in range(len(array_feature_vector))]
preds_post = ['' for i in range(len(array_feature_vector))]

r2_score_list = [0 for i in range(len(array_feature_vector))]
mean_absolute_error_list = [0 for i in range(len(array_feature_vector))]
mean_squared_error_list = [0 for i in range(len(array_feature_vector))]
median_absolute_error_list = [0 for i in range(len(array_feature_vector))]

accuracy_list = [0 for i in range(len(array_feature_vector))]
precision_list = [0 for i in range(len(array_feature_vector))]
recall_list = [0 for i in range(len(array_feature_vector))]
f1_list = [0 for i in range(len(array_feature_vector))]

#cross validation
for i in range(len(array_feature_vector)):
    print("subject " + str(i))

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    df_test = array_feature_vector[i]
    for j in range(len(array_feature_vector)):
        if i!=j:
            df_train = pd.concat([df_train,array_feature_vector[j]],ignore_index=True)

    #divisione dei dati dalla classe
    X_train = df_train.iloc[:,:-2]
    y_train = df_train.iloc[:,-1]

    X_test = df_test.iloc[:,:-2]
    label_test = df_test.iloc[:,-2]
    y_test = df_test.iloc[:,-1]

    #sostituzione dei valori NaN con la media della colonna per evitare errori, si può provare a eliminare le righe con valori NaN
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    #oversampling: applicazione di SMOTE per bilanciare le classi generando nuove istanze della classe minoritaria
    #sm = SMOTE(sampling_strategy='minority',n_jobs=-1)
    #X_train, y_train = sm.fit_resample(X_train, y_train)


    #addestramento modello
    

    model = XGBRegressor(
            max_depth=3,
            gamma = 2,
            eta = 0.8,
            reg_alpha = 0.5,
            reg_lambda = 0.5,
            n_estimators=100)
    
    model.fit(X_train,y_train)
    preds[i] = model.predict(X_test)

        
    #salvataggio delle metriche di regressione
    r2_score_list[i] = r2_score(y_test, preds[i])
    mean_absolute_error_list[i] = mean_absolute_error(y_test, preds[i])
    mean_squared_error_list[i] = mean_squared_error(y_test,preds[i])
    median_absolute_error_list[i] = median_absolute_error(y_test, preds[i])


    stampa = pd.DataFrame()
    stampa['angolo_predetto'] = preds[i]
    stampa['angolo_reale'] = y_test

    preds_post[i] = postElaborationRegression(preds[i],75)
    
    stampa['label_predetta'] = preds_post[i]
    stampa['label_reale'] = label_test

    stampa.to_csv(name_file + str(i) + ".csv")

    

    #salvataggio delle metriche di classificazione
    accuracy_list[i] = accuracy_score(label_test,preds_post[i])
    precision_list[i] = precision_score(label_test,preds_post[i],average='macro')
    recall_list[i] = recall_score(label_test,preds_post[i],average='macro')
    f1_list[i] = f1_score(label_test,preds_post[i],average='macro')





    # salvataggio su file della media delle metriche
    with open(name_file, 'a') as text_file:
        print("metrics subject " + str(i) + ":\n", file=text_file)
        print("metriche regressiione angolo:", file=text_file)
        print("r2: " + str(r2_score_list[i]), file=text_file)
        print("mean absolute error: " + str(mean_absolute_error_list[i]), file=text_file)
        print("mean squared error: " + str(mean_squared_error_list[i]), file=text_file)
        print("median absolute error: " + str(median_absolute_error_list[i]), file=text_file)
        print("\n",file=text_file)
        print("metriche classificazione emozione:", file=text_file)
        print("accuracy: " + str(accuracy_list[i]), file=text_file)
        print("precision: " + str(precision_list[i]), file=text_file)
        print("recall: " + str(recall_list[i]), file=text_file)
        print("f1: " + str(f1_list[i]), file=text_file)
        print("\n",file=text_file)
        print("confusion matrix:\n" + str(confusion_matrix(label_test,preds_post[i])), file=text_file)

#salvataggio su file della media delle metriche
with open(name_file,'a') as text_file:
    print("avg r2: "+str(np.mean(r2_score_list)),file=text_file)
    print("avg MAE: "+str(np.mean(mean_absolute_error_list)),file=text_file)
    print("avg MSE: "+str(np.mean(mean_squared_error_list)),file=text_file)
    print("avg MedAE: "+str(np.mean(median_absolute_error_list)),file=text_file)
    print("\n",file=text_file)
    print("avg accuracy: "+str(np.mean(accuracy_list)),file=text_file)
    print("avg precision: "+str(np.mean(precision_list)),file=text_file)
    print("avg recall: "+str(np.mean(recall_list)),file=text_file)
    print("avg f1: "+str(np.mean(f1_list)),file=text_file)
    


risposta = input("salvare il modello? (y/n)")
if risposta == "y":
    nome = input("inserisci il nome del modello:")

    import pickle
    pathModello = nome + ".sav"
    pickle.dump(model, open(pathModello, 'wb'))
    print("model saved")

