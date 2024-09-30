import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
import warnings
from mlxtend.classifier import StackingClassifier
from postElaboration import postElaboration

warnings.filterwarnings("ignore")

path_feature_vector = "C:\\Users\\luca-\\OneDrive\\Data\\Uni\\Tirocinio\\Datasets\\emozioni"
name_file = "results-all-subjects-post-multi18.txt"

#letture dei file contenenti i feature vector
array_feature_vector = []
for path in os.scandir(path_feature_vector):
    df = pd.read_csv(path)
    #drop valori NaN
    df = df.dropna()
    array_feature_vector.append(df)

    
    #stampa del numero di istanze appartenenti alla classe positive e negative per ogni soggetto
    count_positive = len(df[df.iloc[:, -1] != 1])
    count_negative = len(df[df.iloc[:, -1] != -1])
    print(str(path) + ' positive class: ' + str(count_positive) + ', negative class: ' + str(count_negative))

    #sostiusco la label -1 con 0 per renderlo compatibile con xgboost
    df.iloc[:, -1] = df.iloc[:, -1].replace(-1, 0)


preds = [0 for i in range(len(array_feature_vector))]

accuracy = [0 for i in range(len(array_feature_vector))]
precision = [0 for i in range(len(array_feature_vector))]
recall = [0 for i in range(len(array_feature_vector))]
f1 = [0 for i in range(len(array_feature_vector))]

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
    X_train = df_train.iloc[:,:-1]
    y_train = df_train.iloc[:,-1]

    X_test = df_test.iloc[:,:-1]
    y_test = df_test.iloc[:,-1]

    #sostituzione dei valori NaN con la media della colonna per evitare errori, si può provare a eliminare le righe con valori NaN
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    #oversampling: applicazione di SMOTE per bilanciare le classi generando nuove istanze della classe minoritaria
    #sm = SMOTE(sampling_strategy='minority',n_jobs=-1)
    #X_train, y_train = sm.fit_resample(X_train, y_train)


    #addestramento modello
    #classifiers: i modelli da utilizzare separati da ,
    #meta_classifier: il modello che combina i modelli precedenti
    
    model = StackingClassifier(
        classifiers=[
        XGBRFClassifier(
            max_depth=3,
            gamma = 2,
            eta = 0.8,
            reg_alpha = 0.5,
            reg_lambda = 0.5,
            n_estimators=300
        ),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='auto'
        ),
        LogisticRegression(
        ),
        
        
        ],
        use_probas=True,
        meta_classifier= XGBClassifier(
            max_depth=3,
            gamma = 2,
            eta = 0.8,
            reg_alpha = 0.5,
            reg_lambda = 0.5            
        ))
       

    model.fit(X_train, y_train)
    preds[i] = model.predict(X_test)
    preds[i] = postElaboration(preds[i],sw = 150)

    #salvataggio delle metriche
    accuracy[i] = accuracy_score(y_test, preds[i])
    precision[i] = precision_score(y_test, preds[i], average='weighted')
    recall[i] = recall_score(y_test,preds[i], average='weighted')
    f1[i] = f1_score(y_test, preds[i], average='weighted')

    # salvataggio su file della media delle metriche
    with open(name_file, 'a') as text_file:
        print("metrics subject " + str(i) + ":\n", file=text_file)
        print("Test set: " + str(path), file=text_file)
        print("accuracy: " + str(accuracy[i]), file=text_file)
        print("precision: " + str(precision[i]), file=text_file)
        print("recall: " + str(recall[i]), file=text_file)
        print("f1 score: " + str(f1[i]), file=text_file)
        print("\n",file=text_file)

#salvataggio su file della media delle metriche
with open(name_file,'a') as text_file:
    print("avg accuracy: "+str(np.mean(accuracy)),file=text_file)
    print("avg precision: "+str(np.mean(precision)),file=text_file)
    print("avg recall: "+str(np.mean(recall)),file=text_file)
    print("avg f1 score: "+str(np.mean(f1)),file=text_file)


risposta = input("salvare il modello? (y/n)")
if risposta == "y":
    nome = input("inserisci il nome del modello:")

    import pickle
    pathModello = nome + ".sav"
    pickle.dump(model, open(pathModello, 'wb'))
    print("model saved")

