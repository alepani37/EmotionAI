import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
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
import pickle

FOLDS = 5
path_feature_vector = "C:\\Users\\aless\\OneDrive - Università di Cagliari\\ML\\feature vector\\fv_05s_no_standard"
name_file = "results-one-subject-post.txt"


#letture dei file contenenti i feature vector
array_feature_vector = []
for path in os.scandir(path_feature_vector):
    df = pd.read_csv(path)
    df.iloc[:, -1] = df.iloc[:, -1].replace(-1, 0)
    #drop valori NaN
    df = df.dropna()
    array_feature_vector.append(df)

accuracy_avg = [0 for i in range(len(array_feature_vector))]
precision_avg = [0 for i in range(len(array_feature_vector))]
recall_avg = [0 for i in range(len(array_feature_vector))]
f1_avg = [0 for i in range(len(array_feature_vector))]

#eseguiamo la cross validation sui diversi soggetti separatamente
for i in range(len(array_feature_vector)):

    preds = [0 for j in range(FOLDS)]

    accuracy = [0 for j in range(FOLDS)]
    precision = [0 for j in range(FOLDS)]
    recall = [0 for j in range(FOLDS)]
    f1 = [0 for j in range(FOLDS)]


    #divido i feature vectors delle classi in sottoinsiemi (folds) di uguale dimensione
    count_positive = len(array_feature_vector[i][array_feature_vector[i].iloc[:, -1] != 1])
    positive_fv = array_feature_vector[i][array_feature_vector[i].iloc[:, -1] != 1]
    list_positive_folds = [positive_fv[z:z + round(count_positive/FOLDS)] for z in range(0, len(positive_fv), round(count_positive/FOLDS))]

    count_negative = len(array_feature_vector[i][array_feature_vector[i].iloc[:, -1] != 0])
    negative_fv = array_feature_vector[i][array_feature_vector[i].iloc[:, -1] != 0]
    list_negative_folds = [negative_fv[z:z + round(count_negative/FOLDS)] for z in range(0, len(negative_fv), round(count_negative/FOLDS))]

    for f in range(FOLDS):
        train = pd.DataFrame()
        test = pd.DataFrame()

        test = pd.concat([test,list_negative_folds[f], list_positive_folds[f]],ignore_index=True)
        for g in range(FOLDS):
            if f!=g:
                train = pd.concat([train,list_negative_folds[g], list_positive_folds[g]],ignore_index=True)

        #divisione dei dati dalla classe
        X_train = train.iloc[:,:-1]
        y_train = train.iloc[:,-1]

        X_test = test.iloc[:,:-1]
        y_test = test.iloc[:,-1]

        #addestramento modello
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
            max_features='auto',
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
        preds[f] = model.predict(X_test)
        preds[f] = postElaboration(preds[f],75)

        #salvataggio delle metriche
        accuracy[f] = accuracy_score(y_test, preds[f])
        precision[f] = precision_score(y_test, preds[f], average='weighted')
        recall[f] = recall_score(y_test,preds[f], average='weighted')
        f1[f] = f1_score(y_test, preds[f], average='weighted')

        # salvataggio su file della media delle metriche
        with open(name_file, 'a') as text_file:
            print("metrics subject " + str(i) + " fold " + str(f) + ":", file=text_file)
            print("accuracy: " + str(accuracy[f]), file=text_file)
            print("precision: " + str(precision[f]), file=text_file)
            print("recall: " + str(recall[f]), file=text_file)
            print("f1 score: " + str(f1[f]), file=text_file)
            print("\n",file=text_file)

    #salvataggio su file della media delle metriche
    with open(name_file,'a') as text_file:
        print("avg accuracy: "+str(np.mean(accuracy)),file=text_file)
        accuracy_avg[i] = np.mean(accuracy)
        print("avg precision: "+str(np.mean(precision)),file=text_file)
        precision_avg[i] = np.mean(precision)
        print("avg recall: "+str(np.mean(recall)),file=text_file)
        recall_avg[i] = np.mean(recall)
        print("avg f1 score: "+str(np.mean(f1)),file=text_file)
        f1_avg[i] = np.mean(f1)
        print("\n", file=text_file)

with open(name_file,'a') as text_file:
    print("All subjects metrics:",file=text_file)
    print("avg accuracy: "+str(np.mean(accuracy_avg)),file=text_file)
    print("avg precision: "+str(np.mean(precision_avg)),file=text_file)
    print("avg recall: "+str(np.mean(recall_avg)),file=text_file)
    print("avg f1 score: "+str(np.mean(f1_avg)),file=text_file)

