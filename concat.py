import pandas as pd
import random

import os

def concatDataframe(dir_path, onlyOne = False):
    numFile = len(os.listdir(dir_path))

    rand = random.randint(0,numFile)

    count = 0

    print("Concatenating dataframes...")
    for path in os.scandir(dir_path):
        if count == 0:
            dfPrec = pd.read_csv(path)
        if count != rand:
            this = pd.read_csv(path)
            df = pd.concat([dfPrec,this],ignore_index=True)
            dfPrec = df
        if count == rand:
            dfRet = pd.read_csv(path)
        count = count + 1
        perc = (count/numFile)*100
        print(str(perc)+"%")
    print("concatenation done")
    if onlyOne:
        df_total = pd.concat([df,dfRet],ignore_index=True)
        return df_total
    return df,dfRet

