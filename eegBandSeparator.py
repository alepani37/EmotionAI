import pandas as pd
from scipy.fft import fft,fftfreq
import numpy as np
import scipy.signal as signal

'''
data: segnale da filtrare
fs: frequenza di campionamento
f1: frequenza minima
f2: frequenza massima
return: segnale filtrato
'''
def band_filter(data,fs,f1,f2):
    N = len(data)  # numero di campioni
    freqs = fftfreq(N, 1/fs)
    data = data.to_numpy()

    # Calcola la trasformata di Fourier del segnale
    X = fft(data)

    # Filtra il segnale
    mask = np.logical_and(freqs >= f1, freqs <= f2)
    index_firts_True = np.where(mask == True)[0][0]
    index_last_True = np.where(mask == True)[0][-1]

    risultato = X[index_firts_True:index_last_True+1]
    return risultato

'''
fs: frequenza di campionamento del segnale
data: dataframe contenente il segnale
bw: onda celebrale
return: dataframe con i segnali filtrati
'''
def eeg_separator(fs,data,bw):
    if bw == 'delta':
        f1 = 0.5
        f2 = 4
    elif bw == 'theta':
        f1 = 4
        f2 = 8
    elif bw == 'alpha':
        f1 = 8
        f2 = 13
    elif bw == 'beta':
        f1 = 13
        f2 = 30
    elif bw == 'gamma':
        f1 = 30
        f2 = 44

    brainwave = band_filter(data, fs, f1, f2)

    return brainwave.astype('float64')