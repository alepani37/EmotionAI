from eegBandSeparator import *

'''
data: dataframe da cui estrarre le features
fs: frequenza di campionamento
sw: sliding window in secondi (opzionale, default 1)
shift: percentuale di scorrimento (opzionale, default 1)
return: dataframe con le features estratte
'''
def fv_gen(data, fs=256, sw=1, shift=1):
    sw_samples = int(fs * sw)
    shift_samples = int(fs * shift)
    brainwaves = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    print("Start feature extraction")
    df_fv = pd.DataFrame()

    #estraiamo le sw da ogni canale
    df_sw = []
    for col in data.columns:
        df_sw.append(sliding_windows(data[col], sw_samples, shift_samples))

    shape = df_sw[0].shape
    for sw_id in range(0,shape[1]):
        #verifichiamo che la sw contenga campioni provenienti da una sola classe
        uniche = np.unique(df_sw[-1].iloc[:, sw_id])
        if len(uniche) == 1:
            classe = uniche[0]
            #analizziamo le sw raccolte nello stesso momento dai diversi canali
            fv = []
            for col in range(0,4):
                raw_signal = df_sw[col].iloc[:, sw_id]
                #convertiamo ogni sw utilizzando la FFT
                for bw in brainwaves:
                    bw_signal = eeg_separator(fs, raw_signal, bw)
                    fv.append(np.mean(bw_signal))
                    fv.append(np.median(bw_signal))
                    fv.append(np.var(bw_signal))
                    fv.append(np.std(bw_signal))
                    fv.append(np.min(bw_signal))
                    fv.append(np.max(bw_signal))
                    fv.append(np.max(bw_signal)-np.min(bw_signal))
                    fv.append(first_order_diff(bw_signal))
                    fv.append(second_order_diff_avg_abs(bw_signal))
            fv.append(int(classe))
            df_fv = pd.concat([df_fv, pd.DataFrame(fv).T], ignore_index=True)
    return df_fv

def fv_gen_regression(data, fs=256, sw=1, shift=1):
    sw_samples = int(fs * sw)
    shift_samples = int(fs * shift)
    brainwaves = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    print("Start feature extraction")
    df_fv = pd.DataFrame()

    #estraiamo le sw da ogni canale
    df_sw = []
    for col in data.columns:
        df_sw.append(sliding_windows(data[col], sw_samples, shift_samples))

    shape = df_sw[0].shape
    for sw_id in range(0,shape[1]):
        
            #analizziamo le sw raccolte nello stesso momento dai diversi canali
            fv = []
            for col in range(0,4):
                raw_signal = df_sw[col].iloc[:, sw_id]
                #convertiamo ogni sw utilizzando la FFT
                for bw in brainwaves:
                    bw_signal = eeg_separator(fs, raw_signal, bw)
                    fv.append(np.mean(bw_signal))
                    fv.append(np.median(bw_signal))
                    fv.append(np.var(bw_signal))
                    fv.append(np.std(bw_signal))
                    fv.append(np.min(bw_signal))
                    fv.append(np.max(bw_signal))
                    fv.append(np.max(bw_signal)-np.min(bw_signal))
                    fv.append(first_order_diff(bw_signal))
                    fv.append(second_order_diff_avg_abs(bw_signal))
            fv.append(np.max(df_sw[-3].iloc[:, sw_id]))
            fv.append(np.max(df_sw[-2].iloc[:, sw_id]))
            fv.append(np.max(df_sw[-1].iloc[:, sw_id]))
            df_fv = pd.concat([df_fv, pd.DataFrame(fv).T], ignore_index=True)
    return df_fv

'''
signal: segnale registrato da un singolo canale
sw_size = grandezza della finestra scorrevole

shift: di quanti campioni mandare avanti la finestra scorrevole
#return: dataframe in cui ogni colonna corrisponde ad una finestra scorrevole
'''
def sliding_windows(signal, sw_size, shift):
    signal = list(signal)
    sw_result = pd.DataFrame()
    size = len(signal)
    for z in range(0, size, shift):
        if z + sw_size <= size:
            x = pd.DataFrame(signal[z:z + sw_size])
            sw_result = pd.concat([sw_result, x], axis=1)
    return sw_result

def first_order_diff(signal):
    n = len(signal)
    mad_values = [0] * n
    
    for i in range(1, n):
        mad_values[i] = abs(signal[i] - signal[i-1])
    
    return sum(mad_values) / n

def second_order_diff_avg_abs(signal):
    diff = np.diff(signal, n=2)  # Compute the second-order differences
    avg_abs = np.mean(np.abs(diff))  # Compute the average absolute value
    return avg_abs



