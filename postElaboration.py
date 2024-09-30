
import numpy as np
import math as m
def postElaboration(preds,sw = 1):
    preds_post = np.zeros(len(preds))
    i = 0
    while i+ sw < len(preds):
        sw_tmp = np.array(preds[i:i+sw-1])
        preds_post[i:i+sw-1] = label_piu_freq(sw_tmp)
        i = i + sw
    #ultimi valori
    sw_tmp = np.array(preds[i:])
    preds_post[i:] = label_piu_freq(sw_tmp)
    return preds_post


def postElaborationRegression(preds,sw = 1):

    #convertiamo da angoli a emozioni
    preds_post = ['' for i in range(len(preds))]

   
    

    for i in range(len(preds)):
        
        x = preds[i][0]
        y = preds[i][1]

        #conversione da x,y ad angolo
        sin = y/m.sqrt(x**2 + y**2)
        cos = x/m.sqrt(x**2 + y**2)
        angle = m.atan2(sin,cos)*180/m.pi

        #conversione angoli negativi
        if angle < 0:
            angle = angle + 360

        #conversione da angolo a emozione
        if angle < 0:
            angle = angle + 360
        if angle >= 22.5 and angle <= 67.5:
            preds_post[i] = 'fiducia'
        elif angle > 67.5 and angle <= 112.5:
            preds_post[i] = 'gioia'
        elif angle > 112.5 and angle <= 157.5:
            preds_post[i] = 'interesse'
        elif angle > 157.5 and angle <= 202.5:
            preds_post[i] = 'rabbia'
        elif angle > 202.5 and angle <= 247.5:
            preds_post[i] = 'noia'
        elif angle > 247.5 and angle <= 292.5:
            preds_post[i] = 'tristezza'
        elif angle > 292.5 and angle <= 337.5:
            preds_post[i] = 'sorpresa'
        else:
            preds_post[i] = 'paura'

    #rimozione errori in una finestra
    #res = postElaboration(preds_post,sw)
    #return res
    return preds_post  

def label_piu_freq(array):
    uniche, conteggi = np.unique(array, return_counts=True)
    
    # Trova l'indice dell'etichetta più frequente
    indice_max = np.argmax(conteggi)
    
    # Restituisci l'etichetta più frequente come stringa
    label_piu_frequente = uniche[indice_max]

    return label_piu_frequente