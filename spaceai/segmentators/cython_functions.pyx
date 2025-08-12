# compute_spectral_centroid.pyx
from scipy.stats import kurtosis, skew
import numpy as np
cimport numpy as np
cimport numpy as cnp
from libc.math cimport fabs
import pandas as pd
from libc.stdlib cimport malloc, free
# Importa la classe RocketExtracted e il dizionario transformations dal modulo Python
from cython.parallel import prange
import itertools
import random
from libc.math cimport sqrt

from scipy.signal import stft

cpdef stft_spectral_std(cnp.ndarray[double, ndim=1] values, int nperseg=10, double epsilon=1e-6):
    """
    Calcola la deviazione standard dello spettrogramma ottenuto tramite STFT,
    normalizzata rispetto alla media del segnale.
    """
    cdef int n = values.shape[0]
    if n < 2:
        return 0.0
    
    # Calcolo STFT
    _, _, Zxx = stft(values, nperseg=nperseg)
    
    # Calcoliamo il modulo dello spettrogramma
    spectrogram_magnitude = np.abs(Zxx)
    
    # Sommiamo tutte le componenti di frequenza per ciascun istante di tempo
    summed_spectrum = np.sum(spectrogram_magnitude, axis=0)
    
    # Calcoliamo la deviazione standard
    cdef double std_spectrum = np.std(summed_spectrum)
    
    # Normalizziamo rispetto alla media del segnale
    cdef double mean_value = np.mean(values)
    
    return std_spectrum / (mean_value + epsilon)


cpdef double moving_average_error(cnp.ndarray[double, ndim=1] data):
    """
    Calcola l'errore di predizione per ogni punto utilizzando un predittore a media mobile.
    Restituisce la media degli errori assoluti.
    """
    cdef int n = data.shape[0]
    cdef double error_sum = 0.0
    cdef int count = 0
    cdef double moving_avg
    cdef int i, j
    cdef int window_size = int(0.1 * len(data))
    for i in range(window_size, n):
        moving_avg = 0.0
        for j in range(i - window_size, i):
            moving_avg += data[j]
        moving_avg /= window_size
        
        error_sum += fabs(moving_avg - data[i])
        count += 1
    
    return error_sum / count if count > 0 else 0.0

cpdef compute_spectral_centroid(np.ndarray[np.float32_t, ndim=1] window, double fs=1.0):
    """
    Calcola il spectral centroid di una finestra di segnale.

    Il spectral centroid rappresenta la frequenza media dello spettro,
    ponderata dalle ampiezze delle componenti spettrali.

    Parametri:
      window : np.ndarray[np.double_t, ndim=1]
          Array monodimensionale contenente i campioni del segnale.
      fs : double, opzionale
          Frequenza di campionamento del segnale (default 1.0).

    Ritorna:
      spectral_centroid : double
          Il spectral centroid in Hz.
    """
    # Calcola la FFT della finestra (usando rFFT per segnali reali)
    cdef np.ndarray fft_vals = np.fft.rfft(window)
    cdef np.ndarray magnitudes = np.abs(fft_vals)
    # Calcola le frequenze associate ai coefficienti della rFFT
    cdef np.ndarray freqs = np.fft.rfftfreq(window.shape[0], d=1.0/fs)
    
    cdef double sum_magnitudes = np.sum(magnitudes)
    if sum_magnitudes == 0:
        return 0.0
    
    cdef double spectral_centroid = np.sum(freqs * magnitudes) / sum_magnitudes
    return spectral_centroid


cpdef double calculate_slope(np.ndarray[np.double_t, ndim=1] values, int sample_size=300):
    cdef int n = values.shape[0]
    cdef int i
    # Se il numero di campioni supera sample_size, esegui un sampling casuale (usando np.random.choice)
    # Qui manteniamo il richiamo a np.random.choice, che è una funzione Python, poiché implementare un sampling efficiente in C richiede più lavoro.
    if n > sample_size:
        values = np.random.choice(values, size=sample_size, replace=False)
        n = values.shape[0]
    
    cdef double sum_y = 0.0, sum_xy = 0.0
    for i in range(n):
        sum_y += values[i]
        sum_xy += i * values[i]
    
    cdef double sum_x = (n - 1) * n / 2.0
    cdef double sum_x2 = (n - 1) * n * (2 * n - 1) / 6.0
    cdef double numerator = n * sum_xy - sum_x * sum_y
    cdef double denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0
    return numerator / denominator

cpdef double spearman_correlation(np.ndarray[np.double_t, ndim=1] values, int sample_size=300):
    """
    Calcola la correlazione di Spearman tra un array di valori e l'array 0,1,...,n-1.
    Questa implementazione assume che non vi siano tie (valori uguali) in values.
    """
    cdef int n = values.shape[0]
    cdef int i, j
    cdef double rank, d, sum_d2 = 0.0

    # Se il numero di campioni supera sample_size, effettua un sampling casuale
    if n > sample_size:
        values = np.random.choice(values, size=sample_size, replace=False)
        n = values.shape[0]
    
    # Calcola i rank per "values" in maniera naïve (O(n^2)). In assenza di tie, il rank di un elemento è:
    # rank[i] = 1 + (numero di valori minori di values[i])
    cdef np.ndarray[np.double_t, ndim=1] ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        rank = 1.0
        for j in range(n):
            if values[j] < values[i]:
                rank += 1.0
        ranks[i] = rank

    # Poiché l'array x è [1, 2, ..., n] (o 0,1,...,n-1 se preferisci; qui consideriamo 1-indexed per la formula classica)
    # Calcola la somma dei quadrati delle differenze: d_i = rank[i] - (i+1)
    for i in range(n):
        d = ranks[i] - (i + 1)
        sum_d2 += d * d

    # Formula di Spearman: ρ = 1 - (6 * sum_d2) / (n*(n^2 - 1))
    return 1.0 - (6.0 * sum_d2) / (n * (n*n - 1))



cpdef list apply_transformations_to_channel_cython(object self, np.ndarray data, np.ndarray anomaly_indices,
                masks: np.ndarray, bint train=True):
    """
    Applies transformations to segments of the data and labels each segment based on whether it
    intersects an anomaly (label 1). If a segment does not
    intersect any of these intervals, it is labeled as 0.
    
    Args:
        data: 2D numpy array (time x features)
        anomaly_indices: 2D numpy array of shape (n_anomalies, 2) containing (start, end) indices of anomalies
        rare_event_indices: 2D numpy array of shape (n_rare_events, 2) containing (start, end) indices for rare events
    
    Returns:
        A list of segments, where each segment is a list of floats with its first element being the event label.
    """
    from spaceai.segmentators.shapelet_miner import ShapeletMiner
    cdef list results = []
    cdef list segment
    cdef int anomaly_index = 0
    cdef int mask_index = 0
    cdef int index = 0
    cdef int event = 0
    cdef int seg_start, seg_end
    cdef bint intersects_anomaly, intersects_invalid
    cdef int telecommand_idx
    cdef np.ndarray values

    while index + self.segment_duration < len(data):

        seg_start = index
        seg_end = index + self.segment_duration
        
        while mask_index < masks.shape[0] and seg_start > masks[mask_index][1]:
            mask_index += 1

        intersects_invalid = False
        if mask_index < masks.shape[0]:
            if max(seg_start, <int>masks[mask_index][0]) < min(seg_end, <int>masks[mask_index][1]):
                intersects_invalid = True
        else:
            if not train and len(masks)>0:
                break
            
        while anomaly_index < anomaly_indices.shape[0] and seg_start > anomaly_indices[anomaly_index][1]:
            anomaly_index += 1
        # Check for intersection with anomaly
        intersects_anomaly = False
        if anomaly_index < anomaly_indices.shape[0]:
            if max(seg_start, <int>anomaly_indices[anomaly_index][0]) < min(seg_end, <int>anomaly_indices[anomaly_index][1]):
                intersects_anomaly = True

        if train:
            # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            # allenamento: escludi (skip) le finestre invalid e
            # marca le anomalie con event=1, le buone con=0
            #–––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            if intersects_invalid:
                event = -1
            else:
                event = 1 if intersects_anomaly else 0
       
        else:
            # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            # test/challenge: **tieni solo** le finestre invalid
            # e salta tutte le altre
            #–––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            if not intersects_invalid:
                event = -1
            else:
                event = 1 if intersects_anomaly else 0

        segment = [event]

        values = data[seg_start:seg_end, 0]
        # Append transformation features computed on the primary channel (column 0)
        segment.extend([seg_start, seg_end])
        segment.extend([
            self.available_transformations[transformation](values)
            for transformation in self.transformations
        ])

        if self.telecommands:
            for telecommand_idx in range(1, data.shape[1]):
                segment.append(int(np.sum(data[seg_start:seg_end, telecommand_idx])))

        if self.use_shapelets:
            shapelet_features = self.shapelet_miner._transform(
                X = values.reshape(1, 1, -1).astype(np.float32)
            ).to_numpy().flatten()
            segment.extend(list(shapelet_features))
  
        results.append(segment)
        if event == -1 and len(masks)>0:
            if train:
                index = masks[mask_index][1] + 1
            else:
                index = masks[mask_index][0] + 1
        index += self.step_duration

    return results



cpdef tuple _apply_kernel_univariate2(np.ndarray[np.float32_t, ndim=1] X,
                                      np.ndarray[np.float32_t, ndim=1] weights,
                                      int length, float bias, int dilation, int padding):
    """
    Applica un kernel univariato a una serie temporale X (float32) e restituisce 
    il massimo valore di attivazione (convoluzione) normalizzato per la lunghezza.
    
    In altre parole, per ogni finestra (con dilatazione e padding),
    viene calcolato:
        _sum = bias + dot(weights, window_norm)
    Dove window_norm è la finestra normalizzata (media 0, std 1).
    La funzione ritorna il massimo _sum, diviso per il parametro 'length'.
    """
    cdef int n_timepoints = X.shape[0]
    cdef int end = (n_timepoints + padding) - ((length - 1) * dilation)
    cdef float _max = -3.4e38  # Valore iniziale molto basso
    cdef float _min = 3.4e38  # Valore iniziale molto basso
    cdef int i, j, idx
    cdef float _sum, m, std, dot_val, diff
    cdef int stride = max(1, int(length * 0.02))

    # Allocazione delle finestre
    cdef np.ndarray[np.float32_t, ndim=1] window = np.empty(length, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] window_norm = np.empty(length, dtype=np.float32)

    for i in range(-padding, end, stride):
        _sum = bias
        idx = i
        # Estrazione della finestra con dilatazione
        for j in range(length):
            if idx >= 0 and idx < n_timepoints:
                window[j] = X[idx]
            else:
                window[j] = 0.0
            idx += dilation

        # Calcolo della media della finestra
        m = 0.0
        for j in range(length):
            m += window[j]
        m /= length

        # Calcolo della deviazione standard della finestra
        std = 0.0
        for j in range(length):
            diff = window[j] - m
            std += diff * diff
        std = sqrt(std / length)
        if std < 1e-8:
            std = 1e-8

        # Normalizzazione della finestra
        for j in range(length):
            window_norm[j] = (window[j] - m) / std

        # Calcolo del prodotto scalare (convoluzione)
        dot_val = 0.0
        for j in range(length):
            dot_val += weights[j] * window_norm[j]
        _sum = bias + dot_val

        # Aggiorna il massimo se necessario
        if _sum > _max:
            _max = _sum
        if _sum < _min:
            _min = _sum

    return np.float32(_max) / length, np.float32(_min) / length



cpdef tuple _apply_kernel_multivariate(np.ndarray[np.float32_t, ndim=2] X,
                                         np.ndarray[np.float32_t, ndim=2] weights,
                                         int length, float bias, int dilation, int padding,
                                         int num_channel_indices,
                                         np.ndarray[np.int32_t, ndim=1] channel_indices):
    """
    Applica un kernel multivariato a una serie temporale multicanale X.
    
    Parametri:
      - X: array 2D di forma (n_channels, n_timepoints) contenente la serie temporale.
      - weights: array 2D dei pesi del kernel, di forma (num_channel_indices, length)
      - length: lunghezza del kernel.
      - bias: bias da aggiungere.
      - dilation: fattore di dilatazione.
      - padding: padding applicato alla serie.
      - num_channel_indices: numero di canali usati dal kernel.
      - channel_indices: array di indici dei canali da usare.
    
    Ritorna:
      Una tupla contenente:
        (proporzione di attivazioni positive, valore massimo ottenuto)
    """
    cdef int n_columns = X.shape[0]
    cdef int n_timepoints = X.shape[1]

    cdef int output_length = (n_timepoints + 2 * padding) - ((length - 1) * dilation)
    cdef int _ppv = 0
    cdef double _max = -1e300  # Un valore molto basso
    cdef int end = (n_timepoints + padding) - ((length - 1) * dilation)
    cdef int i, j, k, index, idx_anom
    cdef double _sum
    cdef int stride = max(1, int(length * 0.02))

    for i in range(-padding, end, stride):
        _sum = bias
        index = i
        for j in range(length):
            if index >= 0 and index < n_timepoints:
                for k in range(num_channel_indices):
                    _sum += weights[k, j] * X[channel_indices[k], index]
            index += dilation
        if _sum > _max:
            _max = _sum
        if _sum > 0:
            _ppv += 1

    return np.float32(_ppv / output_length), np.float32(_max)

cpdef np.ndarray[np.float32_t, ndim=2] _apply_kernels2(np.ndarray[np.float32_t, ndim=3] X, object kernels):
    """
    Applica i kernel a ciascuna istanza delle serie temporali in X.
    
    Parametri:
      - X: array 3D di forma (n_instances, n_channels, series_length)
      - kernels: tupla contenente i parametri dei kernel:
          (weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices)
    
    Ritorna:
      Un array 2D di forma (n_instances, num_kernels*2) con 2 feature per kernel.
    """

    # Unpack dei kernel e conversione ai tipi appropriati
    cdef np.ndarray[np.float32_t, ndim=1] weights = np.asarray(kernels[0], dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] lengths = np.asarray(kernels[1], dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] biases = np.asarray(kernels[2], dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] dilations = np.asarray(kernels[3], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] paddings = np.asarray(kernels[4], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] num_channel_indices = np.asarray(kernels[5], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] channel_indices = np.asarray(kernels[6], dtype=np.int32)
    
    cdef int n_instances = X.shape[0]
    cdef int num_kernels = lengths.shape[0]
    
    cdef np.ndarray[np.float32_t, ndim=2] _X = np.zeros((n_instances, num_kernels * 2), dtype=np.float32)
    cdef int i, j
    cdef int a1, a2, a3, b1, b2, b3
    cdef np.ndarray[np.float32_t, ndim=2] _weights = None  # Dichiarazione fuori dal blocco

    for i in range(n_instances):  # Uso di range invece di prange
       
        a1 = 0
        a2 = 0
        a3 = 0
        for j in range(num_kernels):         
            b1 = a1 + num_channel_indices[j] * lengths[j]
            b2 = a2 + num_channel_indices[j]
            b3 = a3 + 2
            if num_channel_indices[j] == 1:
                _X[i, a3:b3] = _apply_kernel_univariate2(
                    X[i, channel_indices[a2]],
                    weights[a1:b1],
                    lengths[j],
                    0,
                    dilations[j],
                    paddings[j],
                )
            else:
                _weights = weights[a1:b1].reshape((num_channel_indices[j], lengths[j]))
                _X[i, a3:b3] = _apply_kernel_multivariate(
                    X[i],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    num_channel_indices[j],
                    channel_indices[a2:b2],
                )
            a1 = b1
            a2 = b2
            a3 = b3
    return _X.astype(np.float32)