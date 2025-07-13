def number_of_peaks_finding(array):
    if array.size == 0:
        return 0
    prominence = 0.1 * (np.max(array) - np.min(array))
    peaks = sig.find_peaks(array, prominence=prominence)[0]
    return len(peaks)


def duration(df):
    t1 = pd.Timestamp(df.head(1).timestamp.values[0])
    t2 = pd.Timestamp(df.tail(1).timestamp.values[0])
    return (t2 - t1).seconds


def smooth10_n_peaks(array):
    kernel = np.ones(10) / 10
    array_convolved = np.convolve(array, kernel, mode="same")
    return number_of_peaks_finding(array_convolved)


def smooth20_n_peaks(array):
    kernel = np.ones(20) / 20
    array_convolved = np.convolve(array, kernel, mode="same")
    return number_of_peaks_finding(array_convolved)


def diff_peaks(array):
    array_diff = np.diff(array)
    return number_of_peaks_finding(array_diff)


def diff2_peaks(array):
    array_diff = np.diff(array, n=2)
    return number_of_peaks_finding(array_diff)


def diff_var(array):
    array_diff = np.diff(array)
    return np.var(array_diff)


def diff2_var(array):
    array_diff = np.diff(array, n=2)
    return np.var(array_diff)


import pandas as pd
import pymannkendall as mk  # Assicurati di avere installato pymannkendall
from scipy.stats import (
    linregress,
    spearmanr,
)


def deviation_from_expected_sp_array(values, sample_size=300, epsilon=1e-6):
    values = np.array(values)
    n = len(values)
    # Se l'array è molto grande, campiona per il calcolo della correlazione
    if n > sample_size:
        # Campiona i valori (la correlazione verrà calcolata su un campione)
        values_sample = np.random.choice(values, size=sample_size, replace=False)
        x = np.arange(sample_size)
    else:
        values_sample = values
        x = np.arange(n)

    # Calcola la correlazione di Spearman tra x e il campione di valori
    sp_corr, _ = spearmanr(x, values_sample)
    mean_value = np.mean(values)
    expected_value = mean_value * (1 + sp_corr)
    last_value = values[-1]
    return (last_value - expected_value) / (mean_value + epsilon)


def deviation_from_expected_mk_array(values, sample_size=300, epsilon=1e-6):

    values = np.array(values)
    n = len(values)
    if n > sample_size:
        values_sample = np.random.choice(values, size=sample_size, replace=False)
    else:
        values_sample = values

    # Esegui il test di Mann-Kendall sul campione per ottenere Tau
    result = mk.original_test(values_sample)
    tau = result.Tau
    mean_value = np.mean(values)
    expected_value = mean_value * (1 + tau)
    last_value = values[-1]
    return (last_value - expected_value) / (mean_value + epsilon)


def value_over_median_array(values, sample_size=300):

    values = np.array(values)
    n = len(values)
    if n > sample_size:
        values_sample = np.random.choice(values, size=sample_size, replace=False)
    else:
        values_sample = values
    med = np.median(values_sample)
    if med == 0:
        return 0
    return values[-1] / med


def deviation_from_slope_array(values, sample_size=300, epsilon=1e-6):

    values = np.array(values)
    n = len(values)
    # Per la regressione, se l'array è molto grande, campiona gli indici in modo ordinato
    if n > sample_size:
        indices = np.sort(np.random.choice(n, size=sample_size, replace=False))
        sampled_values = values[indices]
        x = indices
    else:
        x = np.arange(n)
        sampled_values = values

    slope, intercept, _, _, _ = linregress(x, sampled_values)
    predicted_value = intercept + slope * (n - 1)
    mean_value = np.mean(values)
    last_value = values[-1]
    return (last_value - predicted_value) / (mean_value + epsilon)


def gaps_squared(df, sample_size=300):
    if len(values) > sample_size:
        values = np.random.choice(values, size=sample_size, replace=False)
    df = df.copy()
    # df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp2"] = df["timestamp"].shift(1)
    df = df.reset_index().iloc[1:, :]
    df["time_delta"] = (df.timestamp - df.timestamp2).dt.seconds
    df["time_delta_squared"] = df["time_delta"] ** 2
    return df.time_delta_squared.sum()


def calculate_slope(values, sample_size=300):
    if len(values) > sample_size:
        values = np.random.choice(values, size=sample_size, replace=False)
    x = np.arange(len(values))
    slope, _, _, _, _ = linregress(x, values)
    return slope


def spearman_correlation(values, sample_size=300):
    if len(values) > sample_size:
        values = np.random.choice(values, size=sample_size, replace=False)
    x = np.arange(len(values))
    correlation, _ = spearmanr(x, values)
    return correlation


def mann_kendall_test(values, sample_size=300):
    if len(values) < 2:
        return 0
    if len(values) > sample_size:
        values = np.random.choice(values, size=sample_size, replace=False)
    result = mk.original_test(values)
    return result.slope


def mann_kendall_test_tau(values, sample_size=300):
    if len(values) > sample_size:
        values = np.random.choice(values, size=sample_size, replace=False)
    result = mk.original_test(values)
    return result.Tau


def deviation_from_expected(df, epsilon=1e-6):

    required_columns = ["sp_correlation", "mk_tau", "slope", "mean", "value"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Le seguenti colonne sono mancanti nel DataFrame: {missing_columns}"
        )

    # Converte le colonne in valori numerici, sostituendo errori con 0
    df["sp_correlation"] = pd.to_numeric(df["sp_correlation"], errors="coerce").fillna(
        0
    )
    df["mk_tau"] = pd.to_numeric(df["mk_tau"], errors="coerce").fillna(0)
    df["slope"] = pd.to_numeric(df["slope"], errors="coerce").fillna(0)
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce").fillna(0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    # Calcola il valore atteso e la deviazione per sp
    df["expected_value_sp"] = df["mean"] * (1 + df["sp_correlation"])
    df["deviation_from_expected_sp"] = (df["value"] - df["expected_value_sp"]) / (
        df["mean"] + epsilon
    )

    # Calcola il valore atteso e la deviazione per il tau del test MK
    df["expected_value_mk"] = df["mean"] * (1 + df["mk_tau"])
    df["deviation_from_expected_mk"] = (df["value"] - df["expected_value_mk"]) / (
        df["mean"] + epsilon
    )

    # Calcola il valore atteso e la deviazione per lo slope
    df["expected_value_slope"] = df["mean"] * (1 + df["slope"])
    df["deviation_from_expected_slope"] = (df["value"] - df["expected_value_slope"]) / (
        df["mean"] + epsilon
    )

    return df


def deviation_from_expected_mk(value, tau, mean, epsilon=1e-6):
    expected_value = mean * (1 + tau)
    return (value - expected_value) / (mean + epsilon)


def moving_average_prediction_error(values, epsilon=1e-6):
    values = np.array(values)
    n = len(values)

    if n == 0:
        return 0  # Se il vettore è vuoto, restituiamo 0 per evitare errori

    window_size = max(1, int(0.1 * n))  # Assicuriamoci che sia almeno 1

    if window_size > n:
        window_size = n  # Se la finestra è più grande dell'array, la limitiamo

    # Controllo se `values` è troppo corto per la convoluzione
    if n < window_size:
        return 0  #

    # Calcolo della media mobile
    y_ma = np.convolve(values, np.ones(window_size) / window_size, mode="valid")[-1]

    y_real = values[-1]

    error_ma = np.abs(y_real - y_ma)

    # Calcoliamo la media dell'errore per normalizzarlo
    mean_value = np.mean(values)

    return np.mean(error_ma) / (mean_value + epsilon)


from scipy.signal import stft


def stft_spectral_std(values, nperseg=80, epsilon=1e-6):
    values = np.array(values)
    if len(values) < 2:
        return 0

    # Applichiamo la STFT con una finestra di lunghezza nperseg
    _, _, Zxx = stft(values, nperseg=max(1, len(values) - 1))

    # Calcoliamo il modulo dello spettrogramma
    spectrogram_magnitude = np.abs(Zxx)

    # Sommiamo tutte le componenti di frequenza per ciascun istante di tempo
    summed_spectrum = np.sum(spectrogram_magnitude, axis=0)

    # Calcoliamo la deviazione standard del segnale sommato nel tempo
    std_spectrum = np.std(summed_spectrum)

    # Normalizziamo rispetto alla media della serie per stabilità numerica
    mean_value = np.mean(values)
    return std_spectrum / (mean_value + epsilon)


def autoregressive_deviation(values, lag=3, epsilon=1e-6):
    values = np.array(values)
    n = len(values)

    # Controlliamo che ci siano abbastanza dati per il modello AR
    min_required_samples = max(lag + 1, 4)  # Assicuriamo almeno 4 dati disponibili
    if n < min_required_samples:
        return 0  # Se i dati sono insufficienti, restituiamo 0

    # Adattiamo il lag se necessario per evitare errori
    lag = min(lag, n - 1)

    try:
        # Addestriamo il modello AR con tutti i valori tranne l'ultimo
        model = AutoReg(values[:-1], lags=lag)
        model_fitted = model.fit()

        # Predizione del valore successivo
        predicted_value = model_fitted.predict(
            start=len(values) - 1, end=len(values) - 1
        )[0]

    except Exception as e:
        # print(f"Errore nel modello AR: {e}")
        return 0  # Se il modello fallisce, restituiamo 0

    # Valore reale dell'ultimo punto della serie
    last_real_value = values[-1]

    # Calcolo della deviazione normalizzata rispetto alla media della serie
    mean_value = np.mean(values[:-1])  # Media senza l'ultimo valore
    deviation = np.abs(last_real_value - predicted_value) / (mean_value + epsilon)

    return deviation


def spectral_energy(values, nperseg=80):
    _, _, Zxx = stft(values, nperseg=len(values))
    return np.sum(np.abs(Zxx))


def dominant_frequency(values, nperseg=80):
    f, _, Zxx = stft(values, nperseg=len(values))
    spectrogram_magnitude = np.abs(Zxx)
    energy_per_freq = np.sum(
        spectrogram_magnitude, axis=1
    )  # Somma sulle finestre temporali
    return f[np.argmax(energy_per_freq)]  # Restituisce la frequenza dominante


def spectral_slope(values, nperseg=80):
    f, _, Zxx = stft(values, nperseg=len(values))
    magnitudes = np.abs(Zxx).mean(axis=1)  # Media sulle finestre temporali
    slope, _, _, _, _ = linregress(f, magnitudes)
    return slope


def wavelet_features(values, wavelet_name="db4", levels=3):
    """
    Calcola MAX e PPV per ogni livello di decomposizione wavelet su una serie temporale,
    adattando automaticamente il numero di livelli per evitare boundary effects.

    Args:
    - values (array): Segmento di serie temporale.
    - wavelet_name (str): Nome della wavelet Daubechies (es. 'db4').
    - levels (int): Numero massimo di livelli di decomposizione.

    Returns:
    - dict: Contiene MAX e PPV per ogni livello.
    """
    values = np.array(values)

    # Calcola il numero massimo di livelli sicuro per evitare boundary effects
    max_levels = pywt.dwt_max_level(len(values), pywt.Wavelet(wavelet_name).dec_len)
    levels = min(
        levels, max_levels
    )  # Usa il minimo tra quello richiesto e quello massimo possibile

    coeffs = pywt.wavedec(values, wavelet=wavelet_name, level=levels)

    # Dizionario per le feature
    features = {"max": [], "ppv": []}

    for c in coeffs:
        if len(c) == 0:
            features["max"].append(0)
            features["ppv"].append(0)
        else:
            features["max"].append(np.max(c))
            features["ppv"].append(np.sum(c > 0) / len(c) if len(c) > 0 else 0)

    return features
