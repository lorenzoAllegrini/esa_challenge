import pandas as pd
import os
import numpy as np
import time
from scipy.stats import percentileofscore
from spaceai.data import ESA,ESAMission,ESAMissions
import pandas as pd

def extract_feature_series(
    df: pd.DataFrame,
    feature: str,
    series_column_name: str = 'series_column_name',
    timestamp_column: str = 'end_time',
    label_column: str = 'anomaly'
) -> pd.DataFrame:
    """
    Estrae una serie temporale relativa a una specifica feature da un DataFrame segmentato,
    convertendo il timestamp in UTC e formattandolo in modo da terminare con "Z".

    Se non esiste 'label_column', viene creata con zero.
    Se non esiste 'series_column_name', viene creata con "serie_unica".
    Se non esiste 'timestamp_column', prova a usare 'timestamps'.

    Parametri:
        df: DataFrame contenente il dataset segmentato con le feature.
        feature: nome della feature da analizzare (es. "stft", "mean", ecc.).
        series_column_name: nome della colonna che identifica la serie (default "series_column_name").
        timestamp_column: nome della colonna che contiene il timestamp dell'ultimo timestep del segmento.
        label_column: nome della colonna contenente l'etichetta del segmento (default "anomaly").

    Ritorna:
        Un DataFrame con le colonne: series, timestamp, value, label.
        - "value" contiene il valore della feature specificata per ciascun segmento.
        - "timestamp" è una stringa ISO8601 con "Z" come suffisso (UTC).
        - La prima riga con label==1 diventa label=1, tutte le altre originariamente con label==1 diventano label=2.
    """
    if feature not in df.columns:
        raise ValueError(f"La feature '{feature}' non è presente nel DataFrame.")

    df = df.copy()

    # Se non esiste la colonna per la serie, la creiamo con un valore di default
    if series_column_name not in df.columns:
        df[series_column_name] = "serie_unica"

    # Gestione del timestamp: se la colonna specificata non esiste, uso 'timestamps'
    if timestamp_column not in df.columns:
        if 'timestamps' in df.columns:
            df[timestamp_column] = df['timestamps'].apply(
                lambda ts: ts[-1] if isinstance(ts, (list, tuple)) and len(ts) > 0 else None
            )
        else:
            df[timestamp_column] = pd.date_range(start="2020-01-01T00:00:00Z", periods=len(df), freq='s')
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='raise', utc=True)

    # Formatto il timestamp in ISO con "Z"
    df[timestamp_column] = df[timestamp_column].apply(lambda x: x.isoformat().replace('+00:00', 'Z'))

    # Se non esiste la colonna delle label, la creiamo impostandola a 0
    if label_column not in df.columns:
        df[label_column] = 0

    # Preparo il DataFrame di output selezionando e rinominando le colonne
    result_df = df[[series_column_name, timestamp_column, feature, label_column]].copy()
    result_df.rename(columns={
        series_column_name: 'series',
        feature: 'value',
        timestamp_column: 'timestamp',
        label_column: 'label'
    }, inplace=True)

    # Ordino per timestamp
    result_df.sort_values(by='timestamp', inplace=True)

    # Ridefinisco le label:
    #   - la prima riga con label==1 viene mantenuta a 1
    #   - tutte le altre righe originariamente con label==1 diventano 2
    first_found = False
    for idx in result_df.index:
        if result_df.at[idx, 'label'] == 1:
            if not first_found:
                result_df.at[idx, 'label'] = 1
                first_found = True
            else:
                result_df.at[idx, 'label'] = 2

    return result_df





def load_train_channel(channel_id: str):
    train_channel = ESA(
        root="datasets",
        mission=ESAMissions.MISSION_1.value,
        channel_id=channel_id,
        mode="challenge",
        overlapping=True,
        seq_length=1,
        n_predictions=1,
    )

    # 1. Costruzione del DataFrame dai dati ESA
    try:
        df = train_channel.data.to_pandas()
    except AttributeError:
        df = pd.DataFrame({channel_id: train_channel.data[:, 0]})

    if channel_id not in df.columns:
        raise KeyError(f"Colonna '{channel_id}' non trovata nel DataFrame restituito da ESA(...)")

    # 2. Garantisco la colonna 'end_time'
    if "end_time" not in df.columns and "timestamps" not in df.columns:
        df["end_time"] = pd.date_range(
            start="2020-01-01T00:00:00Z",
            periods=len(df),
            freq="s",  # usa "s" per evitare FutureWarning
            tz="UTC"
        )
    elif "end_time" not in df.columns and "timestamps" in df.columns:
        df["end_time"] = df["timestamps"].apply(
            lambda ts: ts[-1] if isinstance(ts, (list, tuple)) and len(ts) > 0 else pd.NaT
        )

    # 3. Aggiungo la colonna 'anomaly' inizialmente a zero
    df["anomaly"] = 0

    # 4. Recupero gli intervalli di anomalia come indici (non datetime)
    #    train_channel.anomalies è [[start_idx1, end_idx1], [start_idx2, end_idx2], ...]
    anomalies = getattr(train_channel, "anomalies", [])
    if anomalies:
        # Per ogni coppia [start_idx, end_idx], setto df.loc[start_idx:end_idx, "anomaly"] = 1
        for start_idx, end_idx in anomalies:
            # Clipping agli estremi validi di df.index
            i0 = max(0, int(start_idx))
            i1 = min(len(df) - 1, int(end_idx))
            df.loc[i0 : i1, "anomaly"] = 1


    # 5. Seleziono solo l'ultimo decimo dei dati
    start_idx = int(0.0 * len(df))
    end_idx = int(0.7 * len(df))
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    print(df.loc[:, "anomaly"].sum())
    return df


def load_challenge_channel(channel_id: str) -> pd.DataFrame:
    """
    Legge il parquet di challenge (test.parquet) e restituisce un DataFrame
    con una singola colonna corrispondente a channel_id, anziché una Series.
    """
    import pyarrow.parquet as pq
    source_folder = os.path.join("datasets", "ESA-Mission1-challenge")
    table = pq.read_table(os.path.join(source_folder, "test.parquet"))
    # Usare doppie parentesi per ottenere un DataFrame invece di una Series:
    df = table.to_pandas()[[channel_id]]
    return df

if __name__ == "__main__":
    # ----------------------------------------------------------------
    channel_id = "channel_48"
    feature_to_extract = channel_id
    test = True
    # ----------------------------------------------------------------

    if test:
        # 1) Carico il canale dal parquet
        df = load_challenge_channel(channel_id)

        # 2) Carico il CSV con le predizioni binarie
        labels_path = os.path.join(
            "experiments", "esa_competition", "predicted_binary_only.csv"
        )
        labels = pd.read_csv(labels_path)  # colonne: 'id', 'pred_binary'

        # 3) Definisco la prima id come punto di partenza
        start_id = int(labels["id"].min())

        # 4) Resetto l’indice di df per avere 0...N-1
        df = df.reset_index(drop=True)

        # 5) Costruisco un set di id anomali
        anomaly_ids = set(labels.loc[labels["pred_binary"] == 1, "id"])

        # 6) Creo la colonna 'anomaly' mappando ogni riga
        df["anomaly"] = df.index.map(
            lambda idx: 1 if (idx + start_id) in anomaly_ids else 0
        )

    else:
        # modalità “train”
        df = load_train_channel(channel_id)
        # Se vuoi, azzeri tutte le anomaly

    # A questo punto 'df' contiene:
    #   - la colonna channel_1 con i valori
    #   - end_time (o generated date_range)
    #   - anomaly = 0/1 come da predicted_binary_labels.csv
    print(df.head())

    # Continua poi con extract_feature_series ecc...
    df_feature = extract_feature_series(df, feature=feature_to_extract)
    # (Opzionale) Aggiunge anomaly metrics alla serie estratta
    # df_feature = add_anomaly_metrics(df_feature)

    # Stampa un riepilogo
    print(f"Serie estratta per {channel_id}, feature '{feature_to_extract}':")
    print(df_feature.head(10))
    print(df_feature.loc[df_feature["label"]==2])
    #df_feature.loc[int(1/10* len(df_feature)) : int(2/10* len(df_feature)) ]
    # Salva il risultato in un CSV
    output_filename = f"output_feature_series_{channel_id}.csv"
    df_feature.to_csv(output_filename, index=False)
    print(f"Salvato: {output_filename}")
