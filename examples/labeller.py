import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# === CONFIGURAZIONE ===
INPUT_PATH = "experiments/esa_competition/predicted_proba_labels.csv"
OUTPUT_PATH = "experiments/esa_competition/predicted_binary_labels.csv"
BINARY_ONLY_OUTPUT_PATH = "experiments/esa_competition/predicted_binary_only.csv"
PEAK_HEIGHT = 0.485
BUFFER_SIZE = 500

# === STEP 1: CARICAMENTO FILE ===
df = pd.read_csv(INPUT_PATH)
assert "id" in df.columns and "is_anomaly" in df.columns, "Il CSV deve contenere colonne 'id' e 'is_anomaly'"

# === STEP 2: ORDINAMENTO (se necessario) ===
df = df.sort_values("id").reset_index(drop=True)

# === STEP 3: TROVA PICCHI ===
y_full = df["is_anomaly"].values
peaks, _ = find_peaks(y_full, height=PEAK_HEIGHT)

# === STEP 4: COSTRUISCI ARRAY BINARIO CON BUFFER ===
y_binary = np.zeros_like(y_full, dtype=int)
for idx in peaks:
    a = max(0, idx - BUFFER_SIZE)
    b = min(len(y_full) - 1, idx + BUFFER_SIZE)
    y_binary[a:b + 1] = 1

# === STEP 5: AGGIUNGI AL DATAFRAME E SALVA ===
df["pred_binary"] = y_binary
df.to_csv(OUTPUT_PATH, index=False)
print(f"Predizioni binarie salvate in: {OUTPUT_PATH}")

# === STEP 6: CREA FILE SOLO CON ID E LABEL BINARIA ===
df_binary_only = df[["id", "pred_binary"]]
df_binary_only.to_csv(BINARY_ONLY_OUTPUT_PATH, index=False)
print(f"File semplificato (solo id e pred_binary) salvato in: {BINARY_ONLY_OUTPUT_PATH}")