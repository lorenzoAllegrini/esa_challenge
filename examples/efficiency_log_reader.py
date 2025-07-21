import pandas as pd
import os 

def read_efficiency_df(df):
    total_time = 0
    avg_cpu = 0
    avg_internal_time = 0
    avg_external_time = 0
    for _,row in df.iterrows():
        print(row["channel_time"])
        total_time += row["channel_time"] if row["channel_time"] > 0 else 0
        avg_internal_time += row["avg_internal_time"] if row["avg_internal_time"] > 0 else 0
        avg_external_time += row["avg_external_time"] if row["avg_external_time"] > 0 else 0
        avg_cpu += row["channel_cpu"] if row["channel_cpu"] > 0 else 0

    return {"total_time": total_time,
            "avg_channel_time": total_time/58,
            "avg_cpu": avg_cpu/58,
            "avg_internal_time": avg_internal_time/58,
            "avg_external_time": avg_external_time/58,
            }

df = pd.read_csv(os.path.join("experiments", "esa_training", "efficiency_log.csv"))
print(read_efficiency_df(df))