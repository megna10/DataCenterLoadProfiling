import pandas as pd
import numpy as np
from scipy.optimize import nnls
from tqdm import tqdm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import random


SERVER_CONFIGS = {

        "Standard": {'P_idle': 1770.0, 'P_peak': 11033.0, 'gpus_per_server': 8},
        'Dense': {'P_idle': 1833.0, 'P_peak': 11154.0, 'gpus_per_server': 8},
        'Extreme': {'P_idle': 1369, 'P_peak': 11895.0, 'gpus_per_server': 8},
    }

def load_machine_metric():
    return pd.read_csv("ai_training_files/metric_clean.csv")

# filter 8-GPU machines

def load_machine_spec():
    return pd.read_csv("ai_training_files/8_gpu_machine_ids.csv")


def load_cluster_gpu_profile():
    return pd.read_csv(
        "ai_training_files/cluster_gpu_profile.csv"
    )

def create_8gpu_metric_csv():
    # df_metric = load_machine_metric()
    # df_machine = load_machine_spec()

    # machine_ids = df_machine["machine_id"].unique()

    # print("Original metric rows:", len(df_metric))
    # print("8-GPU machines:", len(machine_ids))

    # df_8gpu_metric = df_metric[
    #     df_metric["machine_id"].isin(machine_ids)
    # ].copy()

    # print("Filtered metric rows:", len(df_8gpu_metric))

    # output_path = "ai_training_files/metric_8gpu.csv"

    # df_8gpu_metric.to_csv(output_path, index=False)

    # print("Saved:", output_path)

    return pd.read_csv("ai_training_files/metric_8gpu.csv")

def create_gpu_profile_csv():

    df = pd.read_csv("ai_training_files/metric_8gpu.csv")

    bin_all = []

    for mid in df["machine_id"].unique():
        bins = extract_single_machine_gpu_utilizations(df, mid)
        bins.name = mid
        bin_all.append(bins)

    bin_df = pd.concat(bin_all, axis=1)

    cluster_bin_avg = bin_df.mean(axis=1)

    gpu_profile = pd.DataFrame({
        "bin": range(288),
        "machine_gpu": cluster_bin_avg.values
    })

    gpu_profile.to_csv(
        "ai_training_files/cluster_gpu_profile.csv",
        index=False
    )

    print("Saved cluster GPU profile")

def extract_single_machine_gpu_utilizations(df, machine_id):

    df_m = df[df["machine_id"] == machine_id].copy()

    # Drop rows that don't have GPU data (must have at least 8 columns)
    df_m = df_m.dropna(axis=0, thresh=8)

    if df_m.empty:
        return pd.Series([0] * 288)

    df_m = df_m.sort_values("start_time")

    # 24 hour window (relative time)
    start = df_m["start_time"].min()
    end = start + 24 * 3600

    df_m = df_m[(df_m["start_time"] >= start) & (df_m["start_time"] < end)].copy()

    if df_m.empty:
        return pd.Series([0] * 288)

    df_m["timestamp"] = df_m["start_time"] - start

    # gives us 288 bins cause 288 5 min bins in 24 hours
    df_m["bin"] = df_m["timestamp"] // 300


    # Ensure 0–23 hours exist
    # takes mean gpu utilization for that bin
    bins = df_m.groupby("bin")["machine_gpu"].mean()
    bins = bins.reindex(range(288)).interpolate().fillna(0)

    return bins

def load_training_8gpu():
    df_8gpu = load_machine_spec()
    training_ids = load_training_tasks()

    # filters for machine_ids present in training_ids
    common_8gpu_df = df_8gpu[df_8gpu["machine_id"].isin(training_ids)].copy()

    # drop duplicates to only keep unique machine ids
    common_8gpu_df = common_8gpu_df.drop_duplicates(subset=["machine_id"])

    output_path = "ai_training_files/common_8gpu_machines.csv"

    common_8gpu_df.to_csv(output_path, index=False)

def compute_cluster_hourly():
    df_8gpu_metric = create_8gpu_metric_csv()
    bin_all = []

    for mid in df_8gpu_metric["machine_id"].unique():
        bins = extract_single_machine_gpu_utilizations(df_8gpu_metric, mid)
        bin_all.append(bins)

    # combine all machines into a dataframe
    bin_df = pd.concat(bin_all, axis=1)

    # cluster wider average gpu utilization
    cluster_bin_avg = bin_df.mean(axis=1)


    print("Cluster bins:", len(cluster_bin_avg))
    print("Cluster average GPU:", cluster_bin_avg.mean())

    return cluster_bin_avg

def calculate_server_power(workload_profile, num_servers, selected_deployment):
    """
    Converts GPU workload utilization into total GPU server IT power.
    """

    spec = SERVER_CONFIGS[selected_deployment]
    df = workload_profile.copy()

    # in decimal point
    df["per_gpu_util"] = df["machine_gpu"] / (8 *100)

    df["utilization"] = df["per_gpu_util"].clip(0.0, 1.0)

    # Linear power model
    p_idle = spec["P_idle"]
    p_dynamic = spec["P_peak"] - spec["P_idle"]

    df["server_power_kw"] = (p_idle + p_dynamic * df["utilization"]) / 1000.0

    # Power of entire cluster
    df["it_power_kw"] = (num_servers * df["server_power_kw"])

    df["bin_start"] = [i * 300 for i in range(len(df))]

    return df[
        [
            "bin_start" , # seconds from start
            "utilization",
            "server_power_kw",
            "it_power_kw",
        ]
    ]

def calculate_power_profile(selected_deployment, num_servers):

    cluster_bin_avg = load_cluster_gpu_profile()

    power_profile = calculate_server_power(
        cluster_bin_avg,
        num_servers,
        selected_deployment
    )

    power_profile["timestamp"] = power_profile["bin_start"]

    power_profile["hour"] = (
        power_profile["timestamp"] / 3600.0
    )

    return power_profile[
        [
            "timestamp",
            "hour",
            "utilization",
            "server_power_kw",
            "it_power_kw"
        ]
    ]


def main():
    """Main execution block where workflow functions are called."""

    # power_profile = calculate_power_profile(
    #     selected_deployment="Standard",
    #     num_servers=10
    # )

    # print(power_profile.head())
    # create_gpu_profile_csv()

# --- RUN SCRIPT ---

if __name__ == "__main__":
    main()
