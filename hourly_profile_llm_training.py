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
    return pd.read_csv("ai_training_files/8_gpu_machine_ids.csv")["machine_id"].unique()

def load_training_tasks():
    columns = [
        "job_name",
        "task_name",
        "worker_name",
        "inst_id",
        "machine",
        "gpu_name",
        "cpu_usage",
        "gpu_wrk_util",
        "avg_mem",
        "max_mem",
        "avg_gpu_wrk_mem",
        "max_gpu_wrk_mem",
        "read",
        "write",
        "read_count",
        "write_count",
    ]

    training_tasks = [
    'tensorflow', 'worker', 'PyTorchWorker', 'xComputeWorker', 'ps',
    'chief', 'MWorker', 'TfClientWorker', 'TransformGraph',
    'aligraph', 'BladeMain', 'M1', 'M2', 'R3_1_2', 'R2_1'
]
    df = pd.read_csv("ai_training_files/pai_sensor_table.csv", header=None, names=columns)

    df_train = df[df["task_name"].isin(training_tasks)]
    training_machine_ids = df_train["machine"].unique()

    print(len(training_machine_ids))
    print(training_machine_ids)

def extract_single_machine_gpu_utilizations(df, machine_id):

    df_m = df[df["machine_id"] == machine_id].copy()

    # Drop rows that don't have GPU data (must have at least 8 columns)
    df_m = df_m.dropna(axis=0, thresh=8)

    if df.empty:
        return pd.Series([0]*24)

    df_m = df_m.sort_values("start_time")

    # 24 hour window (relative time)
    start = df_m["start_time"].min()
    end = start + 24 * 3600

    df_m = df_m[(df_m["start_time"] >= start) & (df_m["start_time"] < end)].copy()

    if df_m.empty:
        return pd.Series([0]*24)


    df_m["timestamp"] = df_m["start_time"] - start

    # gives us 288 bins cause 288 5 min bins in 24 hours
    df_m["bin"] = df_m["timestamp"] // 300


    # Ensure 0–23 hours exist
    # takes mean gpu utilization for that bin
    bins = df_m.groupby("bin")["machine_gpu"].mean()
    bins = bins.reindex(range(288)).interpolate().fillna(0)

    return bins


def compute_cluster_hourly():
    # load metric file
    df_metric = load_machine_metric()

    # load 8-GPU machine IDs
    machine_ids_8gpu = load_machine_spec()

    # filter metric file to only those machines
    df_metric = df_metric[df_metric["machine_id"].isin(machine_ids_8gpu)]

    bin_all = []

    for mid in machine_ids_8gpu:
        bins = extract_single_machine_gpu_utilizations(df_metric, mid)
        bin_all.append(bins)

    # combine all machines into a dataframe
    bin_df = pd.concat(bin_all, axis=1)

    # cluster wider average gpu utilization
    cluster_bin_avg = bin_df.mean(axis=1)

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

    df["bin_start"] = [i * 300 for i in range(288)]

    return df[
        [
            "bin_start" , # seconds from start
            "utilization",
            "server_power_kw",
            "it_power_kw",
        ]
    ]

def calculate_power_profile(selected_deployment, num_servers):

    cluster_bin_avg = compute_cluster_hourly()

    workload_profile = pd.DataFrame({
        "bin": range(288),
        "machine_gpu": cluster_bin_avg.values,
    })

    power_profile = calculate_server_power(workload_profile, num_servers, selected_deployment)

    power_profile["timestamp"] = (power_profile["bin_start"])

    power_profile["hour"] = (
    (power_profile["timestamp"] - power_profile["timestamp"].min()) / 3600.0
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

    load_training_tasks()

# --- RUN SCRIPT ---

if __name__ == "__main__":
    main()
