import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# time in seconds modeling one day
START_TIME = 0
END_TIME = 86400#172800

SERVER_CONFIGS = {"Standard": {"P_idle": 307, "P_peak": 1133, "num_cpu": 2, "total_max_cores": 64},
                  "Dense": {"P_idle": 345, "P_peak": 1376, "num_cpu": 2, "total_max_cores": 128}
                  }

def get_target_vmid_table():
    int_active_vms = pd.read_csv('general_compute_files/interactive_4cores.csv')
    # grabs every entry in the first column & converts IDs into a set
    target_vm_ids = set(int_active_vms.iloc[:, 0])

    # fetch the azure dataset directly from github
    links_url = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetLinksV2.txt"

    # streams the sites directly without downloading the links to the desktop
    links_df = pd.read_csv(links_url, header=None, names=['url'])

    # slices lines 12 through 18 by row index
    target_files_df = links_df.iloc[11:18]

    # extract the urls into a python list of strings
    target_files = [url.strip() for url in target_files_df['url']]

    filtered_chunks = []
    for url in target_files:
        # read directly from cloud URL
        # trace format: [timestamp, vm id, min_cpu, max_cpu, avg_cpu]
        df = pd.read_csv(url,
                        header=None,
                        names=['timestamp', 'vm_id', 'min_cpu', 'max_cpu', 'avg_cpu'],
                        compression='infer')

        # keep only target vm_ids (data frame only keeps true values)
        df_filtered = df[df['vm_id'].isin(target_vm_ids)].copy()
        filtered_chunks.append(df_filtered)

    # combines all extracted data into one table
    extracted_data = pd.concat(filtered_chunks, ignore_index=True)
    extracted_data.to_csv('general_compute_files/target_vm_rows_1_7.csv', index=False)

    return extracted_data

def get_avg_cpu_utilizations(start_time, end_time, interval_seconds=300, num_vms=100,
    random_seed=42):
    """
    Creates a complete CPU utilization profile for the requested
    time period.

    Default:
        5-minute intervals
        24 hours = 288 data points
    """
    # Read the CPU utilization data
    df = pd.read_csv("general_compute_files/target_vm_rows_1_7.csv")

    # Get unique VM IDs
    vm_ids = df["vm_id"].unique()

    # Randomly select 100 VMs
    rng = np.random.default_rng(random_seed)

    selected_vm_ids = rng.choice(
        vm_ids,
        size=min(num_vms, len(vm_ids)),
        replace=False
    )

    print(f"Selected {len(selected_vm_ids)} VMs")

    # Keep ALL measurements belonging to those VMs
    df = df[df["vm_id"].isin(selected_vm_ids)].copy()

    # Only use timestamps within the requested range
    df = df[
        (df["timestamp"] >= start_time) &
        (df["timestamp"] <= end_time)
    ]

    timestamps = np.arange(
        start_time,
        end_time,
        interval_seconds
    )

    # Assign every observation to a 5-minute bucket
    df["timestamp_bin"] = (
        df["timestamp"] // interval_seconds
    ) * interval_seconds

    # Average CPU across all selected VMs/data points
    cpu_profile = (
        df.groupby("timestamp_bin")["avg_cpu"]
        .mean()
        .reindex(timestamps)
        .fillna(0)
        .reset_index()
    )

    cpu_profile.columns = [
        "timestamp",
        "overall_avg_cpu"
    ]

    return cpu_profile


def calculate_server_power(cpu_profile, num_servers, deployment):
    """
    converts cpu utilization into total server IT power
    """

    spec = SERVER_CONFIGS[deployment]
    df = cpu_profile.copy()

    # convert cpu utilization from percentage into a fraction
    df["utilization"] = df["overall_avg_cpu"] / 100.0

    # prevent utilization from going below 0 percent or above 100
    df["utilization"] = df["utilization"].clip(0.0, 1.0)

    # linear server power model
    p_idle = spec["P_idle"]
    p_dynamic = spec["P_peak"] - spec["P_idle"]

    df["server_power_kw"] = ( p_idle + p_dynamic * df["utilization"]) / 1000.0

    # Total cluster power
    df["it_power_kw"] = (num_servers * df["server_power_kw"])

    return df[["timestamp", "utilization", "server_power_kw", "it_power_kw"]]


def calculate_power_profile(selected_deployment, num_servers):

    cpu_profile = get_avg_cpu_utilizations( START_TIME, END_TIME, 300)

    expected_timestamps = np.arange(START_TIME, END_TIME, 300)

    full_profile = pd.DataFrame({
        "timestamp": expected_timestamps
    })

    # Match actual CPU measurements onto the full timeline
    full_profile = full_profile.merge(
        cpu_profile,
        on="timestamp",
        how="left"
    )

    # No workload = 0% utilization
    full_profile["overall_avg_cpu"] = (
        full_profile["overall_avg_cpu"]
        .fillna(0)
    )

    power_profile = calculate_server_power(full_profile, num_servers, selected_deployment)

    result = power_profile.copy()

    result["hour"] = (
        result["timestamp"]
        - result["timestamp"].min()
    ) / 3600.0

    return result[
        [
            "timestamp",
            "hour",
            "utilization",
            "server_power_kw",
            "it_power_kw",
        ]
    ]


# ------------- Plotting --------------------------

# def plot_profiles(profile_df, interval_num):
#     """ Plot the main CPU load curve"""

#     plt.plot(
#         profile_df[f'interval_{interval_num}m'],
#         profile_df['avg_cpu'],
#         color='#1f77b4',
#         linewidth=2,
#         marker='o',
#         markersize=3,
#         label=f'{interval_num}-Min Avg CPU Utilization'
#     )

#     # Format the X-Axis to display time ticks every 2 hours (every 8 15-min intervals)
#     total_bins = (24 * 60) // interval_num
#     bins_per_hour = 60 // interval_num          # e.g., 12 for 5-min, 4 for 15-min
#     tick_step = bins_per_hour * 2

#     tick_intervals = range(0, total_bins + 1, tick_step)
#     tick_labels = [f"{(i * interval_num) // 60:02d}:00" for i in tick_intervals]

#     plt.xticks(ticks=tick_intervals, labels=tick_labels, rotation=0)
#     plt.xlim(0, total_bins - 1)

#     # Axis labels & Title
#     plt.title(f"24-Hour Azure VM CPU Load Profile ({interval_num}-Minute Resolution)", fontsize=14, pad=15)
#     plt.xlabel("Time of Day (HH:MM)", fontsize=11, labelpad=10)
#     plt.ylabel("Average CPU Utilization (%)", fontsize=11, labelpad=10)

#     # Styling details
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(loc='upper right')
#     plt.tight_layout()

#     # Display the plot
#     plt.show()

# def plot_overall_cpu_util_sections(section_results):
#     plt.figure(figsize=(12, 6))

#     plt.plot(
#         section_results["timestamp"],
#         section_results["overall_avg_cpu"],
#         linewidth=2,
#         marker="o",
#         markersize=3
#     )


#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Average CPU Utilization (%)")
#     plt.title("Overall CPU Utilization Over Time")

#     plt.grid(True)
#     plt.show()


def main():
    """Main execution block where workflow functions are called."""

    power_profile = calculate_power_profile(
        selected_deployment="Standard",
        num_servers=10
    )

if __name__ == "__main__":
    main()
