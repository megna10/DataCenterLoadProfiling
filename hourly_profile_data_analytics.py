import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SERVER HARDWARE CONFIGURATION
# =============================================================================
# Defines the hardware characteristics and power consumption for each
# server deployment type.
#
# cores_per_node:
#     Number of CPU cores available on one server.
#
# P_idle:
#     Power consumed by one server when it is idle, in Watts.
#
# P_peak:
#     Maximum power consumed by one server at full utilization, in Watts.
#
# These values can be adjusted if the hardware assumptions change.
# =============================================================================

SERVER_CONFIG = {
    'Standard': {
        'cores_per_node': 32,
        'P_idle': 199.0,  # Watts
        'P_peak': 834.0,  # Watts
    },
    'Dense': {
        'cores_per_node': 128,
        'P_idle': 431.0,
        'P_peak': 1793.0,
    },
    'Extreme': {
        'cores_per_node': 256,
        'P_idle': 579.0,
        'P_peak': 2340.0,
    },
}

def calculate_server_power(workload_profile, num_servers, selected_deployment):
    """
    Convert CPU utilization into server and total cluster IT power.

    Parameters
    ----------
    workload_profile : pandas.DataFrame, Dataframe containing the time-series workload information.

        Required columns:
            - timestamp
            - hour
            - utilization

    num_servers : int, Number of servers in the cluster.

    selected_deployment : str

    Returns
    -------
    pandas.DataFrame
        Time-series dataframe containing:
            - timestamp
            - hour
            - utilization
            - server_power_kw
            - it_power_kw
    """

    spec = SERVER_CONFIG[selected_deployment]
    df = workload_profile.copy()

    # Server power parameters
    p_idle = spec["P_idle"]
    p_dynamic = spec["P_peak"] - spec["P_idle"]

    # Calculate power of one server
    df["server_power_kw"] = (p_idle+ p_dynamic * df["utilization"]) / 1000.0

    # Calculate total cluster power
    df["it_power_kw"] = (num_servers * df["server_power_kw"])


    return df[
    [
        "timestamp",
        "hour",
        "utilization",
        "server_power_kw",
        "it_power_kw",
    ]
]

def calculate_power_profile(selected_deployment, num_servers=100, target_peak_util=.8):
    """
    Generate a cluster power profile from pre-aggregated CPU workload data.

    The input dataset contains the amount of CPU work performed during
    each time bin. This function converts that workload into a normalized
    utilization value and then converts utilization into electrical power.
    """

    data = pd.read_csv("data_analytics_files/analytic_data_2.csv").copy()

    # finds maxiumum workload demand in the dataset
    peak_trace_work = data["total_work_core_seconds"].max()

    data["relative_activity"] = data["total_work_core_seconds"] / peak_trace_work

    data["utilization"] = (data["relative_activity"] * target_peak_util).clip(0.0, 1.0)

    data["timestamp"] = data["bin_start_sec"]
    data["hour"] = (data["timestamp"] - data["timestamp"].min()) / 3600.0

    power_profile = calculate_server_power(
            data[["timestamp", "hour", "utilization"]],
            num_servers,
            selected_deployment
        )

    return power_profile[
        [
            "timestamp",
            "hour",
            "utilization",
            "server_power_kw",
            "it_power_kw",
        ]
    ]

# def plot_power_profile(sim_results):

    # Convert seconds to hours for a cleaner x-axis
    sim_results['time_hours'] = sim_results['bin_start_sec'] / 3600.0

    plt.figure(figsize=(12, 5))
    plt.plot(
        sim_results['time_hours'],
        sim_results['cluster_power_kw'],
        color='#1f77b4',
        marker='o',
        markersize=3,
        linewidth=2,
    )
    plt.title('Dense Tier (100x Dell R7725) - Electrical Power Footprint')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Total Cluster Power (kW)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    """Main execution block where workflow functions are called."""

    # loads bigquery import
    calculate_power_profile(selected_deployment="Standard", num_servers=100, target_peak_util=.8)


if __name__ == "__main__":
    main()
