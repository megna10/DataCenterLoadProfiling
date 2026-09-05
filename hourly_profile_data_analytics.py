import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#defines  hardware configs (Adjust idle/peak Watts anytime)
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
    Converts compute utilization into total server IT power.

    workload_profile must contain:
        - timestamp
        - utilization
    """
    spec = SERVER_CONFIG[selected_deployment]
    df = workload_profile.copy()

    # Server power parameters
    p_idle = spec["P_idle"]
    p_dynamic = spec["P_peak"] - spec["P_idle"]

    # Calculate power of one server
    df["server_power_kw"] = (
        p_idle
        + p_dynamic * df["utilization"]
    ) / 1000.0

    # Calculate total cluster power
    df["it_power_kw"] = (
        num_servers * df["server_power_kw"]
    )


    return df[
    [
        "timestamp",
        "utilization",
        "server_power_kw",
        "it_power_kw",
    ]
]

def calculate_power_profile(selected_deployment, num_servers=100):

    data = pd.read_csv("analytic_data_2.csv").copy()

    spec = SERVER_CONFIG[selected_deployment]

    bin_seconds = 300
    # gets max execution capacity in core-seconds in one server
    server_capacity_sec = spec['cores_per_node'] * bin_seconds

    # Total Cluster Capacity (N * Capacity_unit)
    total_cluster_capacity_sec = num_servers * server_capacity_sec

    # Utilization = min(100%, Workload in seconds/ Capacity_Total)

    data["utilization"] = (
        data["total_work_core_seconds"]
        / total_cluster_capacity_sec
    )

    # Prevent utilization from exceeding 100%
    data["utilization"] = data["utilization"].clip(
        0.0,
        1.0
    )

    power_profile = calculate_server_power(
        data[[
                "bin_start_sec",
                "utilization",
            ]].rename(
            columns={
                "bin_start_sec": "timestamp"
            }
        ),
        num_servers,
        selected_deployment,
    )

    # Add time information back
    power_profile["hour"] = (
        power_profile["timestamp"]
        - power_profile["timestamp"].min()
    ) / 3600.0

    return power_profile[
        [
            "timestamp",
            "hour",
            "utilization",
            "server_power_kw",
            "it_power_kw",
        ]
    ]

def plot_power_profile(sim_results):

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
    calculate_power_profile(selected_deployment="Standard", num_servers=100)


if __name__ == "__main__":
    main()
