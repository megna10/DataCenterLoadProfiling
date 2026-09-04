import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#defines  hardware configs (Adjust idle/peak Watts anytime)
HARDWARE_TIERS = {
    'Standard': {
        'cores_per_node': 32,
        'p_idle': 199.0,  # Watts
        'p_peak': 834.0,  # Watts
    },
    'Dense': {
        'cores_per_node': 128,
        'p_idle': 431.0,
        'p_peak': 1793.0,
    },
    'Extreme': {
        'cores_per_node': 256,
        'p_idle': 579.0,
        'p_peak': 2340.0,
    },
}


def simulate_cluster_power(
    data, config_key, num_servers=100, bin_seconds=300
):

    server_config = HARDWARE_TIERS[config_key]
    results = data.copy()

    # gets max execution capacity in core-seconds in one server
    node_capacity_sec = server_config['cores_per_node'] * bin_seconds

    # Total Cluster Capacity (N * Capacity_unit)
    total_cluster_capacity_sec = num_servers * node_capacity_sec

    # Utilization = min(100%, Workload in seconds/ Capacity_Total)
    results['utilization_pct'] = np.minimum(
        100.0,
        (results['total_work_core_seconds'] / total_cluster_capacity_sec) * 100.0,
    )

    # Per-Node Power Draw (Watts)
    u_frac = results['utilization_pct'] / 100.0
    p_dynamic = server_config['p_peak'] - server_config['p_idle']
    node_power_w = server_config['p_idle'] + (p_dynamic * u_frac)

    # Total Cluster Electrical Power (kW) = N * Node_Power / 1000
    results['cluster_power_kw'] = (num_servers * node_power_w) / 1000.0

    # Map time axis to Hours (0.0 to 24.0)
    start_offset = results['bin_start_sec'].min()

    results['hour_of_day'] = (results['bin_start_sec'] - start_offset) / 3600.0

    return results

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
    df = pd.read_csv('analytic_data_2.csv')

    sim_results = simulate_cluster_power(df, config_key='Dense', num_servers=100)
    plot_power_profile(sim_results)


if __name__ == "__main__":
    main()
